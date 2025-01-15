#####################################################################
# Add these lines BEFORE other Panda3D imports to silence logs:
#####################################################################
import logging

# Silence "Loading model ..." logs from your python code
logging.getLogger("action_extractor.megapose.action_identifier_megapose").setLevel(logging.WARNING)

import panda3d.core as p3d
# Suppress "Known pipe types", "Xlib extension missing", etc.
p3d.load_prc_file_data("", "notify-level-glxdisplay fatal\n")
p3d.load_prc_file_data("", "notify-level-x11display fatal\n")
p3d.load_prc_file_data("", "notify-level-gsg fatal\n")

#####################################################################
# Normal imports
#####################################################################
import os
import h5py
import numpy as np
import torch
from tqdm import tqdm
import imageio
import cv2
import zarr
from glob import glob
from einops import rearrange
from zarr import DirectoryStore, ZipStore
import shutil

from action_extractor.action_identifier import load_action_identifier, VariationalEncoder
from action_extractor.utils.dataset_utils import (
    hdf5_to_zarr_parallel_with_progress,
    preprocess_data_parallel,
    directorystore_to_zarr_zip,
    frontview_K, frontview_R, sideview_K, sideview_R,
    agentview_K, agentview_R, sideagentview_K, sideagentview_R,
    pose_inv
)
import robomimic.utils.obs_utils as ObsUtils
from robomimic.utils.file_utils import get_env_metadata_from_dataset
from robomimic.utils.env_utils import create_env_from_metadata

from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder

from pathlib import Path
from megapose.utils.logging import get_logger
logger = get_logger(__name__)

from robosuite.utils.camera_utils import get_real_depth_map, get_camera_extrinsic_matrix, get_camera_intrinsic_matrix

#####################################################################
# Instead of from action_extractor.megapose.action_identifier_megapose 
# we STILL import the same run_inference_on_data, but we will pass 
# our pre-loaded model to it so it doesn't re-load. We'll define 
# a small wrapper or local approach for that.
#####################################################################
from action_extractor.megapose.action_identifier_megapose import (
    make_detections_from_object_data,
    make_object_dataset_from_folder,
    estimate_pose
)

def combine_videos_quadrants(top_left_video_path, top_right_video_path, bottom_left_video_path, bottom_right_video_path, output_path):
    import imageio

    top_left_reader = imageio.get_reader(top_left_video_path)
    top_right_reader = imageio.get_reader(top_right_video_path)
    bottom_left_reader = imageio.get_reader(bottom_left_video_path)
    bottom_right_reader = imageio.get_reader(bottom_right_video_path)

    fps = top_left_reader.get_meta_data()["fps"]  # or handle if they differ

    with imageio.get_writer(output_path, fps=fps) as writer:
        while True:
            try:
                tl_frame = top_left_reader.get_next_data()
                tr_frame = top_right_reader.get_next_data()
                bl_frame = bottom_left_reader.get_next_data()
                br_frame = bottom_right_reader.get_next_data()
            except (StopIteration, IndexError):
                # Means at least one reader had no more frames
                break

            # Combine frames in a quadrant layout
            top = np.hstack([tl_frame, tr_frame])
            bottom = np.hstack([bl_frame, br_frame])
            combined = np.vstack([top, bottom])

            writer.append_data(combined)

    top_left_reader.close()
    top_right_reader.close()
    bottom_left_reader.close()
    bottom_right_reader.close()


def quaternion_to_rotation_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y**2 + z**2),     2*(x*y - z*w),         2*(x*z + y*w)],
        [2*(x*y + z*w),           1 - 2*(x**2 + z**2),   2*(y*z - x*w)],
        [2*(x*z - y*w),           2*(y*z + x*w),         1 - 2*(x**2 + y**2)]
    ], dtype=np.float32)


def pose_vector_to_matrix(translation, quaternion):
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = quaternion_to_rotation_matrix(quaternion)
    T[:3, 3]  = translation
    return T


def convert_mp4_to_webp(input_path, output_path, quality=80):
    import subprocess
    import shutil
    
    ffmpeg_path = shutil.which('ffmpeg')
    if ffmpeg_path is None:
        raise RuntimeError("ffmpeg not found. Install via: sudo apt-get install ffmpeg")
    
    cmd = [
        ffmpeg_path, '-i', input_path,
        '-c:v', 'libwebp',
        '-quality', str(quality),
        '-lossless', '0',
        '-compression_level', '6',
        '-qmin', '0',
        '-qmax', '100',
        '-preset', 'default',
        '-loop', '0',
        '-vsync', '0',
        '-f', 'webp',
        output_path
    ]
    subprocess.run(cmd, check=True)
    

def find_green_bounding_box(img_array):
    """
    Given an RGB image (H x W x 3) with a green gripper,
    return a bounding box (x_min, y_min, x_max, y_max) 
    that encloses the largest green region,
    using a stricter threshold and morphological cleanup.
    """
    # 1. Convert from RGB to HSV
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    
    # 2. Define a narrower green color range in HSV
    #    Adjust these as needed for your specific shade of green.
    #    (H: 50–70, S: 150–255, V: 50–255 are somewhat "stringent")
    lower_green = np.array([50, 150, 50], dtype=np.uint8)
    upper_green = np.array([70, 255, 255], dtype=np.uint8)
    
    # 3. Create a mask for pixels within this green range
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # 4. Morphological operations to remove small noise
    #    - Erode and then dilate (opening), or vice versa if needed
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    
    # 5. Find connected components (so we can get the largest green blob)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    # stats is an array of shape (num_labels, 5): [label, left, top, width, height, area]
    # The first row (index 0) is the background label.

    if num_labels <= 1:
        # No green component found
        return None
    
    # 6. Identify the largest non-background component by area
    #    stats[1:, cv2.CC_STAT_AREA] -> area of each labeled component (excluding background)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  # +1 offset for background row
    
    # 7. Extract bounding box of that largest component
    x = stats[largest_label, cv2.CC_STAT_LEFT]
    y = stats[largest_label, cv2.CC_STAT_TOP]
    w = stats[largest_label, cv2.CC_STAT_WIDTH]
    h = stats[largest_label, cv2.CC_STAT_HEIGHT]
    
    # Return in (x_min, y_min, x_max, y_max) format
    return (x, y, x + w, y + h)

def imitate_trajectory_with_action_identifier(
    dataset_path="/home/yilong/Documents/policy_data/lift/lift_smaller_2000",
    mesh_dir="/home/yilong/Documents/action_extractor/action_extractor/megapose/panda_hand_mesh",
    output_dir="/home/yilong/Documents/action_extractor/debug/megapose_lift_smaller_2000",
    n=100,
    save_webp=False,
    cameras=["frontview_image", "sideview_image"],
):
    """
    - Only loads the Megapose model once (caching).
    - Suppresses Panda3D logging noise.
    - Uses (1,4,4) SE(3) from pose_estimates.poses, but via the new `estimate_pose`.
    - Also uses a simpler video codec (libx264rgb).
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1) Build object_dataset from mesh_dir using your new function
    object_dataset = make_object_dataset_from_folder(Path(mesh_dir))

    # 2) Load the model ONCE
    model_name = "megapose-1.0-RGB-multi-hypothesis"
    from megapose.utils.load_model import NAMED_MODELS, load_named_model
    model_info = NAMED_MODELS[model_name]
    logger.info(f"Loading model {model_name} once at script start.")
    pose_estimator = load_named_model(model_name, object_dataset).cuda()
    
    left_finger_pose_estimator = load_named_model(model_name, finger_object_dataset).cuda()
    right_finger_pose_estimator = load_named_model(model_name, finger_object_dataset).cuda()

    # 3) Preprocess dataset if needed (HDF5->Zarr)
    sequence_dirs = glob(f"{dataset_path}/**/*.hdf5", recursive=True)
    for seq_dir in sequence_dirs:
        ds_dir = seq_dir.replace('.hdf5', '.zarr')
        zarr_path = seq_dir.replace('.hdf5', '.zarr.zip')
        if not os.path.exists(zarr_path):
            hdf5_to_zarr_parallel_with_progress(seq_dir)
            store = DirectoryStore(ds_dir)
            root = zarr.group(store, overwrite=False)
            store.close()
            directorystore_to_zarr_zip(ds_dir, zarr_path)
            shutil.rmtree(ds_dir)

    # 4) Collect the Zarr files
    zarr_files = glob(f"{dataset_path}/**/*.zarr.zip", recursive=True)
    stores = [ZipStore(zarr_file, mode='r') for zarr_file in zarr_files]
    roots = [zarr.group(store) for store in stores]

    # 5) Initialize environment 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env_meta = get_env_metadata_from_dataset(dataset_path=sequence_dirs[0])
    obs_modality_specs = {
        "obs": {
            "rgb": cameras,
            "depth": [f"{cam.split('_')[0]}_depth" for cam in cameras],
        }
    }
    ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs)

    # Create envs
    env_camera0 = create_env_from_metadata(env_meta=env_meta, render_offscreen=True)
    
    
    example_image = roots[0]["data"]['demo_0']["obs"]["frontview_image"][0]
    camera_height, camera_width = example_image.shape[:2]

    # Now call robosuite functions:
    K = get_camera_intrinsic_matrix(env_camera0.env.sim, camera_name="frontview",
                                    camera_height=camera_height, camera_width=camera_width)
    R = get_camera_extrinsic_matrix(env_camera0.env.sim, camera_name="frontview")
    
    env_camera0 = VideoRecordingWrapper(
        env_camera0,
        video_recoder=VideoRecorder.create_h264(
            fps=20,
            codec='h264',  
            input_pix_fmt='rgb24',
            crf=22,
            thread_type='FRAME',
            thread_count=1,
        ),
        steps_per_render=1, width=camera_width, height=camera_height,
        mode='rgb_array', camera_name=cameras[0].split('_')[0]
    )

    env_camera1 = create_env_from_metadata(env_meta=env_meta, render_offscreen=True)
    env_camera1 = VideoRecordingWrapper(
        env_camera1,
        video_recoder=VideoRecorder.create_h264(
            fps=20,
            codec='h264',
            input_pix_fmt='rgb24',
            crf=22,
            thread_type='FRAME',
            thread_count=1
        ),
        steps_per_render=1, width=camera_width, height=camera_height,
        mode='rgb_array', camera_name=cameras[1].split('_')[0]
    )

    n_success = 0
    total_n = 0
    results = []

    # A function to convert rotation matrix -> quaternion
    def rotation_matrix_to_quaternion(R):
        eps = 1e-7
        trace = R[0,0] + R[1,1] + R[2,2]
        if trace > 0.0:
            S = np.sqrt(trace + 1.0) * 2
            qw = 0.25 * S
            qx = (R[2,1] - R[1,2]) / S
            qy = (R[0,2] - R[2,0]) / S
            qz = (R[1,0] - R[0,1]) / S
        else:
            if (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
                S = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
                qw = (R[2,1] - R[1,2]) / S
                qx = 0.25 * S
                qy = (R[0,1] + R[1,0]) / S
                qz = (R[0,2] + R[2,0]) / S
            elif (R[1,1] > R[2,2]):
                S = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
                qw = (R[0,2] - R[2,0]) / S
                qx = (R[0,1] + R[1,0]) / S
                qy = 0.25 * S
                qz = (R[1,2] + R[2,1]) / S
            else:
                S = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
                qw = (R[1,0] - R[0,1]) / S
                qx = (R[0,2] + R[2,0]) / S
                qy = (R[1,2] + R[2,1]) / S
                qz = 0.25 * S
        return [qw, qx, qy, qz]

    # Loop over demos
    for root in roots:
        demos = list(root["data"].keys())[:n] if n else list(root["data"].keys())
        for demo in tqdm(demos, desc="Processing demos"):
            demo_id = demo.replace("demo_", "")
            upper_left_video_path  = os.path.join(output_dir, f"{demo_id}_upper_left.mp4")
            upper_right_video_path = os.path.join(output_dir, f"{demo_id}_upper_right.mp4")
            lower_left_video_path  = os.path.join(output_dir, f"{demo_id}_lower_left.mp4")
            lower_right_video_path = os.path.join(output_dir, f"{demo_id}_lower_right.mp4")
            combined_video_path    = os.path.join(output_dir, f"{demo_id}_combined.mp4")

            obs_group = root["data"][demo]["obs"]
            num_samples = obs_group["frontview_image"].shape[0]

            # Build the left videos
            upper_left_frames = [obs_group["frontview_image"][i] for i in range(num_samples)]
            lower_left_frames = [obs_group["sideview_image"][i]   for i in range(num_samples)]
            with imageio.get_writer(upper_left_video_path, fps=20) as writer:
                for frame in upper_left_frames:
                    writer.append_data(frame)
            with imageio.get_writer(lower_left_video_path, fps=20) as writer:
                for frame in lower_left_frames:
                    writer.append_data(frame)

            all_poses_world = []
            for i in tqdm(range(num_samples), desc="Inferring pose for each sample"):
                # Build the bounding box. For instance, we do
                # from megapose.datasets.scene_dataset import ObjectData
                from megapose.datasets.scene_dataset import ObjectData
                rgb_image = obs_group["frontview_image"][i]

                # e.g. we do a naive bounding box [0,0,128,128] or find_green_bounding_box
                object_data = [ObjectData(label="panda-hand", bbox_modal=find_green_bounding_box(rgb_image))]
                detections = make_detections_from_object_data(object_data)

                # Use the new estimate_pose with the pre-loaded pose_estimator + model_info
                pose_estimates = estimate_pose(
                    image_rgb=rgb_image,
                    K=K.astype(np.float32),
                    detections=detections,
                    pose_estimator=pose_estimator,
                    model_info=model_info,
                    depth=None,              # or an actual depth if you have it
                    output_dir=None
                )

                # If no detection, shape might be (0,4,4)
                if pose_estimates.poses.shape[0] < 1:
                    all_poses_world.append(None)
                    continue

                T_cam_obj = pose_estimates.poses[0].cpu().numpy()  # shape (4,4)
                
                # We assume frontview_R is camera->world (or world->camera).
                T_camera_world = R
                T_world_obj = T_camera_world @ T_cam_obj
                all_poses_world.append(T_world_obj)

            # Compute delta actions
            actions_for_demo = []
            for i in range(num_samples - 1):
                if (all_poses_world[i] is None) or (all_poses_world[i+1] is None):
                    actions_for_demo.append(np.zeros(7, dtype=np.float32))
                    continue

                pos_i  = all_poses_world[i][:3, 3]   # shape (3,)
                pos_i1 = all_poses_world[i+1][:3, 3]

                # Subtract positions directly in global frame
                p_delta = 80.0 * (pos_i1 - pos_i)  # a 3D difference vector

                action = np.zeros(7, dtype=np.float32)
                action[:3] = p_delta
                # if you want orientation in last 4
                # action[3:] = q_delta
                actions_for_demo.append(action)

            # Roll out environment
            initial_state = root["data"][demo]["states"][0]
            env_camera0.reset()
            env_camera0.reset_to({"states": initial_state})
            env_camera0.file_path = upper_right_video_path
            env_camera0.step_count = 0

            for act in actions_for_demo:
                env_camera0.step(act)

            env_camera0.video_recoder.stop()
            env_camera0.file_path = None

            env_camera1.reset()
            env_camera1.reset_to({"states": initial_state})
            env_camera1.file_path = lower_right_video_path
            env_camera1.step_count = 0

            for act in actions_for_demo:
                env_camera1.step(act)

            env_camera1.video_recoder.stop()
            env_camera1.file_path = None

            # success check
            success = env_camera0.is_success()['task'] and env_camera1.is_success()['task']
            if success:
                n_success += 1
            total_n += 1
            results.append(f"{demo}: {'success' if success else 'failed'}")

            # Combine quadrant videos
            combine_videos_quadrants(
                upper_left_video_path,
                upper_right_video_path,
                lower_left_video_path,
                lower_right_video_path,
                combined_video_path
            )
            os.remove(upper_left_video_path)
            os.remove(upper_right_video_path)
            os.remove(lower_left_video_path)
            os.remove(lower_right_video_path)

    success_rate = (n_success / total_n) * 100 if total_n else 0
    results.append(f"\nFinal Success Rate: {n_success}/{total_n}: {success_rate:.2f}%")
    with open(os.path.join(output_dir, "trajectory_results.txt"), "w") as f:
        f.write("\n".join(results))

    # optionally convert mp4->webp
    if save_webp:
        print("Converting videos to webp format...")
        mp4_files = glob(os.path.join(output_dir, "*.mp4"))
        for mp4_file in tqdm(mp4_files, desc="Converting to webp"):
            webp_file = mp4_file.replace('.mp4', '.webp')
            try:
                convert_mp4_to_webp(mp4_file, webp_file)
                os.remove(mp4_file)
            except Exception as e:
                print(f"Error converting {mp4_file}: {e}")


if __name__ == "__main__":
    # Now you won't see the model reloaded every time, 
    # and Panda3D logs are suppressed to fatal.
    imitate_trajectory_with_action_identifier(
        dataset_path="/home/yilong/Documents/policy_data/lift/raw/1736557347_6730564/test",
        mesh_dir="/home/yilong/Documents/action_extractor/action_extractor/megapose/panda_hand_mesh",
        output_dir="/home/yilong/Documents/action_extractor/debug/megapose_lift_smaller_2000",
        n=100,
        save_webp=False
    )
