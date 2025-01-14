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

#####################################################################
# Instead of from action_extractor.megapose.action_identifier_megapose 
# we STILL import the same run_inference_on_data, but we will pass 
# our pre-loaded model to it so it doesn't re-load. We'll define 
# a small wrapper or local approach for that.
#####################################################################
from action_extractor.megapose.action_identifier_megapose import (
    run_inference_on_data,
    make_detections_from_object_data,
    make_object_dataset_from_folder,
)

def combine_videos_quadrants(top_left_video_path, top_right_video_path, bottom_left_video_path, bottom_right_video_path, output_path):
    top_left_reader = imageio.get_reader(top_left_video_path)
    top_right_reader = imageio.get_reader(top_right_video_path)
    bottom_left_reader = imageio.get_reader(bottom_left_video_path)
    bottom_right_reader = imageio.get_reader(bottom_right_video_path)
    fps = top_left_reader.get_meta_data()["fps"]

    top_left_frames = list(top_left_reader)
    top_right_frames = list(top_right_reader)
    bottom_left_frames = list(bottom_left_reader)
    bottom_right_frames = list(bottom_right_reader)

    min_length = min(
        len(top_left_frames), len(top_right_frames),
        len(bottom_left_frames), len(bottom_right_frames)
    )
    top_left_frames    = top_left_frames[:min_length]
    top_right_frames   = top_right_frames[:min_length]
    bottom_left_frames = bottom_left_frames[:min_length]
    bottom_right_frames= bottom_right_frames[:min_length]

    combined_frames = [
        np.vstack([
            np.hstack([tl, tr]),
            np.hstack([bl, br])
        ])
        for tl, tr, bl, br in zip(
            top_left_frames, top_right_frames,
            bottom_left_frames, bottom_right_frames
        )
    ]

    with imageio.get_writer(output_path, fps=fps) as writer:
        for frame in combined_frames:
            writer.append_data(frame)


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


#####################################################################
# We'll define a "run_inference_with_cached_model" that reuses model
#####################################################################
def run_inference_with_cached_model(
    pose_estimator,
    image_rgb: np.ndarray,
    K: np.ndarray,
    detections,
    requires_depth: bool = False,
    depth: np.ndarray = None
):
    """
    A local function that calls the same logic as run_inference_on_data,
    but doesn't re-load the model. We only pass in a preloaded 'pose_estimator'.
    """
    from megapose.inference.types import ObservationTensor
    from megapose.utils.logging import get_logger
    logger = get_logger(__name__)  # if you want to log

    # Build ObservationTensor
    observation = ObservationTensor.from_numpy(
        image_rgb,
        depth,
        K
    ).cuda()

    # Just call the pipeline with the existing pose_estimator
    logger.debug("Running inference pipeline with cached model...")

    output, _ = pose_estimator.run_inference_pipeline(
        observation,
        detections=detections.cuda(),
        # no extra named arguments needed if your model_info has been 
        # integrated into the pose_estimator
    )
    return output


def imitate_trajectory_with_action_identifier(
    dataset_path="/home/yilong/Documents/policy_data/lift/lift_smaller_2000",
    mesh_dir="/home/yilong/Documents/action_extractor/action_extractor/megapose/panda_hand_mesh",
    output_dir="/home/yilong/Documents/action_extractor/debug/hyperspherical_lift_1000",
    n=100,
    save_webp=False,
    cameras=["frontview_image", "sideview_image"],
    right_video_mode='megapose_inference'
):
    """
    - Only loads the Megapose model once (caching).
    - Suppresses Panda3D logging noise.
    - Uses (1,4,4) SE(3) from pose_estimates.poses.
    - Also uses a simpler video codec (libx264rgb).
    """
    os.makedirs(output_dir, exist_ok=True)

    from pathlib import Path
    object_dataset = make_object_dataset_from_folder(Path(mesh_dir))

    # ---- LOAD THE MODEL ONLY ONCE ----
    from megapose.utils.load_model import NAMED_MODELS, load_named_model
    from megapose.utils.logging import get_logger
    logger = get_logger(__name__)

    model_info = NAMED_MODELS["megapose-1.0-RGB-multi-hypothesis"]
    logger.info("Loading model 'megapose-1.0-RGB-multi-hypothesis'.")
    pose_estimator = load_named_model("megapose-1.0-RGB-multi-hypothesis", object_dataset).cuda()

    # 2) Convert HDF5->Zarr if needed
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

    zarr_files = glob(f"{dataset_path}/**/*.zarr.zip", recursive=True)
    stores = [ZipStore(zarr_file, mode='r') for zarr_file in zarr_files]
    roots = [zarr.group(store) for store in stores]

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
    env_camera0 = VideoRecordingWrapper(
        env_camera0,
        video_recoder=VideoRecorder.create_h264(
            fps=20,
            codec='libx264rgb',  
            input_pix_fmt='rgb24',
            crf=22,
            thread_type='FRAME',
            thread_count=1
        ),
        steps_per_render=1, width=128, height=128,
        mode='rgb_array', camera_name=cameras[0].split('_')[0]
    )

    env_camera1 = create_env_from_metadata(env_meta=env_meta, render_offscreen=True)
    env_camera1 = VideoRecordingWrapper(
        env_camera1,
        video_recoder=VideoRecorder.create_h264(
            fps=20,
            codec='libx264rgb',
            input_pix_fmt='rgb24',
            crf=22,
            thread_type='FRAME',
            thread_count=1
        ),
        steps_per_render=1, width=128, height=128,
        mode='rgb_array', camera_name=cameras[1].split('_')[0]
    )

    n_success = 0
    total_n = 0
    results = []

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

            # Build left videos
            upper_left_frames = [obs_group["frontview_image"][i] for i in range(num_samples)]
            lower_left_frames = [obs_group["sideview_image"][i]   for i in range(num_samples)]
            with imageio.get_writer(upper_left_video_path, fps=20) as writer:
                for frame in upper_left_frames:
                    writer.append_data(frame)
            with imageio.get_writer(lower_left_video_path, fps=20) as writer:
                for frame in lower_left_frames:
                    writer.append_data(frame)

            all_poses_world = []
            for i in range(num_samples):
                rgb_image = obs_group["frontview_image"][i]

                # Make bounding box as before
                from megapose.datasets.scene_dataset import ObjectData
                object_data = [ObjectData(label="panda-hand", bbox_modal=[0,0,128,128])]
                detections = make_detections_from_object_data(object_data)

                # Instead of run_inference_on_data which reloads, we call our local function
                # but reusing the same 'pose_estimator' we loaded above.
                pose_estimates = run_inference_with_cached_model(
                    pose_estimator=pose_estimator,
                    image_rgb=rgb_image,
                    K=frontview_K.astype(np.float32),
                    detections=detections
                )

                if pose_estimates.poses.shape[0] < 1:
                    all_poses_world.append(None)
                    continue

                # shape (4,4)
                T_cam_obj = pose_estimates.poses[0].cpu().numpy()
                T_camera_world = frontview_R
                T_world_obj = T_camera_world @ T_cam_obj
                all_poses_world.append(T_world_obj)

            # compute delta actions
            actions_for_demo = []
            for i in range(num_samples - 1):
                if (all_poses_world[i] is None) or (all_poses_world[i+1] is None):
                    actions_for_demo.append(np.zeros(7, dtype=np.float32))
                    continue

                T_i  = all_poses_world[i]
                T_i1 = all_poses_world[i+1]

                T_i_inv = np.linalg.inv(T_i)
                T_delta = T_i_inv @ T_i1

                t_delta = T_delta[:3, 3]
                R_delta = T_delta[:3,:3]
                q_delta = rotation_matrix_to_quaternion(R_delta)

                action = np.zeros(7, dtype=np.float32)
                action[:3] = t_delta
                # action[3:] = q_delta
                action[-1] = 1.0  # dummy gripper value
                actions_for_demo.append(action)
                print(action)

            # roll out environment
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

    success_rate = (n_success / total_n)*100 if total_n else 0
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
        dataset_path="/home/yilong/Documents/policy_data/lift/lift_smaller_2000",
        mesh_dir="/home/yilong/Documents/action_extractor/action_extractor/megapose/panda_hand_mesh",
        output_dir="/home/yilong/Documents/action_extractor/debug/hyperspherical_lift_1000",
        n=100,
        save_webp=False
    )
