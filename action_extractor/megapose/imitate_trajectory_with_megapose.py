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

# Import our refactored code
from action_extractor.megapose.action_identifier_megapose import (
    run_inference_on_data,
    make_detections_from_object_data,
    make_object_dataset_from_folder,  # If needed for partial usage
)

def combine_videos_quadrants(top_left_video_path, top_right_video_path, bottom_left_video_path, bottom_right_video_path, output_path):
    # Read videos
    top_left_reader = imageio.get_reader(top_left_video_path)
    top_right_reader = imageio.get_reader(top_right_video_path)
    bottom_left_reader = imageio.get_reader(bottom_left_video_path)
    bottom_right_reader = imageio.get_reader(bottom_right_video_path)
    fps = top_left_reader.get_meta_data()["fps"]

    top_left_frames = [frame for frame in top_left_reader]
    top_right_frames = [frame for frame in top_right_reader]
    bottom_left_frames = [frame for frame in bottom_left_reader]
    bottom_right_frames = [frame for frame in bottom_right_reader]

    # Ensure same number of frames
    min_length = min(len(top_left_frames), len(top_right_frames), len(bottom_left_frames), len(bottom_right_frames))
    top_left_frames = top_left_frames[:min_length]
    top_right_frames = top_right_frames[:min_length]
    bottom_left_frames = bottom_left_frames[:min_length]
    bottom_right_frames = bottom_right_frames[:min_length]

    # Combine frames into quadrants
    combined_frames = [
        np.vstack([
            np.hstack([top_left, top_right]),
            np.hstack([bottom_left, bottom_right])
        ])
        for top_left, top_right, bottom_left, bottom_right in zip(top_left_frames, top_right_frames, bottom_left_frames, bottom_right_frames)
    ]

    # Save combined video
    with imageio.get_writer(output_path, fps=fps) as writer:
        for frame in combined_frames:
            writer.append_data(frame)
            
# Utility to convert quaternion-based pose vector to 4x4 matrix
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
    """Convert mp4 video to webp format"""
    import subprocess
    import shutil
    
    # Find ffmpeg executable path
    ffmpeg_path = shutil.which('ffmpeg')
    if ffmpeg_path is None:
        raise RuntimeError("ffmpeg not found. Please install it with: sudo apt-get install ffmpeg")
        
    cmd = [
        ffmpeg_path,  # Use full path
        '-i', input_path,
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

def compute_direction_difference(actions_dataset, inferred_actions):
    """
    Compute the difference in direction for each axis between corresponding vectors
    in actions_dataset and inferred_actions.

    Parameters:
    actions_dataset (np.ndarray): Array of shape (N, 7) containing the actions from the dataset.
    inferred_actions (np.ndarray): Array of shape (N, 4) containing the inferred actions.

    Returns:
    np.ndarray: Array of shape (N, 3) containing the difference in direction for each axis.
    """
    # Extract the first three dimensions
    actions_dataset_vectors = actions_dataset[:, :3]
    inferred_actions_vectors = inferred_actions[:, :3]

    # Compute norms
    actions_dataset_norms = np.linalg.norm(actions_dataset_vectors, axis=1, keepdims=True)
    inferred_actions_norms = np.linalg.norm(inferred_actions_vectors, axis=1, keepdims=True)

    # Avoid division by zero by setting norms to 1 where they are zero
    actions_dataset_norms = np.where(actions_dataset_norms == 0, 1, actions_dataset_norms)
    inferred_actions_norms = np.where(inferred_actions_norms == 0, 1, inferred_actions_norms)

    # Normalize the vectors to get their direction
    actions_dataset_directions = actions_dataset_vectors / actions_dataset_norms
    inferred_actions_directions = inferred_actions_vectors / inferred_actions_norms

    # Set directions to zero where norms were zero
    actions_dataset_directions = np.where(actions_dataset_norms == 1, 0, actions_dataset_directions)
    inferred_actions_directions = np.where(inferred_actions_norms == 1, 0, inferred_actions_directions)
    
    inferred_actions_directions[:, 2] += 1.0

    # Compute the difference in direction for each axis
    direction_difference = actions_dataset_directions - inferred_actions_directions

    return direction_difference

def find_green_bounding_box(image_rgb):
    """
    Returns (x_min, y_min, x_max, y_max) for the largest green blob, or None if not found.
    image_rgb shape is (H, W, 3) in uint8 format.
    """

    # Convert to HSV
    image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

    # Define green range
    lower_green = np.array([45, 100, 50], dtype=np.uint8)
    upper_green = np.array([75, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(image_hsv, lower_green, upper_green)

    # Morphological operations to remove noise and fill gaps
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours (largest green blob)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None  # No green found

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    largest_contour = contours[0]

    # Bounding box
    x, y, w, h = cv2.boundingRect(largest_contour)
    return (x, y, x + w, y + h)

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
    Refactored so that we now take a `mesh_dir` argument. We will build a valid
    RigidObjectDataset from `mesh_dir` and pass it to `run_inference_on_data`.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Convert the mesh_dir into a RigidObjectDataset
    from pathlib import Path
    object_dataset = make_object_dataset_from_folder(Path(mesh_dir))

    # 2. Preprocess dataset (HDF5->Zarr) if needed
    sequence_dirs = glob(f"{dataset_path}/**/*.hdf5", recursive=True)
    for seq_dir in sequence_dirs:
        ds_dir = seq_dir.replace('.hdf5', '.zarr')
        zarr_path = seq_dir.replace('.hdf5', '.zarr.zip')
        if not os.path.exists(zarr_path):
            hdf5_to_zarr_parallel_with_progress(seq_dir)
            store = DirectoryStore(ds_dir)
            root = zarr.group(store, overwrite=False)
            # do preprocess_data_parallel if needed
            store.close()
            directorystore_to_zarr_zip(ds_dir, zarr_path)
            shutil.rmtree(ds_dir)

    zarr_files = glob(f"{dataset_path}/**/*.zarr.zip", recursive=True)
    stores = [ZipStore(zarr_file, mode='r') for zarr_file in zarr_files]
    roots = [zarr.group(store) for store in stores]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize obs utils
    env_meta = get_env_metadata_from_dataset(dataset_path=sequence_dirs[0])
    obs_modality_specs = {
        "obs": {
            "rgb": cameras,
            "depth": [f"{cam.split('_')[0]}_depth" for cam in cameras],
        }
    }
    ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs)

    # Create the environment for top-right (env_camera0)
    env_camera0 = create_env_from_metadata(env_meta=env_meta, render_offscreen=True)
    env_camera0 = VideoRecordingWrapper(
        env_camera0,
        video_recoder=VideoRecorder.create_h264(
            fps=20,
            codec='h264',
            input_pix_fmt='rgb24',
            crf=22,
            thread_type='FRAME',
            thread_count=1
        ),
        steps_per_render=1,
        width=128, height=128,
        mode='rgb_array',
        camera_name=cameras[0].split('_')[0]
    )

    # Create the environment for bottom-right (env_camera1)
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
        steps_per_render=1,
        width=128, height=128,
        mode='rgb_array',
        camera_name=cameras[1].split('_')[0]
    )

    n_success = 0
    total_n = 0
    results = []

    # For rotation matrix -> quaternion with fallback
    def rotation_matrix_to_quaternion(R):
        eps = 1e-7
        trace = R[0, 0] + R[1, 1] + R[2, 2]
        if trace > 0.0:
            S = np.sqrt(trace + 1.0) * 2.0
            qw = 0.25 * S
            qx = (R[2, 1] - R[1, 2]) / S
            qy = (R[0, 2] - R[2, 0]) / S
            qz = (R[1, 0] - R[0, 1]) / S
        else:
            if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
                S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
                qw = (R[2, 1] - R[1, 2]) / S
                qx = 0.25 * S
                qy = (R[0, 1] + R[1, 0]) / S
                qz = (R[0, 2] + R[2, 0]) / S
            elif R[1, 1] > R[2, 2]:
                S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
                qw = (R[0, 2] - R[2, 0]) / S
                qx = (R[0, 1] + R[1, 0]) / S
                qy = 0.25 * S
                qz = (R[1, 2] + R[2, 1]) / S
            else:
                S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
                qw = (R[1, 0] - R[0, 1]) / S
                qx = (R[0, 2] + R[2, 0]) / S
                qy = (R[1, 2] + R[2, 1]) / S
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

            # Write left videos
            upper_left_frames = [obs_group["frontview_image"][i] for i in range(num_samples)]
            lower_left_frames = [obs_group["sideview_image"][i]   for i in range(num_samples)]
            with imageio.get_writer(upper_left_video_path, fps=20) as writer:
                for frame in upper_left_frames:
                    writer.append_data(frame)
            with imageio.get_writer(lower_left_video_path, fps=20) as writer:
                for frame in lower_left_frames:
                    writer.append_data(frame)

            # For "megapose_inference", gather absolute poses, then compute deltas
            all_poses_world = []
            for i in range(num_samples):
                rgb_image = obs_group["frontview_image"][i]  # shape (128,128,3), np.uint8
                # Optionally find bounding box via find_green_bounding_box(rgb_image)
                # or just assume full image is bounding box
                from megapose.datasets.scene_dataset import ObjectData
                object_data = [ObjectData(label="panda-hand", bbox_modal=[0,0,128,128])]
                detections = make_detections_from_object_data(object_data)

                # Now we run inference with the object_dataset we built from mesh_dir
                pose_estimates = run_inference_on_data(
                    image_rgb=rgb_image,
                    K=frontview_K.astype(np.float32),
                    detections=detections,
                    model_name="megapose-1.0-RGB-multi-hypothesis",
                    object_dataset=object_dataset,  # <--- Using the actual dataset
                    requires_depth=False,
                    depth=None,
                    output_dir=None
                )

                poses_np = pose_estimates.poses.cpu().numpy()  # shape (1,7) if single detection
                if poses_np.shape[0] < 1:
                    # If no detection, store None
                    all_poses_world.append(None)
                    continue

                tx, ty, tz, qw, qx, qy, qz = poses_np[0]

                # frontview_R might be camera->world or world->camera
                # If it's camera->world, we do:
                T_camera_world = frontview_R
                T_cam_obj = pose_vector_to_matrix([tx, ty, tz], [qw, qx, qy, qz])
                T_world_obj = T_camera_world @ T_cam_obj
                all_poses_world.append(T_world_obj)

            # Compute delta actions
            actions_for_demo = []
            for i in range(num_samples - 1):
                if (all_poses_world[i] is None) or (all_poses_world[i+1] is None):
                    actions_for_demo.append(np.zeros(7, dtype=np.float32))
                    continue
                
                T_i   = all_poses_world[i]
                T_i1  = all_poses_world[i+1]

                # delta = T_i^-1 * T_i+1
                T_i_inv = np.linalg.inv(T_i)
                T_delta = T_i_inv @ T_i1

                # scale translation
                t_delta = T_delta[:3, 3] * 80.0

                R_delta = T_delta[:3, :3]
                q_delta = rotation_matrix_to_quaternion(R_delta)

                action = np.zeros(7, dtype=np.float32)
                action[:3] = t_delta
                action[3:] = q_delta
                actions_for_demo.append(action)

            # Roll out the actions
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

    # Optionally convert mp4->webp
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
    # Example usage, now specifying `mesh_dir` for the object dataset
    imitate_trajectory_with_action_identifier(
        dataset_path="/home/yilong/Documents/policy_data/lift/lift_smaller_2000",
        mesh_dir="/home/yilong/Documents/action_extractor/action_extractor/megapose/panda_hand_mesh",
        output_dir="/home/yilong/Documents/action_extractor/debug/hyperspherical_lift_1000",
        n=100,
        save_webp=False
    )