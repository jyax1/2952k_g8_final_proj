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

# video (unlabeled) -> use pose estimation to label videos -> video with pseudo labels -> train policy

# This script:
# 1) Takes video from dataset
# 2) Pass video into megapose to get pose estimations
# 3) Use pose estimations to get actions
# 4) Use actions to roll-out in the simulator
# 5) Save video into designated mp4 file
# Function: evaluate quality of pose estimation

#####################################################################
# Normal imports
#####################################################################
import os
import numpy as np
import torch
import math
import imageio
import cv2
import zarr
import shutil
from glob import glob
from tqdm import tqdm
from pathlib import Path
from zarr import DirectoryStore, ZipStore
import itertools

import robomimic.utils.obs_utils as ObsUtils
from robomimic.utils.file_utils import get_env_metadata_from_dataset
from robomimic.utils.env_utils import create_env_from_metadata

from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder

# Megapose
from megapose.utils.load_model import NAMED_MODELS, load_named_model
from megapose.utils.logging import get_logger
logger = get_logger(__name__)

# Our local imports
from action_extractor.utils.dataset_utils import (
    hdf5_to_zarr_parallel_with_progress,
    directorystore_to_zarr_zip,
)

from action_extractor.utils.angles import *

from action_extractor.megapose.action_identifier_megapose import (
    make_object_dataset_from_folder,
    bounding_box_center,   # might be unused
    find_color_bounding_box,
    pixel_to_world,
    poses_to_absolute_actions,
    poses_to_absolute_actions_average,
    poses_to_absolute_actions_from_closest_camera,
    poses_to_absolute_actions_from_multiple_cameras,
    poses_to_absolute_actions_mixed_ori_v1,
    ActionIdentifierMegapose
)
from robosuite.utils.camera_utils import (
    get_camera_extrinsic_matrix,
    get_camera_intrinsic_matrix,
)

from transforms3d.euler import quat2euler, euler2quat

from scipy.spatial.transform import Rotation as R

import open3d as o3d
from sklearn.decomposition import PCA


def combine_videos_quadrants(top_left_video_path, top_right_video_path, 
                             bottom_left_video_path, bottom_right_video_path, 
                             output_path):
    """
    Combines four videos into a single quadrant layout video. 
    Continues until the *longest* video ends.
    If a shorter video ends, we freeze its last frame until all videos are done.
    """
    readers = [
        imageio.get_reader(top_left_video_path),
        imageio.get_reader(top_right_video_path),
        imageio.get_reader(bottom_left_video_path),
        imageio.get_reader(bottom_right_video_path)
    ]

    # We'll assume the FPS of the first video, but you can adapt if they differ.
    fps = readers[0].get_meta_data().get("fps", 20)

    # Keep track of whether each video is done reading
    done = [False, False, False, False]
    # Store the last frame for each quadrant
    last_frames = [None, None, None, None]

    # Initialize each video with its first frame if possible
    for i in range(4):
        try:
            last_frames[i] = readers[i].get_next_data()
        except (StopIteration, IndexError):
            # If no frame, mark done and last_frames[i] = None
            done[i] = True
            last_frames[i] = None

    with imageio.get_writer(output_path, fps=fps) as writer:
        while True:
            # If all are done, stop
            if all(done):
                break

            # Attempt to read the next frame from each video not yet done
            for i in range(4):
                if not done[i]:
                    try:
                        new_frame = readers[i].get_next_data()
                        last_frames[i] = new_frame
                    except (StopIteration, IndexError):
                        # Mark that video as done; keep last frame frozen
                        done[i] = True

            # At this point, we have an updated 'last_frames' for each quadrant
            # Some might be frozen if that video is done

            # If any of the last_frames is None from the start, create a black image 
            # matching the shape of a non-None frame. If *all* are None, we have no data left.
            if all(frame is None for frame in last_frames):
                # Means all videos had 0 frames from the start, or we used them up
                break

            # For a None frame (no data ever), freeze as black image matching shape of any valid frame
            # We'll find the first valid shape
            shape_for_black = None
            for f in last_frames:
                if f is not None:
                    shape_for_black = f.shape
                    break
            if shape_for_black is None:
                # No valid shape at all => end
                break

            # For any quadrant that is None, produce a black image of the same shape
            for i in range(4):
                if last_frames[i] is None:
                    last_frames[i] = np.zeros(shape_for_black, dtype=np.uint8)

            tl_frame = last_frames[0]
            tr_frame = last_frames[1]
            bl_frame = last_frames[2]
            br_frame = last_frames[3]

            # Combine frames in a quadrant layout
            top = np.hstack([tl_frame, tr_frame])
            bottom = np.hstack([bl_frame, br_frame])
            combined = np.vstack([top, bottom])

            writer.append_data(combined)

    # Close all readers
    for r in readers:
        r.close()


def convert_mp4_to_webp(input_path, output_path, quality=80):
    """Converts MP4 video to WebP format using ffmpeg."""
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

def load_ground_truth_poses_as_actions(obs_group, env_camera0):
    """
    Modified version:
      - We compute delta quaternions between consecutive steps, 
      - then 'add' (i.e. multiply) that delta with 'current_orientation'.
      - Then convert that updated orientation to axis-angle for each action.
    """

    # (N, 3)
    pos_array = obs_group["robot0_eef_pos"][:]   # end-effector positions
    # (N, 4) => [qx, qy, qz, qw] in the world frame
    quat_array = obs_group["robot0_eef_quat"][:] 

    num_samples = pos_array.shape[0]

    # Some internal environment variable -- adjust if needed.
    current_orientation = env_camera0.env.env._eef_xquat.astype(np.float32)
    current_orientation = quat_normalize(current_orientation)

    # We will have one fewer action than the number of samples,
    # because we form deltas between consecutive frames.
    num_actions = num_samples - 1

    all_actions = np.zeros((num_actions, 7), dtype=np.float32)

    # Keep track of the previous axis-angle to unify sign across consecutive frames
    prev_rvec = None

    for i in range(num_actions):
        q_i   = quat_array[i]
        q_i1  = quat_array[i + 1]
        q_i   = quat_normalize(q_i)
        q_i1  = quat_normalize(q_i1)

        # 1) compute delta: how do we go from q_i to q_i1?
        #    delta = q_i1 * inv(q_i)
        q_delta = quat_multiply(quat_inv(q_i), q_i1)
        q_delta = quat_normalize(q_delta)

        # 2) "add" this delta to current_orientation (i.e. multiply in quaternion space)
        current_orientation = quat_multiply(current_orientation, q_delta)
        current_orientation = quat_normalize(current_orientation)

        # 3) Convert to axis-angle
        rvec = quat2axisangle(current_orientation)

        # 4) optional sign unify (avoid ± flips in axis representation)
        if prev_rvec is not None and np.dot(rvec, prev_rvec) < 0:
            rvec = -rvec
        prev_rvec = rvec

        # For position, you can choose either pos_array[i] or pos_array[i+1]. 
        # Typically we take the next position: 
        px, py, pz = pos_array[i+1]

        all_actions[i, 0:3] = [px, py, pz]
        all_actions[i, 3:6] = rvec
        all_actions[i, 6]   = 1.0  # e.g., keep gripper = open = 1

    return all_actions

def get_poses_from_pointclouds(point_clouds_points, point_clouds_colors):
    """
    Estimate accurate poses from green points in point clouds using a predefined rectangular box.
    
    Args:
        point_clouds_points (list): List of (N,3) numpy arrays of 3D points (meters)
        point_clouds_colors (list): List of (N,3) numpy arrays of RGB colors (0-255)
    
    Returns:
        list: SE(3) pose matrices (4x4) for each point cloud
    """
    # Predefined box dimensions (converted from mm to meters)
    box_dims = np.array([0.063045, 0.204516, 0.091946])  # [thickness, width, length]

    poses = []

    for points, colors in zip(point_clouds_points, point_clouds_colors):
        # 1. Filter green points using a strict threshold
        green_mask = (colors[:, 1] > colors[:, 0]) & \
                     (colors[:, 1] > colors[:, 2]) & \
                     (colors[:, 1] > 200) & \
                     (colors[:, 0] < 50) & \
                     (colors[:, 2] < 50)
        
        green_points = points[green_mask]
        
        if len(green_points) < 10:  # Skip if insufficient points
            poses.append(np.eye(4))
            continue

        # 2. Create Open3D point cloud and remove outliers
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(green_points)

        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

        # 3. Compute Oriented Bounding Box (OBB)
        obb = pcd.get_oriented_bounding_box()
        
        # 4. Align OBB to the known box dimensions
        R_obb = obb.R  # Rotation matrix (3x3)
        center = obb.center  # OBB center

        # Reorder the OBB axes to match the known box dimensions
        obb_dims = obb.extent  # Estimated dimensions

        # Find the best permutation of obb_dims that matches box_dims
        permuted_dims = sorted(obb_dims)
        sorted_box_dims = sorted(box_dims)

        if not np.allclose(permuted_dims, sorted_box_dims, atol=0.02):  # Allow small tolerance
            print("Warning: OBB dimensions don't match the expected box size!")

        # Align OBB axes with known dimensions
        axis_permutation = np.argsort(obb_dims)  # Reorder axes to match sorted dimensions
        aligned_R = R_obb[:, axis_permutation]  # Apply permutation to rotation matrix

        # 5. Compute the bottom center (assuming thickness is along Z)
        bottom_center_local = np.array([0, 0, -box_dims[2] / 2])  # Local frame
        bottom_center_world = center + aligned_R @ bottom_center_local

        # 6. Construct SE(3) pose matrix
        pose = np.eye(4)
        pose[:3, :3] = aligned_R
        pose[:3, 3] = bottom_center_world
        
        poses.append(pose)

    return poses

def debug_pointcloud_poses(point_clouds_points, point_clouds_colors, output_dir="debug/pointclouds"):
    """
    Save point clouds with colors and detected bounding boxes for visualization.

    Args:
        point_clouds_points (list): List of (N,3) numpy arrays
        point_clouds_colors (list): List of (N,3) numpy arrays (0-255)
        output_dir (str): Output directory for PLY files
    """
    os.makedirs(output_dir, exist_ok=True)

    for idx, (points, colors) in enumerate(zip(point_clouds_points, point_clouds_colors)):
        # Create main colored point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)

        # Save the original point cloud
        o3d.io.write_point_cloud(os.path.join(output_dir, f"pc_{idx:03d}.ply"), pcd)

        # Filter green points
        green_mask = (colors[:, 1] > colors[:, 0]) & \
                     (colors[:, 1] > colors[:, 2]) & \
                     (colors[:, 1] > 100) & \
                     (colors[:, 0] < 100) & \
                     (colors[:, 2] < 100)
        green_points = points[green_mask]

        if len(green_points) > 10:
            # Create green points point cloud
            green_pcd = o3d.geometry.PointCloud()
            green_pcd.points = o3d.utility.Vector3dVector(green_points)
            green_pcd.paint_uniform_color([0, 1, 0])  # Green color
            
            # Save the green points separately
            o3d.io.write_point_cloud(os.path.join(output_dir, f"green_pc_{idx:03d}.ply"), green_pcd)

            # Compute Oriented Bounding Box (OBB)
            obb = green_pcd.get_oriented_bounding_box()
            obb.color = [1, 0, 0]  # Red color for bbox

            # Convert OBB to a LineSet for visualization
            obb_lineset = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb)

            # Save bounding box separately
            o3d.io.write_line_set(os.path.join(output_dir, f"obb_{idx:03d}.ply"), obb_lineset)

            # Create a coordinate frame at the bottom center of the box
            extent = obb.extent
            R = obb.R
            bottom_center = obb.center + R @ np.array([0, 0, -extent[2] / 2])
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            coord_frame.translate(bottom_center)
            coord_frame.rotate(R)

            # Save coordinate frame separately
            o3d.io.write_triangle_mesh(os.path.join(output_dir, f"coord_frame_{idx:03d}.ply"), coord_frame)

            # Visualize all elements
            o3d.visualization.draw_geometries([pcd, green_pcd, obb_lineset, coord_frame])

def imitate_trajectory_with_action_identifier(
    dataset_path="/home/yilong/Documents/policy_data/lift/lift_smaller_2000",
    hand_mesh_dir="/home/yilong/Documents/action_extractor/action_extractor/megapose/panda_hand_mesh",
    output_dir="/home/yilong/Documents/action_extractor/debug/megapose_lift_smaller_2000",
    num_demos=100,
    save_webp=False,
    cameras: list[str] = ["squared0view_image", "sidetableview_image", "squared0view2_image"],
    batch_size=40,
):
    """
    General version where 'cameras' is a list of camera observation strings,
    e.g. ["frontview_image", "sideview_image", "birdview_image", ...].

    This code now:
      - Computes intrinsic and extrinsic parameters for every camera in the list,
        storing them in dictionaries (camera_Ks and camera_Rs).
      - Initializes the two rendering environments (env_camera0 and env_camera1) using cameras[0] and cameras[1].
      - When calling the pose estimator, it now passes dictionaries of frames and depth lists for all cameras.
      - (Later you can update the pose-to-action conversion function to combine an arbitrary number of cameras.)
    """
    # 0) Create output directory.
    os.makedirs(output_dir, exist_ok=True)

    # 1) Build object dataset.
    hand_object_dataset = make_object_dataset_from_folder(Path(hand_mesh_dir))

    # 2) Load the pose estimation model once.
    model_name = "megapose-1.0-RGB-multi-hypothesis-icp"
    model_info = NAMED_MODELS[model_name]
    logger.info(f"Loading model {model_name} once at script start.")
    hand_pose_estimator = load_named_model(model_name, hand_object_dataset).cuda()

    # 3) Preprocess dataset => convert HDF5 to Zarr.
    sequence_dirs = glob(f"{dataset_path}/**/*.hdf5", recursive=True)
    for seq_dir in sequence_dirs:
        ds_dir = seq_dir.replace(".hdf5", ".zarr")
        zarr_path = seq_dir.replace(".hdf5", ".zarr.zip")
        if not os.path.exists(zarr_path):
            hdf5_to_zarr_parallel_with_progress(seq_dir, max_workers=16)
            store = DirectoryStore(ds_dir)
            root_z = zarr.group(store, overwrite=False)
            store.close()
            directorystore_to_zarr_zip(ds_dir, zarr_path)
            shutil.rmtree(ds_dir)

    # 4) Collect Zarr files.
    zarr_files = glob(f"{dataset_path}/**/*.zarr.zip", recursive=True)
    stores = [ZipStore(zarr_file, mode="r") for zarr_file in zarr_files]
    roots = [zarr.group(store) for store in stores]

    # 5) Create environment metadata.
    env_meta = get_env_metadata_from_dataset(dataset_path=sequence_dirs[0])
    env_meta['env_kwargs']['controller_configs']['control_delta'] = False
    env_meta['env_kwargs']['controller_configs']['type'] = 'OSC_POSE'

    # Extract base camera names from the observation strings.
    # For example, "frontview_image" becomes "frontview".
    camera_names = [cam.split("_")[0] for cam in cameras]

    # Setup observation modality specs.
    obs_modality_specs = {
        "obs": {
            "rgb": cameras,
            "depth": [f"{cam.split('_')[0]}_depth" for cam in cameras],
        }
    }
    ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs)

    # Create a rendering environment (we'll use it to obtain image dimensions and camera parameters).
    env_camera0 = create_env_from_metadata(env_meta=env_meta, render_offscreen=True)
    example_image = roots[0]["data"]["demo_0"]["obs"][cameras[0]][0]
    camera_height, camera_width = example_image.shape[:2]

    # 6) Compute intrinsics and extrinsics for every camera in the list.
    camera_Ks = {}
    camera_Rs = {}
    for cam in camera_names:
        camera_Ks[cam] = get_camera_intrinsic_matrix(
            env_camera0.env.sim,
            camera_name=cam,
            camera_height=camera_height,
            camera_width=camera_width,
        )
        camera_Rs[cam] = get_camera_extrinsic_matrix(
            env_camera0.env.sim,
            camera_name=cam,
        )

    # 7) Initialize rendering environments for at least two cameras.
    # Use cameras[0] and cameras[1] for video recording.
    env_camera0 = VideoRecordingWrapper(
        env_camera0,
        video_recoder=VideoRecorder.create_h264(fps=20, codec="h264", input_pix_fmt="rgb24", crf=22),
        steps_per_render=1,
        width=camera_width,
        height=camera_height,
        mode="rgb_array",
        camera_name=camera_names[0],
    )
    env_camera1 = create_env_from_metadata(env_meta=env_meta, render_offscreen=True)
    env_camera1 = VideoRecordingWrapper(
        env_camera1,
        video_recoder=VideoRecorder.create_h264(fps=20, codec="h264", input_pix_fmt="rgb24", crf=22),
        steps_per_render=1,
        width=camera_width,
        height=camera_height,
        mode="rgb_array",
        camera_name=camera_names[1],
    )

    # 8) Build the ActionIdentifierMegapose using all camera info.
    # Here we update the signature to accept dictionaries instead of individual cameras.
    action_identifier = ActionIdentifierMegapose(
        pose_estimator=hand_pose_estimator,
        camera_Rs=camera_Rs,  # dictionary mapping camera name -> extrinsic matrix
        camera_Ks=camera_Ks,  # dictionary mapping camera name -> intrinsic matrix
        model_info=model_info,
        batch_size=batch_size,
        scale_translation=80.0,
    )

    n_success = 0
    total_n = 0
    results = []

    # 9) Loop over demos.
    for root_z in roots:
        demos = list(root_z["data"].keys())[:num_demos] if num_demos else list(root_z["data"].keys())
        for demo in tqdm(demos, desc="Processing demos"):
            demo_id = demo.replace("demo_", "")
            upper_left_video_path  = os.path.join(output_dir, f"{demo_id}_upper_left.mp4")
            upper_right_video_path = os.path.join(output_dir, f"{demo_id}_upper_right.mp4")
            lower_left_video_path  = os.path.join(output_dir, f"{demo_id}_lower_left.mp4")
            lower_right_video_path = os.path.join(output_dir, f"{demo_id}_lower_right.mp4")
            combined_video_path    = os.path.join(output_dir, f"{demo_id}_combined.mp4")

            obs_group = root_z["data"][demo]["obs"]
            num_samples = obs_group[cameras[0]].shape[0]

            # 10) For each camera, extract frames and (if available) depth.
            cameras_frames = {}
            cameras_depth = {}
            for cam_obs in cameras:
                base = cam_obs.split("_")[0]
                cameras_frames[base] = [obs_group[cam_obs][i] for i in range(num_samples)]
                depth_key = f"{base}_depth"
                if depth_key in obs_group:
                    cameras_depth[base] = [obs_group[depth_key][i] for i in range(num_samples)]
                else:
                    cameras_depth[base] = None
                    
            with imageio.get_writer(upper_left_video_path, fps=20) as writer:
                for frame in cameras_frames[camera_names[0]]:
                    writer.append_data(frame)
            with imageio.get_writer(lower_left_video_path, fps=20) as writer:
                for frame in cameras_frames[camera_names[1]]:
                    writer.append_data(frame)
                    
            point_clouds_points = [points for points in obs_group[f"pointcloud_points"]]
            point_clouds_colors = [colors for colors in obs_group[f"pointcloud_colors"]]
                    
            all_hand_poses = get_poses_from_pointclouds(point_clouds_points, point_clouds_colors)
            
            # debug_pointcloud_poses(point_clouds_points[:10], point_clouds_colors[:10], output_dir=os.path.join(output_dir, "pointcloud_debug"))

            # 12) Build absolute actions.
            # (Assume you have updated a function to combine poses from an arbitrary number of cameras.)
            actions_for_demo = poses_to_absolute_actions(
                poses=all_hand_poses,
                gripper_actions=[root_z["data"][demo]['actions'][i][-1] for i in range(num_samples)],
                env=env_camera0  # using camera0 environment for execution
            )

            initial_state = root_z["data"][demo]["states"][0]

            # 13) Execute actions and record videos.
            # For simplicity, we use env_camera0 and env_camera1 for two views;
            # you can later extend this to record from all cameras.
            # Top-right video from camera0 environment:
            env_camera0.reset()
            env_camera0.reset_to({"states": initial_state})
            env_camera0.file_path = upper_right_video_path
            env_camera0.step_count = 0
            for action in actions_for_demo:
                env_camera0.step(action)
            env_camera0.video_recoder.stop()
            env_camera0.file_path = None

            # Bottom-right video from camera1 environment:
            env_camera1.reset()
            env_camera1.reset_to({"states": initial_state})
            env_camera1.file_path = lower_right_video_path
            env_camera1.step_count = 0
            for action in actions_for_demo:
                env_camera1.step(action)
            env_camera1.video_recoder.stop()
            env_camera1.file_path = None

            # Success check
            success = env_camera0.is_success()["task"]
            if success:
                n_success += 1
            total_n += 1
            results.append(f"{demo}: {'success' if success else 'failed'}")

            # Combine videos from all cameras (if desired).
            # Here, we assume a function that can combine multiple videos.
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
    results.append(f"\nFinal Success Rate: {n_success}/{total_n} => {success_rate:.2f}%")
    with open(os.path.join(output_dir, "trajectory_results.txt"), "w") as f:
        f.write("\n".join(results))

    if save_webp:
        print("Converting to webp...")
        mp4_files = glob(os.path.join(output_dir, "*.mp4"))
        for mp4_file in tqdm(mp4_files):
            webp_file = mp4_file.replace(".mp4", ".webp")
            try:
                convert_mp4_to_webp(mp4_file, webp_file)
                os.remove(mp4_file)
            except Exception as e:
                print(f"Error converting {mp4_file}: {e}")

    print(f"Wrote results to {os.path.join(output_dir, 'trajectory_results.txt')}")


if __name__ == "__main__":
    imitate_trajectory_with_action_identifier(
        dataset_path="/home/yilong/Documents/policy_data/square_d0/raw/test/test_pointcloud",
        hand_mesh_dir="/home/yilong/Documents/action_extractor/action_extractor/megapose/panda_hand_mesh",
        output_dir="/home/yilong/Documents/action_extractor/debug/megapose_weighted_average_squared0view12",
        num_demos=3,
        save_webp=False,
        batch_size=40
    )