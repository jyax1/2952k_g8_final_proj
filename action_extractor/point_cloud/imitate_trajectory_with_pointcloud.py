import os
import numpy as np
import imageio
import zarr
import shutil
from glob import glob
from tqdm import tqdm
from pathlib import Path
from zarr import DirectoryStore, ZipStore

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

from action_extractor.poses_to_actions import (
    poses_to_absolute_actions,
    poses_to_delta_actions,
)

from action_extractor.point_cloud.action_identifier_point_cloud import (
    get_poses_from_pointclouds,
)

from action_extractor.poses_to_actions import *

from robosuite.utils.camera_utils import ( # type: ignore
    get_camera_extrinsic_matrix,
    get_camera_intrinsic_matrix,
)

from transforms3d.euler import quat2euler, euler2quat

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

def save_pointclouds_with_bbox_as_ply(point_clouds_points,
                                      point_clouds_colors,
                                      poses,
                                      box_dims = np.array([0.063045, 0.204516, 0.091946]),
                                      output_dir="debug/pointcloud_traj"):
    """
    Saves each original point cloud (points + colors), along with an overlaid bounding box
    (drawn as thick red lines formed by additional points), into a single .ply file per cloud.

    Args:
      point_clouds_points : list of np.ndarray of shape (N,3)
      point_clouds_colors : list of np.ndarray of shape (N,3) in [0,1] for R,G,B
      poses               : list of np.ndarray of shape (4,4), each an SE(3) transform
                            placing the bounding box in the global coordinate system
      box_dims            : (3,) array for bounding box (X, Y, Z) dimensions
      output_dir          : directory to save .ply files

    The .ply file will contain:
      - The original point cloud in its original colors
      - Additional points forming "thick lines" along the box edges, colored bright red.
    """

    os.makedirs(output_dir, exist_ok=True)
    half_dims = 0.5 * box_dims

    # We'll sample each of the 12 edges with a certain resolution to appear "thick"
    # in the final point set. Increase N_samples for a denser line.
    N_samples = 15

    # For convenience, define the 8 corners of the box in its local coordinate system:
    #    [±half_dims[0], ±half_dims[1], ±half_dims[2]]
    # Then define the 12 edges as pairs of corners.
    # corners = all 8 sign combinations
    corners_local = np.array([
        [ half_dims[0],  half_dims[1],  half_dims[2]],
        [ half_dims[0],  half_dims[1], -half_dims[2]],
        [ half_dims[0], -half_dims[1],  half_dims[2]],
        [ half_dims[0], -half_dims[1], -half_dims[2]],
        [-half_dims[0],  half_dims[1],  half_dims[2]],
        [-half_dims[0],  half_dims[1], -half_dims[2]],
        [-half_dims[0], -half_dims[1],  half_dims[2]],
        [-half_dims[0], -half_dims[1], -half_dims[2]],
    ])

    # Define edges as pairs of indices into the corners array
    # so we can sample line segments between them.
    edges = [
        (0, 1), (0, 2), (0, 4),  # edges from corner 0
        (7, 6), (7, 5), (7, 3),  # edges from corner 7
        (1, 3), (1, 5),         # edges that connect top face
        (2, 3), (2, 6),         # edges that connect top face
        (4, 5), (4, 6),         # edges that connect bottom face
    ]
    # (Note: there are multiple ways to define the 12 edges, this is one arrangement.)

    # We'll create a helper function to sample along an edge:
    def sample_edge_points(p0, p1, n_samples=10):
        """
        Returns n_samples points uniformly sampled on the line segment from p0 to p1.
        """
        t_vals = np.linspace(0.0, 1.0, n_samples)
        return (1 - t_vals)[:, None] * p0 + t_vals[:, None] * p1

    for i, (pts, cols) in enumerate(zip(point_clouds_points, point_clouds_colors)):
        pose = poses[i]
        # If the pose is identity (or the bounding box doesn't apply), we still show the identity box

        # Decompose pose
        R = pose[:3, :3]  # 3x3 rotation
        t = pose[:3, 3]   # 3x1 translation vector

        # Construct bounding-box line points in the global frame
        # We'll store them in a list, along with bright red color
        bbox_line_points = []
        for (idx0, idx1) in edges:
            corner0 = corners_local[idx0]
            corner1 = corners_local[idx1]
            # Sample along this edge
            local_line = sample_edge_points(corner0, corner1, N_samples)
            # Transform to global: p_global = R * p_local + t
            line_global = (local_line @ R.T) + t
            bbox_line_points.append(line_global)

        bbox_line_points = np.concatenate(bbox_line_points, axis=0)  # shape (~ 12*N_samples, 3)
        # All red color
        bbox_line_colors = np.tile([1.0, 0.0, 0.0], (bbox_line_points.shape[0], 1))

        # Combine with the original cloud
        combined_points = np.vstack((pts, bbox_line_points))
        combined_colors = np.vstack((cols, bbox_line_colors))

        # Save to PLY
        filename = os.path.join(output_dir, f"pointcloud_{i:04d}.ply")
        with open(filename, 'w') as f:
            # ASCII PLY header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(combined_points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")

            # Write out each point
            for (x, y, z), (r, g, b) in zip(combined_points, combined_colors):
                rr = int(255 * r)
                gg = int(255 * g)
                bb = int(255 * b)
                f.write(f"{x} {y} {z} {rr} {gg} {bb}\n")

        print(f"Saved {filename}")

def imitate_trajectory_with_action_identifier(
    dataset_path="/home/yilong/Documents/policy_data/lift/lift_smaller_2000",
    hand_mesh="",
    output_dir="/home/yilong/Documents/action_extractor/debug/megapose_lift_smaller_2000",
    num_demos=100,
    save_webp=False,
    cameras: list[str] = ["squared0view_image", "sidetableview_image", "squared0view2_image"],
    absolute_actions=True
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
    # 2) Load the pose estimation model once.
    model_name = "megapose-1.0-RGB-multi-hypothesis-icp"
    model_info = NAMED_MODELS[model_name]
    logger.info(f"Loading model {model_name} once at script start.")

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
                    
            all_hand_poses = get_poses_from_pointclouds(point_clouds_points, point_clouds_colors, hand_mesh)

            # 12) Build absolute actions.
            # (Assume you have updated a function to combine poses from an arbitrary number of cameras.)
            if absolute_actions:
                actions_for_demo = poses_to_absolute_actions(
                    poses=all_hand_poses,
                    gripper_actions=[root_z["data"][demo]['actions'][i][-1] for i in range(num_samples)],
                    env=env_camera0,  # using camera0 environment to get initial orientation
                    smooth=True
                )
            else:
                actions_for_demo = poses_to_delta_actions(
                    poses=all_hand_poses,
                    gripper_actions=[root_z["data"][demo]['actions'][i][-1] for i in range(num_samples)],
                    smooth=True
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
        hand_mesh="/home/yilong/Documents/action_extractor/action_extractor/megapose/panda_hand_mesh/panda-hand.ply",
        output_dir="/home/yilong/Documents/action_extractor/debug/pointcloud_no_opt_thresh_best_iterations2_000_000",
        num_demos=100,
        save_webp=False,
        absolute_actions=True,
    )