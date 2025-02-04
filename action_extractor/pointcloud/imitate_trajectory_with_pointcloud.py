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

def get_poses_from_pointclouds(point_clouds_points, point_clouds_colors):
    """
    For each point cloud, find the 6DOF pose (4x4 matrix in SE(3)) of a fixed–sized 
    bounding box that maximizes the number of “very green” points inside.

    We:
      1) Filter for "very green" points.
      2) Use PCA on those green points to get a principal-axis alignment (3x3 rotation).
      3) Transform points to that local frame.
      4) For each axis, use a 1D sliding-window approach to find intervals of length 
         box_dims[i] that contain the maximum # of points (in that axis).
      5) Cross the best intervals from x, y, z to form candidate 3D centers in the local frame.
      6) Pick the center that yields the best coverage. Then transform back to the world frame.

    Returns a list of 4x4 numpy arrays (SE(3) poses).
    If no green points in a given cloud, we return identity for that entry.
    """

    poses = []
    # fixed bounding box dimensions
    box_dims  = np.array([0.063045, 0.204516, 0.091946])
    half_dims = 0.5 * box_dims

    # thresholds for selecting "very green" points
    green_threshold = 0.9
    non_green_max   = 0.7

    # --------------------------------------------------------
    # Helper: 1D sliding-window to find intervals of length L
    # that contain the maximum # of points. We'll return the
    # intervals (by their center) that achieve this max coverage.
    # --------------------------------------------------------
    def best_1d_intervals(coords, L):
        """
        coords : (N,) array, sorted ascending
        L      : length of the bounding box in 1D
        Returns: list of center positions that yield the maximum coverage
        """
        N = len(coords)
        if N == 0:
            return [0.0]  # trivial fallback

        left = 0
        max_count = 0
        best_centers = []

        for right in range(N):
            # Move left pointer while the interval [coords[left], coords[right]] is bigger than L
            while coords[right] - coords[left] > L:
                left += 1
            # Now the interval from coords[left] to coords[right] is <= L
            window_count = right - left + 1
            if window_count > max_count:
                max_count = window_count
                # The best center in 1D for an interval [a, b] of length <= L can be anywhere
                # between (a + L/2) and (b - L/2). A simple choice is the midpoint of [a, b].
                a = coords[left]
                b = coords[right]
                center = 0.5 * (a + b)
                best_centers = [center]
            elif window_count == max_count:
                # Add the center for this interval as well
                a = coords[left]
                b = coords[right]
                center = 0.5 * (a + b)
                best_centers.append(center)

        # remove duplicates (just in case)
        best_centers = list(set(best_centers))
        return best_centers

    # Loop over each point cloud
    for pts, cols in zip(point_clouds_points, point_clouds_colors):
        # 1) Filter green points
        green_mask = (
            (cols[:, 1] > green_threshold) &
            (cols[:, 0] < non_green_max) &
            (cols[:, 2] < non_green_max)
        )
        green_pts = pts[green_mask]

        if len(green_pts) == 0:
            # No green points => identity pose
            poses.append(np.eye(4))
            continue

        # 2) PCA orientation
        mean = np.mean(green_pts, axis=0)
        cov  = np.cov(green_pts.T)
        e_vals, e_vecs = np.linalg.eigh(cov)
        order = e_vals.argsort()[::-1]
        e_vecs = e_vecs[:, order]
        # ensure right-handed
        if np.linalg.det(e_vecs) < 0:
            e_vecs[:, 2] *= -1
        R = e_vecs  # local->global rotation

        # 3) Transform green points into local frame
        local_pts = (green_pts - mean) @ R  # shape: (N_g, 3)

        # 4) For each axis i, find 1D intervals that yield max coverage
        # Sort points along each local axis first
        sorted_x = np.sort(local_pts[:, 0])
        sorted_y = np.sort(local_pts[:, 1])
        sorted_z = np.sort(local_pts[:, 2])

        candidates_x = best_1d_intervals(sorted_x, box_dims[0])
        candidates_y = best_1d_intervals(sorted_y, box_dims[1])
        candidates_z = best_1d_intervals(sorted_z, box_dims[2])

        # 5) Combine the candidate centers for x, y, z
        best_count = -1
        best_center_local = np.zeros(3)

        for cx in candidates_x:
            for cy in candidates_y:
                for cz in candidates_z:
                    center_local = np.array([cx, cy, cz])
                    # Count how many local_pts are inside +/- half_dims
                    diff = local_pts - center_local
                    inside_mask = np.all(np.abs(diff) <= half_dims, axis=1)
                    ccount = np.count_nonzero(inside_mask)
                    if ccount > best_count:
                        best_count = ccount
                        best_center_local = center_local

        # 6) Convert best center back to global coords
        box_center_global = mean + R @ best_center_local

        # Build homogeneous transform
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3]  = box_center_global
        poses.append(T)

    return poses


# def get_poses_from_pointclouds(point_clouds_points, point_clouds_colors):
#     """
#     Given a list of point-clouds (points and corresponding colors),
#     finds, for each point-cloud, a 4x4 SE(3) pose matrix of a fixed-size
#     bounding box that encloses the maximum number of 'very green' points.
    
#     :param point_clouds_points: list of arrays, where each array is shape (N, 3)
#                                 representing (x, y, z) for N points.
#     :param point_clouds_colors: list of arrays, where each array is shape (N, 3)
#                                 representing color channels (R, G, B) in [0, 1]
#                                 for the same N points.
#     :return: a list of 4x4 numpy arrays, each the SE(3) pose of the bounding box
#     """
    
#     # Fixed bounding box dimensions (X, Y, Z)
#     box_dims = np.array([0.063045, 0.204516, 0.091946])
    
#     # Number of random orientations to sample
#     num_random_orientations = 200  # Increase for better coverage (but more computation)
    
#     poses = []
    
#     for points, colors in zip(point_clouds_points, point_clouds_colors):
#         # ----------------------------------------------------
#         # 1) Filter to get the 'very green' points
#         # ----------------------------------------------------
#         # Example thresholding; adjust to your needs/data:
#         #   - G significantly larger than R and B
#         #   - G absolutely above a certain threshold
#         R = colors[:, 0]
#         G = colors[:, 1]
#         B = colors[:, 2]
        
#         # A simple "very green" condition:
#         green_mask = (G > 0.5) & ((G - R) > 0.2) & ((G - B) > 0.2)
        
#         green_points = points[green_mask]
        
#         # If there are no green points at all, just return an identity pose
#         # or some default. We'll do that as a fallback.
#         if len(green_points) == 0:
#             poses.append(np.eye(4))
#             continue
        
#         # ----------------------------------------------------
#         # 2) Randomly sample orientations in SO(3).
#         #    We'll sample random Euler angles for illustration.
#         # ----------------------------------------------------
#         # A small helper to create rotation matrix from Euler angles:
#         def euler_to_rot3(euler_angles):
#             """ Convert euler angles (rx, ry, rz) to a 3x3 rotation matrix. """
#             rx, ry, rz = euler_angles
#             # Rotation about X
#             Rx = np.array([
#                 [1,             0,              0],
#                 [0,  np.cos(rx),   -np.sin(rx)],
#                 [0,  np.sin(rx),    np.cos(rx)]
#             ])
#             # Rotation about Y
#             Ry = np.array([
#                 [ np.cos(ry), 0, np.sin(ry)],
#                 [          0, 1,          0],
#                 [-np.sin(ry), 0, np.cos(ry)]
#             ])
#             # Rotation about Z
#             Rz = np.array([
#                 [np.cos(rz), -np.sin(rz), 0],
#                 [np.sin(rz),  np.cos(rz), 0],
#                 [         0,           0, 1]
#             ])
#             return Rz @ Ry @ Rx
        
#         # Pre-generate a list of random rotations:
#         random_orientations = []
#         for _ in range(num_random_orientations):
#             # Sample each euler angle in [-pi, pi], for instance
#             rx = np.random.uniform(-np.pi, np.pi)
#             ry = np.random.uniform(-np.pi, np.pi)
#             rz = np.random.uniform(-np.pi, np.pi)
#             R_rand = euler_to_rot3((rx, ry, rz))
#             random_orientations.append(R_rand)
        
#         # Also add an identity orientation as a candidate
#         random_orientations.append(np.eye(3))
        
#         # ----------------------------------------------------
#         # 3) For each orientation, transform points, then
#         #    find the best 3D axis-aligned bounding box center
#         #    of size `box_dims` in that rotated space that
#         #    encloses the maximum green points.
#         # ----------------------------------------------------
        
#         max_inliers = 0
#         best_overall_center = None
#         best_overall_rotation = None
        
#         # We'll define half-dims for convenience
#         half_dims = 0.5 * box_dims
        
#         # A function to count how many points are in the box
#         # from (center - half_dims) to (center + half_dims).
#         def count_inliers(rot_pts, candidate_center):
#             """
#             rot_pts: Nx3 points in the rotated frame
#             candidate_center: the center of the bounding box (in rotated frame)
#             """
#             lower = candidate_center - half_dims
#             upper = candidate_center + half_dims
            
#             mask = (
#                 (rot_pts[:, 0] >= lower[0]) & (rot_pts[:, 0] <= upper[0]) &
#                 (rot_pts[:, 1] >= lower[1]) & (rot_pts[:, 1] <= upper[1]) &
#                 (rot_pts[:, 2] >= lower[2]) & (rot_pts[:, 2] <= upper[2])
#             )
#             return np.count_nonzero(mask)
        
#         # A helper to get candidate 1D intervals of length L that cover the
#         # max number of points. Returns a small set of intervals that achieve
#         # near-maximum coverage. We'll do a standard sliding approach on sorted coords.
#         def best_1d_intervals(coords, L):
#             """
#             coords: 1D numpy array of the coordinate to search over
#             L: the length of the bounding box dimension
#             returns: list of (start, end) intervals that have near-max coverage
#             """
#             sorted_coords = np.sort(coords)
#             N = len(sorted_coords)
#             if N == 0:
#                 return [(0.0, L)]  # trivial fallback
            
#             best_count = 0
#             best_intervals = []
#             left = 0
#             for right in range(N):
#                 # Move left pointer while interval is too large
#                 while sorted_coords[right] - sorted_coords[left] > L:
#                     left += 1
#                 # Now [sorted_coords[left], sorted_coords[right]] <= L
#                 window_count = (right - left + 1)
#                 if window_count > best_count:
#                     best_count = window_count
#                     best_intervals = [(sorted_coords[left], sorted_coords[left] + L)]
#                 elif window_count == best_count:
#                     best_intervals.append((sorted_coords[left], sorted_coords[left] + L))
            
#             return best_intervals
        
#         # Main search over orientations
#         for R_candidate in random_orientations:
#             # Transform green points into this orientation
#             # We'll treat the origin as (0,0,0) for rotation only
#             rotated_points = green_points @ R_candidate.T  # shape (G, 3)
            
#             # For each dimension, find best intervals of length box_dims[d]
#             x_intervals = best_1d_intervals(rotated_points[:, 0], box_dims[0])
#             y_intervals = best_1d_intervals(rotated_points[:, 1], box_dims[1])
#             z_intervals = best_1d_intervals(rotated_points[:, 2], box_dims[2])
            
#             # We form candidate centers by taking midpoints from each dimension
#             # We'll combine them in a small cross-product to avoid enumerating too many.
#             # If each dimension has M_x, M_y, M_z best intervals, we get M_x*M_y*M_z combos
#             candidate_centers = []
#             for (x_start, x_end) in x_intervals:
#                 x_center = 0.5 * (x_start + x_end)
#                 for (y_start, y_end) in y_intervals:
#                     y_center = 0.5 * (y_start + y_end)
#                     for (z_start, z_end) in z_intervals:
#                         z_center = 0.5 * (z_start + z_end)
#                         candidate_centers.append(np.array([x_center, y_center, z_center]))
            
#             # Evaluate each candidate center
#             for c_center in candidate_centers:
#                 # Count how many green points fall inside the box
#                 inliers = count_inliers(rotated_points, c_center)
#                 if inliers > max_inliers:
#                     max_inliers = inliers
#                     best_overall_center = c_center
#                     best_overall_rotation = R_candidate
        
#         # ----------------------------------------------------
#         # 4) We now have best rotation and best center in the
#         #    rotated space. We must convert center back to the
#         #    original coordinate system. That is:
#         #
#         #    p_in_original = R * p_in_rotated
#         #
#         #    But c_center is in the rotated frame, so the actual
#         #    center c_orig = R_candidate * c_center.
#         # ----------------------------------------------------
#         if best_overall_rotation is None:
#             # fallback if something went wrong
#             poses.append(np.eye(4))
#             continue
        
#         R_final = best_overall_rotation
#         c_rotated = best_overall_center
#         c_final = R_final @ c_rotated  # Convert center back to original frame
        
#         # ----------------------------------------------------
#         # 5) Construct the 4x4 pose matrix
#         # ----------------------------------------------------
#         T = np.eye(4)
#         T[:3, :3] = R_final
#         T[:3, 3] = c_final
        
#         poses.append(T)
    
#     return poses


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
            
            save_pointclouds_with_bbox_as_ply(
                point_clouds_points,
                point_clouds_colors,
                all_hand_poses,
                box_dims=np.array([0.063045, 0.204516, 0.091946]),
                output_dir="debug/pointcloud_traj"
            )
            
            # debug_pointcloud_poses(point_clouds_points[:10], point_clouds_colors[:10], output_dir=os.path.join(output_dir, "pointcloud_debug"))

            # 12) Build absolute actions.
            # (Assume you have updated a function to combine poses from an arbitrary number of cameras.)
            actions_for_demo = poses_to_absolute_actions(
                poses=all_hand_poses,
                gripper_actions=[root_z["data"][demo]['actions'][i][-1] for i in range(num_samples)],
                env=env_camera0,  # using camera0 environment for execution
                smooth=False
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