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
import copy
import random

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

def load_model_as_pointcloud(model_path, num_points=30000, model_in_mm=True):
    mesh = o3d.io.read_triangle_mesh(model_path)
    if mesh.is_empty():
        raise ValueError(f"Could not load mesh from: {model_path}")

    if model_in_mm:
        mesh.scale(0.001, center=(0,0,0))

    pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    return pcd

def cluster_and_keep_largest(pcd_o3d, eps=0.02, min_points=20):
    if len(pcd_o3d.points) == 0:
        return pcd_o3d

    labels = np.array(pcd_o3d.cluster_dbscan(eps=eps, min_points=min_points))
    if len(labels) == 0 or np.all(labels < 0):
        return o3d.geometry.PointCloud()

    unique_labels, counts = np.unique(labels, return_counts=True)
    valid_mask = (unique_labels >= 0)
    if not np.any(valid_mask):
        return o3d.geometry.PointCloud()
    valid_labels = unique_labels[valid_mask]
    valid_counts = counts[valid_mask]
    largest_label = valid_labels[np.argmax(valid_counts)]
    indices_largest = np.where(labels == largest_label)[0]
    return pcd_o3d.select_by_index(indices_largest)

# Helper to convert a 3x3 rotation matrix to a quaternion (x,y,z,w)
def matrix_to_quat(R_mat):
    R_mat_copy = np.array(R_mat, copy=True, dtype=np.float64)
    rot = R.from_matrix(R_mat_copy)
    q = rot.as_quat()  # [x, y, z, w]
    return q

# Compute the orientation difference (in radians) between a rotation matrix R_mat
# and a base quaternion q_base (x,y,z,w).
def orientation_angle_diff(R_mat, q_base):
    q_curr = matrix_to_quat(R_mat)
    # Dot product of unit quaternions => cos(half the angle)
    dot = np.abs(np.dot(q_curr, q_base))
    dot = np.clip(dot, -1.0, 1.0)
    angle = 2.0 * np.arccos(dot)
    return angle

def get_poses_from_pointclouds(point_clouds_points,
                               point_clouds_colors,
                               model_path,
                               green_threshold=0.9,
                               non_green_max=0.7,
                               voxel_size=0.002,
                               mesh_num_points=30000,
                               debug_dir="debug/pointclouds_with_model",
                               model_in_mm=True,
                               dbscan_eps=0.02,
                               dbscan_min_points=20,
                               base_orientation_quat=np.array([0.7253942, 0.6844675, 0.05062998, 0.05238412]),
                               max_orientation_angle=np.pi/2
                               ):
    """
    For every frame:
      - If it's the first frame (i=0), do multi-seed RANSAC + pick only solutions
        whose orientation is within `max_orientation_angle` of 'base_orientation_quat'.
        If all solutions are out of range, we re-try RANSAC multiple times.
        If still none found, fallback to identity.
      - Then refine with multi-scale ICP.
      - For subsequent frames, we do the same multi-seed logic (but we do NOT
        filter by base orientation, or you can keep that logic if you want).
        (You can easily adapt to skip or apply the orientation filter for later frames too.)

    :param base_orientation_quat: The reference orientation as [x, y, z, w].
    :param max_orientation_angle: The maximum angle (radians) difference from base.
    """

    os.makedirs(debug_dir, exist_ok=True)

    object_model_o3d = load_model_as_pointcloud(model_path,
                                                num_points=mesh_num_points,
                                                model_in_mm=model_in_mm)

    def preprocess_pcd(pcd, voxel):
        if voxel > 0:
            pcd_down = pcd.voxel_down_sample(voxel)
        else:
            pcd_down = pcd

        pcd_down.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=2.0 * voxel if voxel > 0 else 0.01,
                max_nn=30
            )
        )
        return pcd_down

    def multi_scale_icp(source, target, init_trans):
        thresholds = [10.0 * voxel_size, 5.0 * voxel_size, 2.0 * voxel_size, 1.0 * voxel_size]
        iterations = [1000, 20000, 20000, 20000]

        current_transform = init_trans
        for idx, (dist, max_iter) in enumerate(zip(thresholds, iterations)):
            icp_criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=max_iter,
                relative_fitness=1e-6,
                relative_rmse=1e-6
            )

            result_icp = o3d.pipelines.registration.registration_icp(
                source,
                target,
                dist,
                current_transform,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                criteria=icp_criteria
            )
            current_transform = result_icp.transformation
            print(f"  [Multi-Scale ICP] Level {idx} (dist={dist:.4f}), "
                  f"iter={max_iter} -> fitness={result_icp.fitness:.4f}, "
                  f"inlier_rmse={result_icp.inlier_rmse:.5f}")

        return current_transform
    
    def multi_scale_icp_early_stop(source, target, init_trans):
        """
        Multi-scale ICP that stops early within each scale if the transform/fitness stops improving.
        """
        import numpy as np
        import open3d as o3d

        thresholds = [10.0 * voxel_size, 5.0 * voxel_size, 2.0 * voxel_size, 1.0 * voxel_size]
        max_iters_per_scale = [1000, 20000, 20000, 20000]

        # Criteria for "no further improvement"
        transform_epsilon = 1e-7
        fitness_epsilon   = 1e-7
        min_iters         = 50          # allow at least some iterations
        stable_iters_req  = 15          # number of consecutive stable iterations

        current_transform = init_trans

        for idx, (dist, max_iter) in enumerate(zip(thresholds, max_iters_per_scale)):

            print(f"\n  [EarlyStopICP] Scale {idx} threshold={dist:.4f}, max_iter={max_iter}")
            consecutive_stable_iters = 0
            prev_fitness   = 0.0
            prev_transform = current_transform.copy()

            for iter_i in range(max_iter):
                # Run exactly one iteration of ICP
                result_icp = o3d.pipelines.registration.registration_icp(
                    source, target,
                    dist,
                    current_transform,
                    o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                    criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1)
                )
                new_transform = result_icp.transformation
                fitness = result_icp.fitness

                # Evaluate changes
                delta_transform = np.linalg.norm(new_transform - current_transform)
                delta_fitness   = abs(fitness - prev_fitness)

                current_transform = new_transform
                prev_fitness      = fitness

                # Check if changes are below thresholds
                if delta_transform < transform_epsilon and delta_fitness < fitness_epsilon and iter_i > min_iters:
                    consecutive_stable_iters += 1
                else:
                    consecutive_stable_iters = 0

                if consecutive_stable_iters >= stable_iters_req:
                    print(f"    Early stop at iteration={iter_i}, fitness={fitness:.4f}")
                    break

            # Print final result at this scale
            print(f"    Scale {idx} final => fitness={prev_fitness:.4f}")

        return current_transform
    
    def progressive_icp(source, target, init_trans,
                    start_dist, end_dist,
                    steps=5, max_iter=2000):
        """
        A single-loop approach: we gradually reduce the distance threshold from 'start_dist' 
        to 'end_dist' over 'steps' passes, each pass runs up to 'max_iter' but can exit early.
        """

        import numpy as np
        import open3d as o3d

        current_transform = init_trans

        def adaptive_icp_pass(src, tgt, init_tr, dist_thr, max_it=2000):
            # This pass can be similar to the early-stop approach from above,
            # or a simpler fixed iteration approach. We'll do a simpler version here:
            icp_criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=max_it, relative_fitness=1e-6, relative_rmse=1e-6
            )
            result = o3d.pipelines.registration.registration_icp(
                src, tgt, dist_thr, init_tr,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                criteria=icp_criteria
            )
            return result.transformation, result.fitness

        for step in range(steps):
            # Compute the threshold for this step (geometric interpolation or linear)
            alpha = step / float(steps - 1)
            current_dist = start_dist * (end_dist / start_dist) ** alpha   # geometric
            # Or do a linear approach: current_dist = start_dist + alpha*(end_dist - start_dist)

            print(f"\n  [ProgressiveICP] step={step}, dist={current_dist:.4f}, max_iter={max_iter}")
            new_transform, fitness = adaptive_icp_pass(source, target, current_transform, current_dist, max_iter)
            current_transform = new_transform
            print(f"    => step {step} done. fitness={fitness:.4f}")

        return current_transform
    
    def multi_scale_icp_dynamic_skip(source, target, init_trans):
        """
        Similar to your existing multi-scale approach, but if we hit a high fitness 
        or minimal transform change, we skip the next levels or reduce iteration.
        """

        import numpy as np
        import open3d as o3d

        thresholds = [10.0 * voxel_size, 5.0 * voxel_size, 2.0 * voxel_size, 1.0 * voxel_size]
        iterations = [1000, 20000, 20000, 20000]

        fitness_skip = 0.99  # If we exceed this, skip the next levels
        transform_epsilon = 1e-7

        current_transform = init_trans
        prev_transform    = init_trans

        for idx, (dist, max_iter) in enumerate(zip(thresholds, iterations)):
            if idx > 0:
                # Compare to see if we want to skip
                # if the last iteration had high fitness or we changed the transform very little
                delta_t = np.linalg.norm(current_transform - prev_transform)
                # We'll do a smaller ICP pass
                if last_fitness > fitness_skip or delta_t < transform_epsilon:
                    print(f"  [DynamicSkipICP] Scale {idx}: skipping because fitness={last_fitness:.4f} or delta_t={delta_t:.1e}")
                    continue

            # If not skipping, proceed
            icp_criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=max_iter,
                relative_fitness=1e-6,
                relative_rmse=1e-6
            )

            result_icp = o3d.pipelines.registration.registration_icp(
                source,
                target,
                dist,
                current_transform,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                criteria=icp_criteria
            )
            prev_transform  = current_transform
            current_transform = result_icp.transformation
            last_fitness   = result_icp.fitness
            print(f"  [DynamicSkipICP] Level {idx} (dist={dist:.4f}), "
                f"iter={max_iter} -> fitness={result_icp.fitness:.4f}, "
                f"inlier_rmse={result_icp.inlier_rmse:.5f}")

        return current_transform
    
    def multi_scale_icp_guarantee_fitness(source, target, init_trans,
                                      desired_fitness=0.99,
                                      max_scale_retries=3):
        """
        Multi-scale ICP that attempts to ensure each scale
        hits at least 'desired_fitness' before proceeding.
        
        If the scale finishes below that fitness, we increase
        the iteration count and re-run the same scale.
        If after 'max_scale_retries' expansions it still fails,
        we move on anyway (or we can just remain stuck).
        
        This can help ensure each scale is "maxed out" in alignment
        before going finer.
        """

        import open3d as o3d
        import numpy as np

        # Original thresholds
        thresholds = [10.0 * voxel_size, 5.0 * voxel_size, 2.0 * voxel_size, 1.0 * voxel_size]
        # Starting iteration counts
        iterations = [1000, 20000, 20000, 20000]

        current_transform = init_trans

        for idx, (dist, base_iter) in enumerate(zip(thresholds, iterations)):

            print(f"\n[GuaranteeFitnessICP] Scale {idx}, dist={dist:.4f}, base_iter={base_iter}")
            # We'll attempt re-runs if we fail to meet desired fitness.
            scale_pass_iter = base_iter
            best_fitness = 0.0
            best_transform = current_transform.copy()

            for retry_i in range(max_scale_retries):
                # Build the ICP criteria with the current iteration count
                icp_criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=scale_pass_iter,
                    relative_fitness=1e-6,
                    relative_rmse=1e-6
                )

                result_icp = o3d.pipelines.registration.registration_icp(
                    source,
                    target,
                    dist,
                    best_transform,  # start from the best transform so far
                    o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                    criteria=icp_criteria
                )

                new_fitness = result_icp.fitness
                new_transform = result_icp.transformation

                print(f"  Scale {idx}, Retry {retry_i}, iter={scale_pass_iter} -> fitness={new_fitness:.4f}, rmse={result_icp.inlier_rmse:.5f}")

                if new_fitness > best_fitness:
                    best_fitness = new_fitness
                    best_transform = new_transform

                # Check if we have reached the desired fitness
                if best_fitness >= desired_fitness:
                    print(f"    Achieved desired fitness {best_fitness:.4f} >= {desired_fitness} => proceed.")
                    break
                else:
                    # Otherwise, expand iteration count for the next try
                    scale_pass_iter = int(scale_pass_iter * 1.5)  # or double, etc.
                    print(f"    Not reached {desired_fitness} fitness => increase iteration to {scale_pass_iter} and re-try...")

            # After possibly multiple retries, we store the best result
            current_transform = best_transform

        return current_transform
    
    def icp_threshold_updown_force_one(
        source,
        target,
        init_transform,
        max_total_iterations=2_000_000,
        debug=True
    ):
        """
        Implements this strategy with factor=1.5 up, factor=0.75 down, ignoring final pass fitness:
        - Start threshold = 8*voxel_size
        - Do an ICP pass:
            if fitness=1 => threshold *= 0.75
            else => threshold *= 1.5
        - Keep repeating until threshold < 1.2 * voxel_size
        - Then do ONE final pass at that threshold, ignoring fitness in the final result.

        The iteration formula for each pass is:
            iteration_count = 10000 * (voxel_size/threshold) * 8 * (1 + 0.1 * times_visited[threshold]),
        plus we clamp to 'max_total_iterations' in total usage across all passes.

        :param source, target: Open3D point clouds
        :param init_transform: initial (4x4) guess
        :param voxel_size: base measure => final threshold ~1 * voxel_size
        :param max_total_iterations: sum of all iteration usage. If we exceed, we must stop.
        :param debug: if True, print threshold, iteration, fitness, rmse each pass.
        :return: final transform from the last pass at threshold < 1.2*voxel_size, ignoring that pass’s fitness
        """

        import open3d as o3d
        import numpy as np

        # We track how many times we've visited each threshold (float) to raise iteration with repeated visits
        times_visited = {}
        total_iterations_used = 0

        # Helper: compute iteration count by your formula
        def get_iteration_count(threshold):
            """
            iteration_count = 10000 * (voxel_size / threshold) * 8 * (1 + 0.1 * times_visited[threshold])
            """
            t_key = float(threshold)
            vcount = times_visited.get(t_key, 0)
            it_count = 10000 * (voxel_size / threshold) * 8 * (1 + 0.1 * vcount)
            return max(1, int(it_count))

        def run_icp(threshold, init_pose):
            """
            Run one ICP pass at the given threshold, with iteration_count from get_iteration_count(...).
            Increase times_visited and clamp usage to max_total_iterations if needed.
            """
            nonlocal total_iterations_used

            # increment times_visited
            t_key = float(threshold)
            old_visits = times_visited.get(t_key, 0)
            times_visited[t_key] = old_visits + 1

            it_count = get_iteration_count(threshold)

            # clamp if we exceed total budget
            if total_iterations_used + it_count > max_total_iterations:
                it_count = max_total_iterations - total_iterations_used
                if it_count <= 0:
                    # no iterations left => return dummy
                    return 0.0, 999999.0, init_pose, 0

            crit = o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=it_count,
                relative_fitness=1e-6,
                relative_rmse=1e-6
            )
            result_icp = o3d.pipelines.registration.registration_icp(
                source,
                target,
                threshold,
                init_pose,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                criteria=crit
            )

            total_iterations_used += it_count
            return result_icp.fitness, result_icp.inlier_rmse, result_icp.transformation, it_count

        # 1) Start
        current_transform = init_transform.copy()
        best_transform = current_transform.copy()
        best_fitness = 0.0
        threshold = 8.0 * voxel_size
        best_threshold = threshold
        best_rmse = 1.0

        # 2) Up/Down logic: if fitness=1 => threshold*=0.75, else => threshold*=1.5
        # Keep going until threshold < 1.2 * voxel_size
        while threshold >= 1.75 * voxel_size:
            fitness, rmse, new_transform, used_iters = run_icp(threshold, current_transform)  

            if debug:
                print(f"[UpDownICP] threshold={threshold:.4f}, used_iters={used_iters}, fitness={fitness:.4f}, rmse={rmse:.5f}")

            if used_iters <= 0:
                if debug:
                    print("No iteration budget left, break up/down loop.")
                break

            if fitness >= 0.96 - 1e-12:
                # success => threshold *= 0.75
                threshold *= 0.5
                current_transform = new_transform
                
                if threshold < best_threshold:
                    best_transform = current_transform.copy()
                    best_threshold = threshold
                    best_fitness = fitness
                    best_rmse = rmse
            else:
                # fail => threshold *= 1.5
                threshold *= 1.5
                current_transform = new_transform

        # 3) We do one final pass at the last threshold, ignoring fitness
        # As per instructions, "the final result comes from the result at threshold < 1.2*voxel_size"
        if threshold >= 1.75 * voxel_size:
            # If we never shrank below 1.2 => just return best transform
            if debug:
                print("Threshold never shrank below 1.75 * voxel_size.")
                print(f"[UpDownICP] Best threshold={best_threshold:.4f}, used_iters={max_total_iterations}, fitness={best_fitness:.4f}, rmse={best_rmse:.5f}")
                
            return best_transform
        else:
            if debug:
                print(f"Threshold < 1.75 * voxel_size => final pass at threshold={threshold:.4f} ignoring fitness...")

        # final pass
        f_final, rmse_final, final_transform, used_iters_final = run_icp(threshold, current_transform)
        if debug:
            print(f"[UpDownICP] Final pass threshold={threshold:.4f}, used_iters={used_iters_final}, fitness={f_final:.4f}, rmse={rmse_final:.5f}")
            print("Ignoring final pass fitness. Returning transform anyway.")

        return final_transform

    def compute_fpfh(pcd, voxel):
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=2.0 * voxel if voxel > 0 else 0.01,
                max_nn=30
            )
        )
        radius_feature = 5.0 * voxel if voxel > 0 else 0.05
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
        )
        return fpfh

    def global_registration(source, target, source_fpfh, target_fpfh, seed):
        o3d.utility.random.seed(seed)
        dist_thresh = max(voxel_size * 1.5, 0.04)
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source,
            target,
            source_fpfh,
            target_fpfh,
            mutual_filter=False,
            max_correspondence_distance=dist_thresh,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            ransac_n=4,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist_thresh)
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
                max_iteration=100000, confidence=0.99
            )
        )
        return result

    def register_green_points_to_model_first_frame(green_pts_np, model_pcd_o3d):
        """
        For the first frame: we do multi-seed RANSAC, keep only transformations
        whose orientation is within 'max_orientation_angle' of 'base_orientation_quat'.
        If all are out of range, we can re-try multiple times. If still none, fallback identity.
        """
        if len(green_pts_np) < 10:
            print("    Not enough green points (<10). Identity.")
            return np.eye(4)

        green_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(green_pts_np))
        largest_cluster = cluster_and_keep_largest(green_pcd, eps=dbscan_eps, min_points=dbscan_min_points)
        if len(largest_cluster.points) < 10:
            print("    Largest cluster <10 points, identity.")
            return np.eye(4)

        green_o3d_clean = preprocess_pcd(largest_cluster, voxel_size)
        model_o3d_clean = preprocess_pcd(model_pcd_o3d, voxel_size)

        if len(green_o3d_clean.points) < 10 or len(model_o3d_clean.points) < 10:
            print("    After cleaning, too few points. Identity.")
            return np.eye(4)

        src_fpfh = compute_fpfh(green_o3d_clean, voxel_size)
        tgt_fpfh = compute_fpfh(model_o3d_clean, voxel_size)

        # We'll allow up to 3 attempts. Each attempt tries multiple seeds.
        # If we find no orientation that passes the threshold, we try again.
        # If still none, fallback to identity.
        found_1_point_0 = False
        best_transform   = None

        attempt = 0

        while True:
            # If you want to enforce a limit, uncomment:
            # if max_tries is not None and attempt >= max_tries:
            #     raise RuntimeError("Never found fitness=1.0 after max_tries seeds!")

            # Generate a random seed
            seed = random.randint(0, 2**31 - 1)  # or a bigger range if you like
            result_ransac = global_registration(green_o3d_clean, model_o3d_clean,
                                                src_fpfh, tgt_fpfh, seed=seed)
            attempt += 1

            print(f"[Try #{attempt}] RANSAC seed={seed} -> fitness={result_ransac.fitness:.4f}, "
                f"rmse={result_ransac.inlier_rmse:.5f}")

            # Check orientation difference
            R_est = result_ransac.transformation[:3, :3]
            angle_diff = orientation_angle_diff(R_est, base_orientation_quat)
            if angle_diff <= max_orientation_angle:
                # It's orientation-compatible
                if result_ransac.fitness >= 1.0 - 1e-12:
                    # Good enough => fitness ~ 1.0
                    found_1_point_0 = True
                    best_transform  = result_ransac.transformation
                    print("Found a valid transform with ~1.0 fitness. Done.")
                    break
                else:
                    print("Orientation is valid, but fitness < 1.0. Trying another seed.")
            else:
                print("Orientation is not valid, trying another seed.")

        # After we exit the loop, we either found 1.0 or we're in an infinite loop if no limit
        if found_1_point_0:
            # proceed with best_transform
            # e.g. init_transform = best_transform
            # Do multi-scale ICP
            # final_transform = multi_scale_icp(green_o3d_clean, model_o3d_clean, best_transform)
            # final_transform = multi_scale_icp_early_stop(green_o3d_clean, model_o3d_clean, best_transform)
            # final_transform = progressive_icp(green_o3d_clean, model_o3d_clean, best_transform, 100*voxel_size, 1*voxel_size, steps=5, max_iter=2000)
            final_transform = icp_threshold_updown_force_one(green_o3d_clean, model_o3d_clean, best_transform)
            # final_transform = multi_scale_icp_dynamic_skip(green_o3d_clean, model_o3d_clean, best_transform)
            return final_transform

        # If we exit the loop, means we failed all attempts
        print("    All attempts failed to find orientation near base. Fallback identity.")
        return np.eye(4)

    def register_green_points_to_model_subsequent_frames(green_pts_np, model_pcd_o3d):
        """
        For subsequent frames, we do the multi-seed RANSAC with no orientation filter.
        (You can keep the filter if you wish. Just replicate the logic above.)
        """
        if len(green_pts_np) < 10:
            return np.eye(4)

        green_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(green_pts_np))
        largest_cluster = cluster_and_keep_largest(green_pcd, eps=dbscan_eps, min_points=dbscan_min_points)
        if len(largest_cluster.points) < 10:
            return np.eye(4)

        green_o3d_clean = preprocess_pcd(largest_cluster, voxel_size)
        model_o3d_clean = preprocess_pcd(model_pcd_o3d, voxel_size)
        if len(green_o3d_clean.points) < 10 or len(model_o3d_clean.points) < 10:
            return np.eye(4)

        src_fpfh = compute_fpfh(green_o3d_clean, voxel_size)
        tgt_fpfh = compute_fpfh(model_o3d_clean, voxel_size)

        best_ransac_fitness = -1.0
        best_transform = np.eye(4)
        seeds_to_try = [0,1,2,3,4,5,6,7,8,9]
        for seed in seeds_to_try:
            result_ransac = global_registration(green_o3d_clean, model_o3d_clean,
                                                src_fpfh, tgt_fpfh, seed=seed)
            if result_ransac.fitness > best_ransac_fitness:
                best_ransac_fitness = result_ransac.fitness
                best_transform = result_ransac.transformation

        final_transform = icp_threshold_updown_force_one(green_o3d_clean, model_o3d_clean, best_transform)
        return final_transform

    poses = []
    for i, (pts, cols) in enumerate(zip(point_clouds_points, point_clouds_colors)):
        print(f"\n=== Frame {i} ===")
        mask = ((cols[:,1] >= green_threshold) &
                (cols[:,0] <= non_green_max) &
                (cols[:,2] <= non_green_max))
        green_pts = pts[mask]
        print(f"  #points={len(pts)}, #green={len(green_pts)}")

        if len(green_pts) < 10:
            print(f"  No green points, identity.")
            poses.append(np.eye(4))
            continue

        if i == 0:
            # First frame => filter orientation vs. base quaternion
            transform_green_to_model = register_green_points_to_model_first_frame(green_pts, object_model_o3d)
        else:
            # Subsequent frames => no orientation filter
            transform_green_to_model = register_green_points_to_model_subsequent_frames(green_pts, object_model_o3d)

        T_model_in_cloud = np.linalg.inv(transform_green_to_model)
        poses.append(T_model_in_cloud)

        # (Optional) Save debug .ply
        green_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(green_pts))
        green_pcd.colors = o3d.utility.Vector3dVector(np.tile([0,1,0], (len(green_pts),1)))
        model_copy = copy.deepcopy(object_model_o3d)
        model_copy.transform(T_model_in_cloud)
        if len(model_copy.points) > 0:
            model_copy.colors = o3d.utility.Vector3dVector(np.tile([1,0,0], (len(model_copy.points),1)))
        out_pcd = model_copy + green_pcd
        out_path = os.path.join(debug_dir, f"frame_{i:04d}.ply")
        o3d.io.write_point_cloud(out_path, out_pcd)
        print(f"  Saved debug PLY: {out_path}")

    return poses

def imitate_trajectory_with_action_identifier(
    dataset_path="/home/yilong/Documents/policy_data/lift/lift_smaller_2000",
    hand_mesh="",
    output_dir="/home/yilong/Documents/action_extractor/debug/megapose_lift_smaller_2000",
    num_demos=100,
    save_webp=False,
    cameras: list[str] = ["squared0view_image", "sidetableview_image", "squared0view2_image"],
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
            
            # for pose in all_hand_poses:
            #     print(pose[:3, 3])
            
            # save_pointclouds_with_bbox_as_ply(
            #     point_clouds_points,
            #     point_clouds_colors,
            #     all_hand_poses,
            #     box_dims=np.array([0.063045, 0.204516, 0.091946]),
            #     output_dir="debug/pointcloud_traj"
            # )
            
            # debug_pointcloud_poses(point_clouds_points[:10], point_clouds_colors[:10], output_dir=os.path.join(output_dir, "pointcloud_debug"))

            # 12) Build absolute actions.
            # (Assume you have updated a function to combine poses from an arbitrary number of cameras.)
            actions_for_demo = poses_to_absolute_actions(
                poses=all_hand_poses,
                gripper_actions=[root_z["data"][demo]['actions'][i][-1] for i in range(num_samples)],
                env=env_camera0,  # using camera0 environment for execution
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
        output_dir="/home/yilong/Documents/action_extractor/debug/megapose_weighted_average_squared0view12",
        num_demos=3,
        save_webp=False
    )