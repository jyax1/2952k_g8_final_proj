import os
import json
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import pandas as pd

import math

import cv2

from typing import List, Optional

from action_extractor.utils.angles import *

from megapose.utils.tensor_collection import PandasTensorCollection

from megapose.datasets.object_dataset import RigidObject, RigidObjectDataset
from megapose.datasets.scene_dataset import CameraData, ObjectData
from megapose.inference.types import DetectionsType, ObservationTensor, PoseEstimatesType
from megapose.inference.utils import make_detections_from_object_data
from megapose.lib3d.transform import Transform
from megapose.utils.load_model import NAMED_MODELS, load_named_model
from megapose.utils.logging import get_logger
from megapose.inference.pose_estimator import PoseEstimator

from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from robomimic.utils.obs_utils import DEPTH_MINMAX

logger = get_logger(__name__)

from robosuite.utils.camera_utils import (
    get_camera_extrinsic_matrix,
    get_camera_intrinsic_matrix,
)


# -----------------------------------------------------------------------------
# Make a dataset from your dedicated mesh folder
# -----------------------------------------------------------------------------

def make_object_dataset_from_folder(mesh_dir: Path) -> RigidObjectDataset:
    """Creates a RigidObjectDataset by scanning mesh_dir for .obj or .ply mesh files."""
    rigid_objects = []
    mesh_units = "mm"

    for fn in mesh_dir.glob("*"):
        if fn.suffix in {".obj", ".ply"}:
            label = fn.stem  
            rigid_objects.append(
                RigidObject(
                    label=label,
                    mesh_path=fn, 
                    mesh_units=mesh_units,
                )
            )

    if len(rigid_objects) == 0:
        raise FileNotFoundError(f"No .obj or .ply found in {mesh_dir}")

    return RigidObjectDataset(rigid_objects)


# -----------------------------------------------------------------------------
# (Optional) Save predictions if needed
# -----------------------------------------------------------------------------

def save_predictions(output_dir: Path, pose_estimates: PoseEstimatesType) -> None:
    """Saves pose estimates as JSON in output_dir/outputs/object_data.json"""
    from megapose.datasets.scene_dataset import ObjectData
    labels = pose_estimates.infos["label"]
    poses = pose_estimates.poses.cpu().numpy()
    object_data = [
        ObjectData(label=label, TWO=Transform(pose)) for label, pose in zip(labels, poses)
    ]
    object_data_json = json.dumps([x.to_json() for x in object_data])
    output_fn = output_dir / "outputs" / "object_data.json"
    output_fn.parent.mkdir(exist_ok=True) 
    output_fn.write_text(object_data_json)
    logger.info(f"Wrote predictions: {output_fn}")


# -----------------------------------------------------------------------------
# Inference function that accepts in-memory image, K, detections, etc.
# -----------------------------------------------------------------------------

def estimate_pose(image_rgb: np.ndarray, K: np.ndarray, detections: DetectionsType, 
                 pose_estimator: PoseEstimator, model_info: dict,
                 depth: Optional[np.ndarray] = None, 
                 output_dir: Optional[Path] = None) -> PoseEstimatesType:
    """Runs pose estimation on a single image with detections."""
    observation = ObservationTensor.from_numpy(image_rgb, depth, K).cuda()
    output, _ = pose_estimator.run_inference_pipeline(
        observation,
        detections=detections.cuda(),
        **model_info["inference_parameters"],
    )
    if output_dir is not None:
        save_predictions(output_dir, output)
    return output

def estimate_pose_batched(list_of_images: list[np.ndarray], list_of_bboxes: list[list[dict]],
                         K: np.ndarray, pose_estimator: PoseEstimator, model_info: dict,
                         depth_list: list[np.ndarray] = None,
                         depth_minmax: list[float] = [0, 1]) -> list[PoseEstimatesType]:
    """Runs pose estimation on multiple images in a single forward pass."""
    assert len(list_of_images) == len(list_of_bboxes), (
        "list_of_images and list_of_bboxes must have same length"
    )
    N = len(list_of_images)
    
    # ----------------------------------------------------------------
    # 1) Construct one big ObservationTensor with shape [N,3,H,W] or [N,4,H,W]
    # ----------------------------------------------------------------
    images_torch = []
    for i in range(N):
        rgb = list_of_images[i]    # shape (H,W,3) in uint8
        # convert to float [0..1], shape (H,W,3)
        rgb_torch = torch.from_numpy(rgb).float() / 255.0
        # [3,H,W]
        rgb_torch = rgb_torch.permute(2,0,1)
        # => shape (1,3,H,W)
        rgb_torch = rgb_torch.unsqueeze(0)

        if depth_list is not None:
            depth_np = depth_list[i]  # shape (H,W)
            # Assuming min_depth and max_depth are in meters, convert to millimeters
            min_depth_meters = depth_minmax[0] * 1000  # convert to millimeters
            max_depth_meters = depth_minmax[1] * 1000  # convert to millimeters
            
            # Convert depth_np to absolute depth distance in millimeters
            depth_mm = (depth_np / 255.0) * (max_depth_meters - min_depth_meters) + min_depth_meters
            
            depth_torch = torch.from_numpy(depth_mm).float()  # (H,W)
            depth_torch = depth_torch.permute(2,0,1).unsqueeze(0)  # (1,1,H,W)
            img_4ch = torch.cat([rgb_torch, depth_torch], dim=1)  # shape (1,4,H,W)
            images_torch.append(img_4ch)
        else:
            images_torch.append(rgb_torch)  # shape (1,3,H,W)

    # stack along batch dimension => shape (N,C,H,W)
    images_batched = torch.cat(images_torch, dim=0)  

    # replicate K => shape (N,3,3)
    K_torch = torch.from_numpy(K).float().unsqueeze(0).expand(N, -1, -1).clone()

    # create ObservationTensor and move to GPU
    observation = ObservationTensor(images_batched, K_torch).cuda()

    # ----------------------------------------------------------------
    # 2) Build a single DetectionsType containing all bounding boxes
    # ----------------------------------------------------------------
    # We'll create a DataFrame with columns like 'batch_im_id','label','instance_id'
    rows = []
    bboxes_list = []
    for i in range(N):
        # list_of_bboxes[i] is e.g.:
        # [ {"label":"panda-hand","bbox":[x1,y1,x2,y2],"instance_id":0}, ... ]
        for obj_dict in list_of_bboxes[i]:
            label = obj_dict["label"]
            box_  = obj_dict["bbox"]   # [x1,y1,x2,y2]
            inst_id = obj_dict.get("instance_id", 0)
            row_dict = {
                "batch_im_id": i,
                "label": label,
                "instance_id": inst_id
            }
            rows.append(row_dict)
            bboxes_list.append(box_)
    
    if len(rows) == 0:
        # no bounding boxes at all => empty
        empty_df = pd.DataFrame([], columns=["batch_im_id","label","instance_id"])
        empty_boxes = torch.empty((0,4), dtype=torch.float32)
        detections_batched = PandasTensorCollection(empty_df, bboxes=empty_boxes)
    else:
        df = pd.DataFrame(rows)
        boxes_np = np.array(bboxes_list, dtype=np.float32)  # shape (#boxes,4)
        boxes_torch = torch.from_numpy(boxes_np)
        detections_batched = PandasTensorCollection(df, bboxes=boxes_torch)

    detections_batched = detections_batched.cuda()

    # ----------------------------------------------------------------
    # 3) Single forward pass for all frames
    # ----------------------------------------------------------------
    pose_est_batched, _ = pose_estimator.run_inference_pipeline(
        observation,
        detections=detections_batched,
        **model_info["inference_parameters"],
    )

    # ----------------------------------------------------------------
    # 4) Split results per image
    # ----------------------------------------------------------------
    results_per_image = [None]*N
    df_res = pose_est_batched.infos
    for i in range(N):
        mask = (df_res["batch_im_id"] == i)
        if not mask.any():
            # no detection for that image => empty slice
            results_per_image[i] = pose_est_batched[[]]
        else:
            idx = mask[mask].index
            subset = pose_est_batched[idx]
            results_per_image[i] = subset

    return results_per_image


COLOR_RANGES = {
    "green":   (np.array([50, 150, 50],  dtype=np.uint8), np.array([70, 255, 255], dtype=np.uint8)),
    "cyan":    (np.array([80, 150, 50],  dtype=np.uint8), np.array([100,255,255],  dtype=np.uint8)),
    "magenta": (np.array([140, 150, 50], dtype=np.uint8), np.array([170,255,255],  dtype=np.uint8)),
}

def find_color_bounding_box(rgb_image: np.ndarray, color_name: str = "green",
                          kernel_size: int = 3, erode_iters: int = 1,
                          dilate_iters: int = 1) -> tuple:
    """Finds largest bounding box for a specified color region in an RGB image."""
    # 1) Convert from RGB to HSV
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)

    # 2) Retrieve the lower & upper HSV bounds for the desired color.
    #    If color_name not in dict, raise error or handle it somehow
    if color_name not in COLOR_RANGES:
        raise ValueError(f"Unknown color '{color_name}'. Choose from {list(COLOR_RANGES.keys())}.")

    lower_hsv, upper_hsv = COLOR_RANGES[color_name]

    # 3) Create a mask for pixels within this HSV range
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # 4) Morphological operations to remove small noise
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=erode_iters)
    mask = cv2.dilate(mask, kernel, iterations=dilate_iters)

    # 5) Find connected components (largest color blob)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    # stats shape: (num_labels, 5) => [label, left, top, width, height, area]
    # label=0 is background

    if num_labels <= 1:
        # No colored component found
        return None

    # 6) Identify the largest non-background component by area
    #    np.argmax(stats[1:, cv2.CC_STAT_AREA]) gives the largest blob among labels [1..]
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

    # 7) Extract bounding box of that largest component
    x = stats[largest_label, cv2.CC_STAT_LEFT]
    y = stats[largest_label, cv2.CC_STAT_TOP]
    w = stats[largest_label, cv2.CC_STAT_WIDTH]
    h = stats[largest_label, cv2.CC_STAT_HEIGHT]

    # Return in (x_min, y_min, x_max, y_max)
    return (x, y, x + w, y + h)


def bounding_box_center(bbox):
    """Calculates center point of a bounding box."""
    x_min, y_min, x_max, y_max = bbox
    return (x_min + x_max) / 2.0, (y_min + y_max) / 2.0

def bounding_box_distance(bbox1, bbox2):
    """Computes Euclidean distance between centers of two bounding boxes."""
    cx1, cy1 = bounding_box_center(bbox1)
    cx2, cy2 = bounding_box_center(bbox2)
    return math.hypot(cx2 - cx1, cy2 - cy1)

def pixel_to_world(u: float,
                   v: float,
                   depth: float,
                   K: np.ndarray,
                   R: np.ndarray) -> np.ndarray:
    """
    Converts a pixel coordinate (u, v) at a given 'depth' from camera coordinates
    into a 3D point in the 'world' coordinate system.

    Args:
        u, v          : The 2D pixel coordinates in the image (origin at top-left).
        depth         : Depth value (distance along the camera's optical axis).
        K             : 3×3 camera intrinsics matrix.
                        Typically:  [[fx,  0, cx],
                                     [ 0, fy, cy],
                                     [ 0,  0,  1]]
        R             : 4×4 matrix that transforms a point from
                        camera coordinates to world coordinates.
                        i.e. X_world = R @ X_cam_homogeneous

    Returns:
        A 3D point (numpy array of shape (3,)) in world coordinates.
    """
    # 1) Unproject pixel (u, v, depth) to camera coordinates (x_cam, y_cam, z_cam).
    #    Using the pinhole camera model:  [u]   [fx  0  cx] [x_cam]
    #                                     [v] = [ 0  fy  cy] [y_cam]
    #                                     [1]   [ 0   0   1] [z_cam]
    #
    #  => [x_cam]   = inv(K) * [u, v, 1]^T * depth
    #     [y_cam]
    #     [z_cam]

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    # Convert (u, v, depth) to 3D in camera space
    x_cam = (u - cx) / fx * depth
    y_cam = (v - cy) / fy * depth
    z_cam = depth

    # 2) Form a homogeneous 4-vector [x_cam, y_cam, z_cam, 1]
    point_cam_h = np.array([x_cam, y_cam, z_cam, 1.0], dtype=np.float64)

    # 3) Transform from camera coords to world coords
    point_world_h = R @ point_cam_h  # shape (4,)

    # 4) Return the 3D world coordinate (x_w, y_w, z_w)
    x_w, y_w, z_w, _ = point_world_h
    return np.array([x_w, y_w, z_w], dtype=np.float64)

def finger_distance_in_world(
    bbox_cyan,
    bbox_magenta,
    depth: float,
    K: np.ndarray,
    R: np.ndarray
) -> float:
    """
    Compute the 3D distance between two bounding box centers in world space.
    Requires a depth_map for unprojection.
    """
    if bbox_cyan is None or bbox_magenta is None:
        return 0.0

    # Get bounding box centers (u, v)
    uc, vc = bounding_box_center(bbox_cyan)  # e.g. int pixel coords
    um, vm = bounding_box_center(bbox_magenta)

    # Unproject each center to world coords
    p_cyan_world = pixel_to_world(uc, vc, depth, K, R)
    p_magenta_world = pixel_to_world(um, vm, depth, K, R)

    # Euclidean distance in world
    return float(np.linalg.norm(p_cyan_world - p_magenta_world))


def add_angle_to_axis_angle(rvec, deg=15.0):
    """
    rvec: 3D axis-angle vector from e.g. as_rotvec() [rx, ry, rz]
    deg: amount in degrees you want to add to the original angle
    returns: new 3D axis-angle vector with the same axis but angle + deg
    """
    angle = np.linalg.norm(rvec)
    if angle < 1e-12:
        # Edge case: zero rotation => choose any axis
        axis = np.array([1.0, 0.0, 0.0])
    else:
        axis = rvec / angle

    angle_new = angle + np.deg2rad(deg)
    rvec_new = axis * angle_new
    return rvec_new

def add_angle_to_dominant_axis(rvec: np.ndarray, deg=15.0) -> np.ndarray:
    """
    Takes an axis-angle vector `rvec = (rx, ry, rz)` (in radians).
    1) Determine the normalized axis direction: axis = rvec / ||rvec||.
    2) Find which coordinate of that axis has the largest absolute value => "dominant axis"
    3) Snap that axis to ±ex, ±ey, or ±ez.
    4) Add +deg degrees to the rotation about that axis.

    Returns the new axis-angle vector (rx', ry', rz').
    """
    angle = np.linalg.norm(rvec)
    if angle < 1e-12:
        # No meaningful rotation => optionally pick x-axis by default
        axis_cardinal = np.array([1.0, 0.0, 0.0])
        angle_new = np.deg2rad(deg)
        return axis_cardinal * angle_new

    # Normalize
    axis = rvec / angle
    # Find which component is largest in absolute value
    idx = np.argmax(np.abs(axis))    # 0 => x, 1 => y, 2 => z
    sign_ = np.sign(axis[idx])       # +1 or -1
    # Construct the "dominant cardinal axis"
    axis_cardinal = np.zeros(3)
    axis_cardinal[idx] = sign_

    # Add deg degrees
    angle_new = angle + np.deg2rad(deg)
    # Rebuild rvec
    rvec_new = axis_cardinal * angle_new
    return rvec_new

# ------------------------
# APPROACHES FOR SCALING AXIS-ANGLE
# ------------------------
MAX_ANGLE_PER_STEP = np.deg2rad(30)

def rotate_in_small_increments(rvec, scale):
    """
    Approach A: multiply the axis-angle 'rvec' by 'scale'
    in multiple smaller steps so we never exceed ±pi in one shot.
    """
    from scipy.spatial.transform import Rotation as R
    angle = np.linalg.norm(rvec)
    if angle < 1e-8:
        return rvec * scale

    axis = rvec / angle
    angle_target = angle * scale
    n_increments = int(np.ceil(abs(angle_target) / MAX_ANGLE_PER_STEP))
    angle_incr = angle_target / n_increments

    rot_accum = R.from_quat([0,0,0,1])  # identity
    rot_step = R.from_rotvec(axis * angle_incr)
    for _ in range(n_increments):
        rot_accum = rot_step * rot_accum
    return rot_accum.as_rotvec()

def rotate_using_quaternion_exp(rvec, scale):
    """
    Approach B: single-step log/exp. 
    If scale*angle > pi, might still jump, but ensures we re-wrap final angle.
    """
    from scipy.spatial.transform import Rotation as R
    angle = np.linalg.norm(rvec)
    if angle < 1e-8:
        return rvec * scale
    axis = rvec / angle
    angle_new = angle * scale
    r_new = axis * angle_new
    # re-wrap
    R1 = R.from_rotvec(r_new).as_matrix()
    return R.from_matrix(R1).as_rotvec()

def smooth_positions(
    poses: list[np.ndarray],
    window_size: int = 2,
    dist_threshold: float = 0.15
) -> np.ndarray:
    """
    Smooths a sequence of SE(3) 4x4 poses by removing sudden 'outlier' 3D positions.
    Returns an (N,3) NumPy array of cleaned translations.

    Steps:
      1) Extract positions from each 4x4 pose => p_i in R^3.
      2) Outlier detection pass:
         - For each frame i, gather its neighbors in [i - window_size .. i + window_size],
           skipping i itself and any indices out of range.
         - Compute the median of those neighbor positions (robust local estimate).
         - If dist(p_i, median) > dist_threshold => mark p_i as outlier.
      3) Outlier replacement pass:
         - For each outlier, find the nearest inlier to the left, and the nearest inlier to the right.
         - If both exist => linearly interpolate.
         - If only one side exists => clamp to that inlier's position.
      4) Return the final array of size (N, 3).

    :param poses: List of N SE(3) 4x4 matrices. poses[i][:3,:3] is rotation, poses[i][:3,3] is translation.
    :param window_size: How many frames on each side to consider as neighbors for local median.
    :param dist_threshold: Distance above which a point is flagged as an outlier from its neighbors.
    :return: Nx3 float32 array of cleaned/smoothed positions.
    """
    N = len(poses)
    if N == 0:
        return np.zeros((0,3), dtype=np.float32)

    # 1) Extract the Nx3 positions
    positions = np.array([pose[:3, 3] for pose in poses], dtype=np.float32)

    # 2) First pass: detect outliers
    outlier_mask = np.zeros(N, dtype=bool)
    for i in range(N):
        # Gather neighbors in range [i-window_size.. i+window_size], excluding i
        left = max(0, i - window_size)
        right = min(N, i + window_size + 1)
        neighbor_indices = [idx for idx in range(left, right) if idx != i]

        if len(neighbor_indices) == 0:
            # No neighbors => can't decide => skip outlier check
            continue

        neighbor_pts = positions[neighbor_indices]  # shape (#neighbors, 3)

        # Median in x, y, z among neighbors
        local_median = np.median(neighbor_pts, axis=0)
        dist_i = np.linalg.norm(positions[i] - local_median)

        if dist_i > dist_threshold:
            outlier_mask[i] = True

    # 3) Second pass: replace outliers by interpolation
    cleaned_positions = positions.copy()
    for i in range(N):
        if not outlier_mask[i]:
            continue

        # Find nearest inlier to the left
        left_idx = i - 1
        while left_idx >= 0 and outlier_mask[left_idx]:
            left_idx -= 1

        # Find nearest inlier to the right
        right_idx = i + 1
        while right_idx < N and outlier_mask[right_idx]:
            right_idx += 1

        if left_idx < 0 and right_idx >= N:
            # Everything is outlier => fallback: keep original or zero
            # We'll just keep original, but you could do something else
            pass
        elif left_idx < 0:
            # No inlier to the left => clamp to right
            cleaned_positions[i] = cleaned_positions[right_idx]
        elif right_idx >= N:
            # No inlier to the right => clamp to left
            cleaned_positions[i] = cleaned_positions[left_idx]
        else:
            # Linear interpolate
            frac = (i - left_idx) / float(right_idx - left_idx)
            p_left = cleaned_positions[left_idx]
            p_right = cleaned_positions[right_idx]
            cleaned_positions[i] = p_left + frac * (p_right - p_left)

    return cleaned_positions

def poses_to_absolute_actions(
    poses, 
    gripper_actions,
    env,
    smooth=True
):
    """
    We smooth out the translation portion by looking ahead for a future
    index j where the jump is acceptable, and linearly interpolate intermediate
    positions. If no such j is found (end of trajectory), we do a linear 
    extrapolation from the past velocity or hold the last stable position.

    The orientation logic remains the same as before:
    - orientation is computed incrementally from poses[i]->poses[i+1].
    - we do not alter orientation if there's a large jump in position.

    Steps in detail:
    1) SMOOTH POSITIONS:
        - We'll create a 'smoothed_positions' array of shape (len(poses), 3).
        - Starting from i=0, if jump i->i+1 is small, keep it.
        If jump is big, search forward for j with a small jump from i.
        If found, linearly interpolate from i..j.
        If not found (end), extrapolate or hold constant.
    2) ORIENTATION:
        - For each i in [0 .. num_samples-2],
        compute delta quaternion from poses[i]-> poses[i+1].
        Accumulate into `current_orientation`.
        Convert to axis-angle (with sign unification).
    3) BUILD ACTIONS:
        - For each i in [0 .. num_samples-2], the final action 
        uses `smoothed_positions[i+1]` as [px,py,pz] 
        and the just-computed orientation as [rx,ry,rz].
        - Set gripper=1.0 or your logic.

    Returns:
    all_actions of shape (num_samples-1, 7).
    """

    num_samples = len(poses)
    if num_samples < 2:
        return np.zeros((0, 7), dtype=np.float32)

    # 1) Compute a smoothed set of positions
    if smooth:
        smoothed_positions = smooth_positions(poses, dist_threshold=0.15)
    else:
        smoothed_positions = np.array([pose[:3, 3] for pose in poses], dtype=np.float32)

    # 2) Start from environment's known initial eef quaternion 
    #    (assuming it's [w, x, y, z], confirm shape/order as needed).
    # starting_orientation = env.env.env._eef_xquat.astype(np.float32)
    current_orientation = env.env.env._eef_xquat.astype(np.float32)
    current_orientation = quat_normalize(current_orientation)
    current_position = env.env.env._eef_xpos.astype(np.float32)

    # We'll have num_actions = num_samples - 1
    num_actions = num_samples - 1
    all_actions = np.zeros((num_actions + 10, 7), dtype=np.float32)

    prev_rvec = None
    z_offset = None

    for i in range(num_actions):
        # --- Orientation from poses[i] -> poses[i+1] (the "real" rotation) ---
        # --- Position from the precomputed 'smoothed_positions' ---
        if z_offset is None:
            z_offset = smoothed_positions[i][2] - current_position[2]
            
        px, py, pz = smoothed_positions[i+1]
        
        R_i  = poses[i][:3, :3]
        R_i1 = poses[i+1][:3, :3]

        q_i  = rotation_matrix_to_quaternion(R_i)
        q_i1 = rotation_matrix_to_quaternion(R_i1)

        q_i   = quat_normalize(q_i)
        q_i1  = quat_normalize(q_i1)
        q_inv = quat_inv(q_i)

        q_delta = quat_multiply(q_inv, q_i1)
        q_delta = quat_normalize(q_delta)
        
        '''
        Only z-axis rotation allowed
        '''
        # rot = R.from_quat(q_delta)     # q_delta => [x, y, z, w]
        # roll, pitch, yaw = rot.as_euler('xyz', degrees=False)

        # # Zero out roll & pitch, keep only yaw
        # roll = 0.0
        # pitch = 0.0

        # # Build new rotation solely around z (yaw)
        # rot_only_yaw = R.from_euler('xyz', [roll, pitch, yaw], degrees=False)
        # q_delta = rot_only_yaw.as_quat()  # back to quaternion [x, y, z, w]
        '''
        Only z-axis rotation allowed
        '''

        # Accumulate orientation
        current_orientation = quat_multiply(current_orientation, q_delta)
        current_orientation = quat_normalize(current_orientation)

        # Convert to axis-angle, unify sign
        rvec = quat2axisangle(current_orientation)
        if prev_rvec is not None and np.dot(rvec, prev_rvec) < 0:
            rvec = -rvec
        prev_rvec = rvec
        
        # print(f"pz_front: {pz_front}, pz_side: {pz}")
        pz -= z_offset

        # Build the 7D action
        current_position = np.array([px, py, pz], dtype=np.float32)
        all_actions[i, :3]  = [px, py, pz]
        all_actions[i, 3:6] = rvec
        all_actions[i][-1] = gripper_actions[i]

    # Add 10 buffer absolute actions that are copies of the last action
    for i in range(10):
        all_actions[num_actions+i] = all_actions[num_actions-1]
        
    return all_actions

def poses_to_absolute_actions_two_cameras(
    poses, 
    poses_side, 
    gripper_actions,
    env,
):
    """
    We smooth out the translation portion by looking ahead for a future
    index j where the jump is acceptable, and linearly interpolate intermediate
    positions. If no such j is found (end of trajectory), we do a linear 
    extrapolation from the past velocity or hold the last stable position.

    The orientation logic remains the same as before:
    - orientation is computed incrementally from poses[i]->poses[i+1].
    - we do not alter orientation if there's a large jump in position.

    Steps in detail:
    1) SMOOTH POSITIONS:
        - We'll create a 'smoothed_positions' array of shape (len(poses), 3).
        - Starting from i=0, if jump i->i+1 is small, keep it.
        If jump is big, search forward for j with a small jump from i.
        If found, linearly interpolate from i..j.
        If not found (end), extrapolate or hold constant.
    2) ORIENTATION:
        - For each i in [0 .. num_samples-2],
        compute delta quaternion from poses[i]-> poses[i+1].
        Accumulate into `current_orientation`.
        Convert to axis-angle (with sign unification).
    3) BUILD ACTIONS:
        - For each i in [0 .. num_samples-2], the final action 
        uses `smoothed_positions[i+1]` as [px,py,pz] 
        and the just-computed orientation as [rx,ry,rz].
        - Set gripper=1.0 or your logic.

    Returns:
    all_actions of shape (num_samples-1, 7).
    """

    num_samples = len(poses)
    if num_samples < 2:
        return np.zeros((0, 7), dtype=np.float32)

    # 1) Compute a smoothed set of positions
    smoothed_positions = smooth_positions(poses, dist_threshold=0.15)
    smoothed_positions_side = smooth_positions(poses_side, dist_threshold=0.15)

    # 2) Start from environment's known initial eef quaternion 
    #    (assuming it's [w, x, y, z], confirm shape/order as needed).
    # starting_orientation = env.env.env._eef_xquat.astype(np.float32)
    current_orientation = env.env.env._eef_xquat.astype(np.float32)
    current_orientation = quat_normalize(current_orientation)
    current_position = env.env.env._eef_xpos.astype(np.float32)

    # We'll have num_actions = num_samples - 1
    num_actions = num_samples - 1
    all_actions = np.zeros((num_actions + 10, 7), dtype=np.float32)

    prev_rvec = None
    z_offset = None

    for i in range(num_actions):
        # --- Orientation from poses[i] -> poses[i+1] (the "real" rotation) ---
        # --- Position from the precomputed 'smoothed_positions' ---
        if z_offset is None:
            z_offset = smoothed_positions[i][2] - current_position[2]
            
        front_pos_delta = np.linalg.norm(smoothed_positions[i+1] - smoothed_positions[i])
        side_pos_delta = np.linalg.norm(smoothed_positions_side[i+1] - smoothed_positions_side[i])
            
        if side_pos_delta < front_pos_delta:
            px, py, pz = smoothed_positions_side[i+1]
            
            R_i  = poses_side[i][:3, :3]
            R_i1 = poses_side[i+1][:3, :3]
            
        else:
            px, py, pz = smoothed_positions[i+1]
            
            R_i  = poses[i][:3, :3]
            R_i1 = poses[i+1][:3, :3]

        q_i  = rotation_matrix_to_quaternion(R_i)
        q_i1 = rotation_matrix_to_quaternion(R_i1)

        q_i   = quat_normalize(q_i)
        q_i1  = quat_normalize(q_i1)
        q_inv = quat_inv(q_i)

        q_delta = quat_multiply(q_inv, q_i1)
        q_delta = quat_normalize(q_delta)
        
        '''
        Only z-axis rotation allowed
        '''
        # rot = R.from_quat(q_delta)     # q_delta => [x, y, z, w]
        # roll, pitch, yaw = rot.as_euler('xyz', degrees=False)

        # # Zero out roll & pitch, keep only yaw
        # roll = 0.0
        # pitch = 0.0

        # # Build new rotation solely around z (yaw)
        # rot_only_yaw = R.from_euler('xyz', [roll, pitch, yaw], degrees=False)
        # q_delta = rot_only_yaw.as_quat()  # back to quaternion [x, y, z, w]
        '''
        Only z-axis rotation allowed
        '''

        # Accumulate orientation
        current_orientation = quat_multiply(current_orientation, q_delta)
        current_orientation = quat_normalize(current_orientation)

        # Convert to axis-angle, unify sign
        rvec = quat2axisangle(current_orientation)
        if prev_rvec is not None and np.dot(rvec, prev_rvec) < 0:
            rvec = -rvec
        prev_rvec = rvec
        
        # print(f"pz_front: {pz_front}, pz_side: {pz}")
        pz -= z_offset

        # Build the 7D action
        current_position = np.array([px, py, pz], dtype=np.float32)
        all_actions[i, :3]  = [px, py, pz]
        all_actions[i, 3:6] = rvec
        all_actions[i][-1] = gripper_actions[i]

    # Add 10 buffer absolute actions that are copies of the last action
    for i in range(10):
        all_actions[num_actions+i] = all_actions[num_actions-1]
        
    return all_actions

def poses_to_absolute_actions_from_closest_camera(
    posesA, 
    posesB, 
    gripper_actions,
    env,
    cameraA_R,  # 4x4 SE(3) matrix for camera A
    cameraB_R   # 4x4 SE(3) matrix for camera B
):
    """
    For each time step, we compute a smoothed set of positions from both
    pose estimates (from camera A and camera B) and then take a weighted average
    based on the distance between each pose estimation and its corresponding camera
    location (extracted from cameraA_R and cameraB_R). The closer camera's estimate
    is given more weight. Orientation deltas from the two views are combined using slerp
    with the same weight.
    
    The rest of the processing is similar to before:
      1) Smooth positions from posesA and posesB.
      2) For each time step, compute the weighted average of positions.
      3) Compute orientation delta for each view (from posesA and posesB), then slerp 
         them (using fraction equal to the weight for cameraB) and accumulate.
      4) Build the final 7D action (position, axis-angle rotation, gripper).
    
    Returns:
      all_actions of shape (num_samples-1 + 10, 7)
    """
    num_samples = len(posesA)
    if num_samples < 2:
        return np.zeros((0, 7), dtype=np.float32)

    # 1) Compute smoothed positions from both sets of poses
    smoothed_positionsA = smooth_positions(posesA, dist_threshold=0.15)
    smoothed_positionsB = smooth_positions(posesB, dist_threshold=0.15)

    # 2) Get initial absolute orientation & position from env
    current_orientation = env.env.env._eef_xquat.astype(np.float32)  # assumed [x,y,z,w]
    current_orientation = quat_normalize(current_orientation)
    current_position = env.env.env._eef_xpos.astype(np.float32)

    num_actions = num_samples - 1
    all_actions = np.zeros((num_actions + 10, 7), dtype=np.float32)
    prev_rvec = None
    z_offset = None

    # Extract camera positions from cameraA_R and cameraB_R (assumed to be 4x4)
    camA_pos = cameraA_R[:3, 3]
    camB_pos = cameraB_R[:3, 3]

    for i in range(num_actions):
        # --- Compute weighted average of positions ---
        posA = smoothed_positionsA[i+1]
        posB = smoothed_positionsB[i+1]

        # On first iteration, define a z_offset to align z with current_position
        if z_offset is None:
            z_offset = posA[2] - current_position[2]

        # Compute distances from each estimated position to its camera
        distA = np.linalg.norm(posA - camA_pos) + 1e-6
        distB = np.linalg.norm(posB - camB_pos) + 1e-6
        # Inverse-distance weighting: closer camera gets higher weight.
        weightA = 1.0 / distA
        weightB = 1.0 / distB
        total_weight = weightA + weightB
        wA = weightA / total_weight
        wB = weightB / total_weight

        # Weighted average position
        avg_pos = wA * posA + wB * posB
        # Adjust z coordinate using the offset
        avg_pos[2] -= z_offset
        px, py, pz = avg_pos

        # --- Compute orientation deltas ---
        # For posesA (front view)
        R_i_A  = posesA[i][:3, :3]
        R_i1_A = posesA[i+1][:3, :3]
        q_i_A  = rotation_matrix_to_quaternion(R_i_A)
        q_i1_A = rotation_matrix_to_quaternion(R_i1_A)
        q_delta_A = quat_multiply(quat_inv(q_i_A), q_i1_A)
        q_delta_A = quat_normalize(q_delta_A)

        # For posesB (side view)
        R_i_B  = posesB[i][:3, :3]
        R_i1_B = posesB[i+1][:3, :3]
        q_i_B  = rotation_matrix_to_quaternion(R_i_B)
        q_i1_B = rotation_matrix_to_quaternion(R_i1_B)
        q_delta_B = quat_multiply(quat_inv(q_i_B), q_i1_B)
        q_delta_B = quat_normalize(q_delta_B)

        # Weight the orientation delta using wB (or equivalently, fraction = wB)
        # If cameraB is much closer (wB near 1), then q_delta_B dominates; if wB near 0, q_delta_A dominates.
        q_delta_weighted = slerp_quat(q_delta_A, q_delta_B, fraction=wB)

        # Accumulate orientation
        current_orientation = quat_multiply(current_orientation, q_delta_weighted)
        current_orientation = quat_normalize(current_orientation)

        # Convert current_orientation to axis-angle
        rvec = quat2axisangle(current_orientation)
        if prev_rvec is not None and np.dot(rvec, prev_rvec) < 0:
            rvec = -rvec
        prev_rvec = rvec

        # Build the action: [px, py, pz, rvec, gripper]
        current_position = np.array([px, py, pz], dtype=np.float32)
        all_actions[i, :3] = [px, py, pz]
        all_actions[i, 3:6] = rvec
        all_actions[i, 6] = gripper_actions[i]

    # Add 10 buffer actions (copies of the last action)
    for j in range(10):
        all_actions[num_actions + j] = all_actions[num_actions - 1]

    return all_actions

def slerp_quat(qA, qB, fraction=0.5):
    """
    Spherical interpolation (slerp) between two quaternions qA and qB (each [x,y,z,w]).
    fraction=0.0 => returns qA
    fraction=1.0 => returns qB
    """
    # Normalize for safety
    qA = quat_normalize(qA)
    qB = quat_normalize(qB)

    # Convert to Rotation objects
    rA = R.from_quat(qA)  # shape (4,)
    rB = R.from_quat(qB)

    # Build a small 'keyframe' rotation array (times 0, 1)
    # so we can slerp at t=fraction
    times = np.array([0.0, 1.0])
    rots = R.concatenate([rA, rB])  # shape=2

    # Create Slerp object
    slerp_obj = Slerp(times, rots)

    # Evaluate at t=fraction (must be array-like)
    r_interp = slerp_obj(np.array([fraction]))  # shape=(1,) of Rotation

    # Return the single quaternion as [x,y,z,w]
    q_interp = r_interp.as_quat()[0].astype(np.float32)

    return q_interp

def poses_to_absolute_actions_average(
    poses, 
    poses_side, 
    gripper_actions,
    env,
):
    """
    We now AVERAGE the absolute position from 'poses' and 'poses_side', 
    as well as slerp (average) the orientation change from each view.
    """

    num_samples = len(poses)
    if num_samples < 2:
        return np.zeros((0, 7), dtype=np.float32)

    # 1) Smooth or extract positions from both
    smoothed_positions_front = smooth_positions(poses, dist_threshold=0.15)
    smoothed_positions_side  = smooth_positions(poses_side, dist_threshold=0.15)

    # 2) Start from environment's known initial orientation
    current_orientation = env.env.env._eef_xquat.astype(np.float32)  # [x,y,z,w]
    current_orientation = quat_normalize(current_orientation)
    current_position    = env.env.env._eef_xpos.astype(np.float32)

    # We'll have num_actions = num_samples - 1
    num_actions = num_samples - 1
    all_actions = np.zeros((num_actions + 10, 7), dtype=np.float32)

    prev_rvec = None

    # We'll store an initial offset in z so that we align the first pose with the current eef z
    # (Same logic as your code, if you want that.)
    z_offset = None

    for i in range(num_actions):
        # ==========================
        #  (A) AVERAGE POSITIONS
        # ==========================
        front_xyz = smoothed_positions_front[i+1]
        side_xyz  = smoothed_positions_side[i+1]
        avg_xyz   = 0.5 * (front_xyz + side_xyz)

        if z_offset is None:
            # first iteration, define offset
            z_offset = avg_xyz[2] - current_position[2]

        # Subtract offset from the final Z
        px, py, pz = avg_xyz
        pz = pz - z_offset

        # ==========================
        #  (B) AVERAGE ORIENTATION (delta)
        # ==========================
        # 1) FRONT orientation delta from i -> i+1
        R_i_front  = poses[i][:3, :3]
        R_i1_front = poses[i+1][:3, :3]
        q_i_front   = rotation_matrix_to_quaternion(R_i_front)
        q_i1_front  = rotation_matrix_to_quaternion(R_i1_front)
        q_delta_front = quat_multiply(quat_inv(q_i_front), q_i1_front)
        q_delta_front = quat_normalize(q_delta_front)

        # 2) SIDE orientation delta from i -> i+1
        R_i_side  = poses_side[i][:3, :3]
        R_i1_side = poses_side[i+1][:3, :3]
        q_i_side   = rotation_matrix_to_quaternion(R_i_side)
        q_i1_side  = rotation_matrix_to_quaternion(R_i1_side)
        q_delta_side = quat_multiply(quat_inv(q_i_side), q_i1_side)
        q_delta_side = quat_normalize(q_delta_side)

        # 3) "Average" (slerp) these two delta quaternions 50/50
        q_delta_averaged = slerp_quat(q_delta_front, q_delta_side, fraction=0.5)

        # 4) Accumulate into current_orientation
        current_orientation = quat_multiply(current_orientation, q_delta_averaged)
        current_orientation = quat_normalize(current_orientation)

        # Convert to axis-angle, unify sign with previous step if needed
        rvec = quat2axisangle(current_orientation)
        if (prev_rvec is not None) and (np.dot(rvec, prev_rvec) < 0.0):
            rvec = -rvec
        prev_rvec = rvec

        # ==========================
        #  (C) BUILD THE ACTION
        # ==========================
        all_actions[i, :3]  = [px, py, pz]
        all_actions[i, 3:6] = rvec
        # If you want the last used gripper action from index i, or i+1, etc.
        all_actions[i, 6]   = gripper_actions[i]

    # Add 10 buffer absolute actions that are copies of the last action
    for j in range(10):
        all_actions[num_actions + j] = all_actions[num_actions - 1]

    return all_actions

def poses_to_absolute_actions_from_multiple_cameras(
    poses_dict: dict[str, list[Optional[np.ndarray]]],
    gripper_actions: list[float],
    env,
) -> np.ndarray:
    """
    Given a dictionary of pose estimates (each a list of 4×4 transforms in world coordinates)
    from an arbitrary number of cameras, this function:
    
      1) Computes smoothed positions for each camera (using smooth_positions),
      2) For each time step, computes inverse-distance weights based on the distance between
         the estimated position and the camera's location (from self.camera_Rs),
      3) Computes a weighted average of positions from all cameras,
      4) Computes for each camera the orientation delta (from frame i to i+1), and combines
         these using a weighted slerp (iteratively) with the same weights,
      5) Accumulates the orientation deltas into a running orientation (initialized from env),
      6) Converts the current orientation into an axis–angle vector (with sign unification),
      7) Assembles the final 7D action as [px, py, pz, rx, ry, rz, gripper].
      
    A buffer of 10 additional copies of the last action is appended.
    
    :param poses_dict: Dictionary mapping camera base name (e.g. "frontview") to a list 
                       of 4×4 pose matrices (or None) for each frame.
    :param gripper_actions: List of gripper commands (one per frame).
    :param env: Environment object from which the initial eef orientation and position are obtained.
    :return: A numpy array of shape ((num_samples-1)+10, 7) of absolute actions.
    """
    # Assume all cameras have the same number of frames.
    any_key = next(iter(poses_dict))
    num_samples = len(poses_dict[any_key])
    if num_samples < 2:
        return np.zeros((0, 7), dtype=np.float32)
    
    # Compute smoothed positions for each camera.
    # Assume smooth_positions takes a list of 4x4 matrices and returns an (N,3) array.
    smoothed_positions_dict = {}
    for cam, pose_list in poses_dict.items():
        smoothed_positions_dict[cam] = smooth_positions(pose_list, dist_threshold=0.15)
    
    # Initialize the current orientation and position from the environment.
    current_orientation = env.env.env._eef_xquat.astype(np.float32)  # [x,y,z,w]
    current_orientation = quat_normalize(current_orientation)
    current_position = env.env.env._eef_xpos.astype(np.float32)

    num_actions = num_samples - 1
    all_actions = np.zeros((num_actions + 10, 7), dtype=np.float32)
    prev_rvec = None
    z_offset = None

    epsilon = 1e-6  # small constant to avoid division by zero
    
    for i in range(num_actions):
        # --- (A) Compute weighted average position ---
        pos_sum = np.zeros(3, dtype=np.float32)
        weight_sum = 0.0
        for cam, pos_array in smoothed_positions_dict.items():
            pos = pos_array[i+1]
            # Use the translation part of the camera extrinsic matrix as the camera's location.
            cam_pos = get_camera_extrinsic_matrix(env.env.env.sim, camera_name=cam)[:3, 3]
            dist = np.linalg.norm(pos - cam_pos) + epsilon
            weight = 1.0 / dist
            pos_sum += weight * pos
            weight_sum += weight
        avg_pos = pos_sum / weight_sum

        if z_offset is None:
            z_offset = avg_pos[2] - current_position[2]
        # Adjust the z coordinate
        avg_pos[2] -= z_offset
        px, py, pz = avg_pos

        # --- (B) Combine orientation deltas ---
        # For each camera, if poses are available at i and i+1, compute delta quaternion.
        # Then weight these deltas using the same weights used for position.
        combined_delta = None
        weight_total = 0.0
        for cam, pose_list in poses_dict.items():
            if (pose_list[i] is None) or (pose_list[i+1] is None):
                continue
            # Extract rotation parts (first 3x3 block)
            R_i = pose_list[i][:3, :3]
            R_i1 = pose_list[i+1][:3, :3]
            q_i = rotation_matrix_to_quaternion(R_i)
            q_i1 = rotation_matrix_to_quaternion(R_i1)
            q_i = quat_normalize(q_i)
            q_i1 = quat_normalize(q_i1)
            q_delta = quat_multiply(quat_inv(q_i), q_i1)
            q_delta = quat_normalize(q_delta)
            # Compute weight based on distance as before.
            pos_array = smoothed_positions_dict[cam]
            pos_cam = pos_array[i+1]
            cam_pos = get_camera_extrinsic_matrix(env.env.env.sim, camera_name=cam)[:3, 3]
            dist = np.linalg.norm(pos_cam - cam_pos) + epsilon
            weight = 1.0 / dist
            # For combining quaternions, we use slerp.
            # If this is the first valid delta, set it as the combined_delta.
            if combined_delta is None:
                combined_delta = q_delta
                weight_total = weight
            else:
                # Compute fraction for slerp relative to the cumulative weight.
                fraction = weight / (weight_total + weight)
                combined_delta = slerp_quat(combined_delta, q_delta, fraction=fraction)
                weight_total += weight

        if combined_delta is None:
            # If none of the cameras provided a delta, use identity (no change)
            combined_delta = np.array([0, 0, 0, 1], dtype=np.float32)
        
        # Accumulate the delta into current_orientation
        current_orientation = quat_multiply(current_orientation, combined_delta)
        current_orientation = quat_normalize(current_orientation)

        # Convert current_orientation to axis-angle (rvec) and unify sign.
        rvec = quat2axisangle(current_orientation)
        if (prev_rvec is not None) and (np.dot(rvec, prev_rvec) < 0):
            rvec = -rvec
        prev_rvec = rvec

        # --- (C) Build the 7D action ---
        current_position = np.array([px, py, pz], dtype=np.float32)
        all_actions[i, :3] = [px, py, pz]
        all_actions[i, 3:6] = rvec
        all_actions[i, 6] = gripper_actions[i]

    # Append 10 buffer actions (copies of the last action)
    for j in range(10):
        all_actions[num_actions+j] = all_actions[num_actions-1]
    
    return all_actions

def poses_to_absolute_actions_mixed_ori_v1(
    poses, 
    poses_side, 
    gripper_actions,
    env,
    axes_side_when_front=(False, False, False),
    axes_front_when_side=(False, False, False),
):
    """
    Modify orientation logic as follows:
      - 'position_from_side' in the code determines which base pose set 
        we use (front vs side).
      - If position_from_side=False => we take orientation from 'poses' 
        but override certain axes (roll/pitch/yaw) from 'poses_side'
        according to axes_side_when_front=(roll_bool, pitch_bool, yaw_bool).
      - If position_from_side=True => we take orientation from 'poses_side'
        but override certain axes from 'poses' according to 
        axes_front_when_side=(roll_bool, pitch_bool, yaw_bool).

    Everything else is exactly the same as before: 
      - We have smooth_positions for both sets, 
      - We accumulate orientation in current_orientation,
      - We apply a z_offset to pz, 
      - We store final actions in [px,py,pz, rx,ry,rz, gripper].
      - We add 10 buffer frames at the end.
    """
    num_samples = len(poses)
    if num_samples < 2:
        return np.zeros((0, 7), dtype=np.float32)

    # 1) Compute a smoothed set of positions
    smoothed_positions = smooth_positions(poses, dist_threshold=0.15)
    smoothed_positions_side = smooth_positions(poses_side, dist_threshold=0.15)

    # 2) Start from environment's known initial eef quaternion 
    #    (assuming it's [w, x, y, z], confirm shape/order as needed).
    starting_orientation = env.env.env._eef_xquat.astype(np.float32)
    current_orientation = env.env.env._eef_xquat.astype(np.float32)
    current_orientation = quat_normalize(current_orientation)
    current_position = env.env.env._eef_xpos.astype(np.float32)

    # We'll have num_actions = num_samples - 1
    num_actions = num_samples - 1
    all_actions = np.zeros((num_actions + 10, 7), dtype=np.float32)

    prev_rvec = None
    z_offset = None
    
    position_from_side = False

    for i in range(num_actions):
        # Decide whether we switch to side-based positions
        if z_offset is None:
            z_offset = smoothed_positions[i][2] - current_position[2]
        if np.linalg.norm(smoothed_positions[i+1] - smoothed_positions[i]) > 0.1:
            position_from_side = True

        # Pick the smoothed (px,py,pz) from front or side
        if position_from_side:
            px, py, pz = smoothed_positions_side[i+1]
        else:
            px, py, pz = smoothed_positions[i+1]

        # Subtract z_offset from pz
        pz -= z_offset

        # ------------------------------------------------------
        #  Orientation
        # ------------------------------------------------------
        # base orientation from front or side?
        if position_from_side:
            # base = side
            R_i  = poses_side[i][:3, :3]
            R_i1 = poses_side[i+1][:3, :3]
            # overrides from front
            euler_base_i  = R.from_matrix(R_i).as_euler('xyz', degrees=False)
            euler_base_i1 = R.from_matrix(R_i1).as_euler('xyz', degrees=False)

            euler_other_i  = R.from_matrix(poses[i][:3, :3]).as_euler('xyz', degrees=False)
            euler_other_i1 = R.from_matrix(poses[i+1][:3, :3]).as_euler('xyz', degrees=False)

            # combine euler_i, euler_i1
            combined_euler_i  = list(euler_base_i)
            combined_euler_i1 = list(euler_base_i1)
            for axis_idx in range(3):  # roll=0, pitch=1, yaw=2
                if axes_front_when_side[axis_idx]:
                    # override from front's euler
                    combined_euler_i[axis_idx]  = euler_other_i[axis_idx]
                    combined_euler_i1[axis_idx] = euler_other_i1[axis_idx]

        else:
            # base = front
            R_i  = poses[i][:3, :3]
            R_i1 = poses[i+1][:3, :3]
            # overrides from side
            euler_base_i  = R.from_matrix(R_i).as_euler('xyz', degrees=False)
            euler_base_i1 = R.from_matrix(R_i1).as_euler('xyz', degrees=False)

            euler_other_i  = R.from_matrix(poses_side[i][:3, :3]).as_euler('xyz', degrees=False)
            euler_other_i1 = R.from_matrix(poses_side[i+1][:3, :3]).as_euler('xyz', degrees=False)

            # combine euler_i, euler_i1
            combined_euler_i  = list(euler_base_i)
            combined_euler_i1 = list(euler_base_i1)
            for axis_idx in range(3):  # roll=0, pitch=1, yaw=2
                if axes_side_when_front[axis_idx]:
                    # override from side's euler
                    combined_euler_i[axis_idx]  = euler_other_i[axis_idx]
                    combined_euler_i1[axis_idx] = euler_other_i1[axis_idx]

        # Convert combined Euler => quaternions
        q_i  = R.from_euler('xyz', combined_euler_i,  degrees=False).as_quat()  # [x,y,z,w]
        q_i1 = R.from_euler('xyz', combined_euler_i1, degrees=False).as_quat()

        q_i  = quat_normalize(q_i)
        q_i1 = quat_normalize(q_i1)

        # Compute q_delta
        q_inv = quat_inv(q_i)
        q_delta = quat_multiply(q_inv, q_i1)
        q_delta = quat_normalize(q_delta)

        # Accumulate orientation
        current_orientation = quat_multiply(current_orientation, q_delta)
        current_orientation = quat_normalize(current_orientation)

        # Convert to axis-angle, unify sign
        rvec = quat2axisangle(current_orientation)
        if prev_rvec is not None and np.dot(rvec, prev_rvec) < 0:
            rvec = -rvec
        prev_rvec = rvec

        # Build the 7D action
        all_actions[i, :3]  = [px, py, pz]
        all_actions[i, 3:6] = rvec
        all_actions[i][6]   = gripper_actions[i]

    # Add 10 buffer absolute actions that are copies of the last action
    for j in range(10):
        all_actions[num_actions + j] = all_actions[num_actions - 1]

    return all_actions



class ActionIdentifierMegapose:
    """
    Processes video frames (optionally with depth) to extract hand poses from multiple cameras.
    For each camera in the provided dictionaries, this class returns 4×4 transforms in world coordinates.
    """
    def __init__(
        self,
        pose_estimator,
        camera_Rs: dict,   # dictionary mapping camera name -> 4×4 extrinsic matrix
        camera_Ks: dict,   # dictionary mapping camera name -> 3×3 intrinsic matrix
        model_info: dict,  # from NAMED_MODELS[model_name]
        batch_size: int = 40,
        scale_translation: float = 80.0,
    ):
        """
        :param pose_estimator: The PoseEstimator for 'panda-hand' (already loaded).
        :param camera_Rs: Dictionary mapping camera name to its 4×4 extrinsic matrix.
        :param camera_Ks: Dictionary mapping camera name to its 3×3 intrinsic matrix.
        :param model_info: Dict with "inference_parameters" etc.
        :param batch_size: How many frames to process at once in estimate_pose_batched.
        :param scale_translation: Multiplier for global translation deltas.
        """
        self.pose_estimator = pose_estimator
        self.model_info = model_info
        self.batch_size = batch_size
        self.scale_translation = scale_translation

        self.camera_Rs = camera_Rs
        self.camera_Ks = camera_Ks

    def get_poses_from_frames(
        self,
        cameras_frames_dict: dict[str, list[np.ndarray]],
        cameras_depth_dict: Optional[dict[str, list[np.ndarray]]] = None,
    ) -> dict[str, list[Optional[np.ndarray]]]:
        """
        Given a dictionary of RGB frames for multiple cameras (optionally with matching depth),
        this function processes the frames in batches and uses the pose estimator to extract hand poses.
        For each camera, the function:
        
        1) Chunks the frames (and optional depth) in 'batch_size' steps.
        2) Finds bounding boxes for the "green" hand in each frame.
        3) Calls estimate_pose_batched to obtain the camera-frame poses.
        4) Converts these poses to world coordinates using the corresponding extrinsic matrix 
            from self.camera_Rs.
        
        :param cameras_frames_dict: Dictionary mapping camera base name (e.g. "frontview") 
                                    to a list of RGB frames, each of shape (H,W,3).
        :param cameras_depth_dict:  Optional dictionary mapping camera base name to a list 
                                    of depth frames (each (H,W)). If a camera key is missing 
                                    or its value is None, depth is not used for that camera.
        :param depth_minmax:        Optional dictionary mapping strings like "frontview_depth" 
                                    to a tuple (min_depth, max_depth) used to normalize depth.
        :return: Dictionary mapping each camera base name to a list (length = num_frames) 
                of estimated 4×4 transforms in world coordinates (or None if no pose was found).
        """
        all_hand_poses = {}  # key: camera base name, value: list of pose matrices (or None)
        
        # Process each camera independently
        for cam, frames_list in cameras_frames_dict.items():
            num_frames = len(frames_list)
            pose_list = [None] * num_frames

            # Get the corresponding depth list for this camera (if provided)
            depth_list = None
            if cameras_depth_dict is not None:
                depth_list = cameras_depth_dict.get(cam, None)

            # Process frames in batches
            for chunk_start in range(0, num_frames, self.batch_size):
                chunk_end = min(chunk_start + self.batch_size, num_frames)
                images_chunk = []
                bboxes_chunk = []
                depth_chunk = None

                if depth_list is not None:
                    depth_chunk = depth_list[chunk_start:chunk_end]

                # For each frame in this batch, compute the bounding box for the "green" hand.
                for idx in range(chunk_start, chunk_end):
                    img = frames_list[idx]
                    box_hand = find_color_bounding_box(img, color_name="green")
                    if box_hand is not None:
                        # Use instance_id 0 for a single object
                        bboxes_chunk.append([{"label": "panda-hand", "bbox": box_hand, "instance_id": 0}])
                    else:
                        bboxes_chunk.append([])
                    images_chunk.append(img)

                # Determine depth_minmax for this camera if available.
                depth_minmax = DEPTH_MINMAX[f"{cam}_depth"]

                # Batched pose estimation for this camera's chunk.
                chunk_results = estimate_pose_batched(
                    list_of_images=images_chunk,
                    list_of_bboxes=bboxes_chunk,
                    K=self.camera_Ks[cam],
                    pose_estimator=self.pose_estimator,
                    model_info=self.model_info,
                    depth_list=depth_chunk,
                    depth_minmax=depth_minmax
                )

                # Convert each result to a 4×4 transform in world coordinates.
                for offset, global_idx in enumerate(range(chunk_start, chunk_end)):
                    pose_est_for_frame = chunk_results[offset]
                    if len(pose_est_for_frame) < 1:
                        pose_list[global_idx] = None
                    else:
                        T_cam_obj = pose_est_for_frame.poses[0].cpu().numpy()
                        T_world_obj = self.camera_Rs[cam] @ T_cam_obj
                        pose_list[global_idx] = T_world_obj

            # Store the results for this camera.
            all_hand_poses[cam] = pose_list

        return all_hand_poses

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #  APPROACH A: small increments
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def compute_actions_simple_euler_small_increments(
        self,
        all_hand_poses_world: list[np.ndarray],
        all_fingers_distances: list[float],
        all_hand_poses_world_from_side: list[np.ndarray],
        side_frames_list: list[np.ndarray],
    ) -> list[np.ndarray]:
        """
        Identical logic to 'compute_actions_simple_euler', but we call
        'axis_angle_vec = rotate_in_small_increments(axis_angle_vec, 9.5)'
        instead of 'axis_angle_vec *= 9.5'.
        """
        num_frames = len(all_hand_poses_world)
        if num_frames < 2:
            return []

        # 1) position_x array
        position_x = []
        for i in range(num_frames):
            side_frame = side_frames_list[i]
            bbox = find_color_bounding_box(side_frame, color_name="green")
            u, v = bounding_box_center(bbox)
            pose_sideview_frame = np.linalg.inv(self.sideview_R) @ all_hand_poses_world[i]
            point_depth = pose_sideview_frame[2, 3]
            point_world = pixel_to_world(u, v, point_depth, self.sideview_K, self.sideview_R)
            position_x.append(point_world[0])

        actions = []
        for i in range(num_frames - 1):
            pose_i = all_hand_poses_world[i]
            pose_i1 = all_hand_poses_world[i+1]
            pose_i_side = all_hand_poses_world_from_side[i]
            pose_i1_side = all_hand_poses_world_from_side[i+1]
            if (pose_i is None) or (pose_i1 is None) or (pose_i_side is None) or (pose_i1_side is None):
                actions.append(np.zeros(7,dtype=np.float32))
                continue

            pos_i = pose_i[:3,3]
            pos_i1 = pose_i1[:3,3]
            dp = (pos_i1 - pos_i)*self.scale_translation

            # front => R_front_delta
            delta_pose_front = np.linalg.inv(pose_i1) @ pose_i
            R_front_delta = delta_pose_front[:3,:3]

            # side => R_side_delta
            delta_pose_side = np.linalg.inv(pose_i1_side) @ pose_i_side
            R_side_delta = delta_pose_side[:3,:3]

            euler_front = R.from_matrix(R_front_delta).as_euler('xyz', degrees=False)
            euler_side  = R.from_matrix(R_side_delta).as_euler('xyz', degrees=False)

            fused_euler = [
                euler_front[0],
                euler_front[1],
                euler_side[2],
            ]
            R_fused = R.from_euler('xyz', fused_euler, degrees=False)
            axis_angle_vec = R_fused.as_rotvec()

            # approach A
            axis_angle_vec = rotate_in_small_increments(axis_angle_vec, 9.5)

            finger_distance1 = all_fingers_distances[i]
            finger_distance2 = all_fingers_distances[i+1]
            delta_finger_distance = finger_distance2 - finger_distance1

            action = np.zeros(7,dtype=np.float32)
            action[:3] = dp
            action[3:-1] = axis_angle_vec

            # Overwrite x with sideview-based position
            pos_x = position_x[i]
            pos_x_next = position_x[i+1]
            action_x = (pos_x_next - pos_x)*self.scale_translation
            action[0] = action_x

            if i>20 and delta_finger_distance<0.02:
                action[-1] = 1
            else:
                action[-1] = -np.sign(delta_finger_distance)

            actions.append(action)
        return actions

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #  APPROACH B: quaternion log/exp
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def compute_actions_simple_euler_quat_exp(
        self,
        all_hand_poses_world: list[np.ndarray],
        all_fingers_distances: list[float],
        all_hand_poses_world_from_side: list[np.ndarray],
        side_frames_list: list[np.ndarray],
    ) -> list[np.ndarray]:
        """
        Same logic as simple_euler, but we do:
        axis_angle_vec = rotate_using_quaternion_exp(axis_angle_vec, 9.5)
        """
        num_frames = len(all_hand_poses_world)
        if num_frames < 2:
            return []

        position_x = []
        for i in range(num_frames):
            side_frame = side_frames_list[i]
            bbox = find_color_bounding_box(side_frame, "green")
            u, v = bounding_box_center(bbox)
            pose_sideview_frame = np.linalg.inv(self.sideview_R) @ all_hand_poses_world[i]
            point_depth = pose_sideview_frame[2, 3]
            point_world = pixel_to_world(u, v, point_depth, self.sideview_K, self.sideview_R)
            position_x.append(point_world[0])

        actions = []
        for i in range(num_frames-1):
            pose_i = all_hand_poses_world[i]
            pose_i1 = all_hand_poses_world[i+1]
            pose_i_side = all_hand_poses_world_from_side[i]
            pose_i1_side = all_hand_poses_world_from_side[i+1]
            if (pose_i is None) or (pose_i1 is None) or (pose_i_side is None) or (pose_i1_side is None):
                actions.append(np.zeros(7,dtype=np.float32))
                continue

            pos_i = pose_i[:3,3]
            pos_i1 = pose_i1[:3,3]
            dp = (pos_i1 - pos_i)*self.scale_translation

            delta_pose_front = np.linalg.inv(pose_i1) @ pose_i
            R_front_delta = delta_pose_front[:3,:3]

            delta_pose_side = np.linalg.inv(pose_i1_side) @ pose_i_side
            R_side_delta = delta_pose_side[:3,:3]

            euler_front = R.from_matrix(R_front_delta).as_euler('xyz', degrees=False)
            euler_side  = R.from_matrix(R_side_delta).as_euler('xyz', degrees=False)

            fused_euler = [
                euler_front[0],
                euler_front[1],
                euler_side[2],
            ]
            R_fused = R.from_euler('xyz', fused_euler, degrees=False)
            axis_angle_vec = R_fused.as_rotvec()

            # approach B
            axis_angle_vec = rotate_using_quaternion_exp(axis_angle_vec, 9.5)

            finger_distance1 = all_fingers_distances[i]
            finger_distance2 = all_fingers_distances[i+1]
            delta_finger_distance = finger_distance2 - finger_distance1

            action = np.zeros(7,dtype=np.float32)
            action[:3] = dp
            action[3:-1] = axis_angle_vec

            pos_x = position_x[i]
            pos_x_next = position_x[i+1]
            action_x = (pos_x_next - pos_x)*self.scale_translation
            action[0] = action_x

            if i>20 and delta_finger_distance<0.02:
                action[-1] = 1
            else:
                action[-1] = -np.sign(delta_finger_distance)

            actions.append(action)
        return actions

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #  APPROACH C: multiply and unwrap
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def compute_actions_simple_euler_unwrap(
        self,
        all_hand_poses_world: list[np.ndarray],
        all_fingers_distances: list[float],
        all_hand_poses_world_from_side: list[np.ndarray],
        side_frames_list: list[np.ndarray],
    ) -> list[np.ndarray]:
        """
        Same logic as simple_euler, but do direct multiply then unwrap angle to (-pi, pi].
        """
        num_frames = len(all_hand_poses_world)
        if num_frames < 2:
            return []

        position_x = []
        for i in range(num_frames):
            side_frame = side_frames_list[i]
            bbox = find_color_bounding_box(side_frame, "green")
            u, v = bounding_box_center(bbox)
            pose_sideview_frame = np.linalg.inv(self.sideview_R) @ all_hand_poses_world[i]
            point_depth = pose_sideview_frame[2,3]
            point_world = pixel_to_world(u,v, point_depth, self.sideview_K, self.sideview_R)
            position_x.append(point_world[0])

        actions = []
        for i in range(num_frames-1):
            pose_i = all_hand_poses_world[i]
            pose_i1 = all_hand_poses_world[i+1]
            pose_i_side = all_hand_poses_world_from_side[i]
            pose_i1_side = all_hand_poses_world_from_side[i+1]
            if (pose_i is None) or (pose_i1 is None) or (pose_i_side is None) or (pose_i1_side is None):
                actions.append(np.zeros(7,dtype=np.float32))
                continue

            pos_i = pose_i[:3,3]
            pos_i1 = pose_i1[:3,3]
            dp = (pos_i1 - pos_i)*self.scale_translation

            delta_pose_front = np.linalg.inv(pose_i1) @ pose_i
            R_front_delta = delta_pose_front[:3,:3]

            delta_pose_side = np.linalg.inv(pose_i1_side) @ pose_i_side
            R_side_delta = delta_pose_side[:3,:3]

            euler_front = R.from_matrix(R_front_delta).as_euler('xyz', degrees=False)
            euler_side  = R.from_matrix(R_side_delta).as_euler('xyz', degrees=False)

            fused_euler = [ euler_front[0], euler_front[1], euler_side[2] ]
            R_fused = R.from_euler('xyz', fused_euler, degrees=False)
            axis_angle_vec = R_fused.as_rotvec()

            # approach C: multiply, then unwrap
            angle = np.linalg.norm(axis_angle_vec)
            if angle<1e-8:
                axis_angle_vec *= 9.5
            else:
                axis = axis_angle_vec/angle
                angle_new = angle*9.5
                while angle_new>np.pi:
                    angle_new -= 2*np.pi
                while angle_new<=-np.pi:
                    angle_new += 2*np.pi
                axis_angle_vec = axis*angle_new

            finger_distance1 = all_fingers_distances[i]
            finger_distance2 = all_fingers_distances[i+1]
            delta_finger_distance = finger_distance2 - finger_distance1

            action = np.zeros(7,dtype=np.float32)
            action[:3] = dp
            action[3:-1] = axis_angle_vec

            pos_x = position_x[i]
            pos_x_next = position_x[i+1]
            action_x = (pos_x_next - pos_x)*self.scale_translation
            action[0] = action_x

            if i>20 and delta_finger_distance<0.02:
                action[-1] = 1
            else:
                action[-1] = -np.sign(delta_finger_distance)

            actions.append(action)
        return actions


    def compute_actions_simple_euler(
        self,
        all_hand_poses_world: list[np.ndarray],
        all_fingers_distances: list[float],
        all_hand_poses_world_from_side: list[np.ndarray],
        side_frames_list: list[np.ndarray],
        skip_n: int = 3,
    ) -> list[np.ndarray]:
        """
        Modified compute_actions_simple_euler with skipping & interpolation.

        1) We choose 'unskipped' frames at intervals of (skip_n+1), plus
        always include the last frame.
        e.g. if skip_n=2 and we have 10 frames, we keep frames [0,3,6,9].
        Then from the last unskipped=9 to final=9 anyway => done.
        If final was 9 but we had 10 frames => we keep 0,3,6,9,9? Actually 9 is the last,
        so it's not repeated. If the last unskipped < num_frames-1, we add num_frames-1.

        2) For each consecutive pair (i, j) of unskipped frames:
        - We "slerp" (or axis-angle interpolate) from T_i to T_j in
            (j - i + 1) steps, filling the new, "interpolated" list for frames i..j.
            That means each intermediate frame is a fraction of the big pose difference.
        - Similarly for 'all_hand_poses_world_from_side'.

        3) Then we apply the original Euler-fusion logic to this new
        "interpolated" pose list for front and side, producing (num_frames-1) actions.

        So the final action list is the *same length* as original (one action
        per step), but the actual poses used for the steps are smoothed or
        subdivided between key frames.

        :param skip_n: how many frames to skip between “key frames”.
                    0 => old behavior (use every frame). 2 => use frames
                    i, i+3, i+6, etc. and interpolate the in-betweens.
        """

        import numpy as np

        num_frames = len(all_hand_poses_world)
        if num_frames < 2:
            return []

        # ----------------------------------------------------------------
        # Helper: SLERP or axis-angle interpolation between two 4x4 poses
        # ----------------------------------------------------------------
        def interpolate_pose(T1, T2, frac: float):
            """
            Interpolate between T1, T2 in SE(3) using linear translation + quaternion slerp.
            T1, T2 are (4,4) or possibly None.
            frac in [0..1].
            Returns a new (4,4) transform or None if T1 or T2 is None.
            """
            if T1 is None or T2 is None:
                return None
            p1 = T1[:3, 3]
            p2 = T2[:3, 3]
            # linear interpolation for translation
            p_interp = p1 + frac * (p2 - p1)

            R1 = T1[:3, :3]
            R2 = T2[:3, :3]
            q1 = R.from_matrix(R1).as_quat()  # x,y,z,w
            q2 = R.from_matrix(R2).as_quat()
            # slerp
            # There's no direct built-in "one-step slerp" in newer scipy, so we do manual:
            #   R_slerp(frac) = R1 * (R1.inv * R2)^frac
            # Or we can do an "squad" approach. But let's do a simpler "single-step"
            # using from_quat => as_quat to get the fraction:
            # We'll do a custom function or we rely on "R.slerp".
            # But to keep no external code, let's do a single-step function:

            # normalized dot
            dot = np.dot(q1, q2)
            if dot < 0.0:
                q2 = -q2
                dot = -dot
            dot = np.clip(dot, -1.0, 1.0)

            if dot > 0.9995:
                # nearly identical => linear
                q_interp = q1 + frac*(q2 - q1)
                q_interp /= np.linalg.norm(q_interp)
            else:
                theta_0 = np.arccos(dot)
                theta = theta_0 * frac
                sin_theta = np.sin(theta)
                sin_theta_0 = np.sin(theta_0)
                s1 = np.cos(theta) - dot * sin_theta / sin_theta_0
                s2 = sin_theta / sin_theta_0
                q_interp = (s1 * q1) + (s2 * q2)

            # build final rotation
            R_interp = R.from_quat(q_interp).as_matrix()

            T_out = np.eye(4)
            T_out[:3, :3] = R_interp
            T_out[:3, 3] = p_interp
            return T_out

        # ----------------------------------------------------------------
        # 1) Identify "unskipped" key frames
        # ----------------------------------------------------------------
        unskipped = []
        i = 0
        step = skip_n + 1
        while i < num_frames:
            unskipped.append(i)
            i += step
        # if the last key frame isn't num_frames-1, force it:
        if unskipped[-1] != num_frames - 1:
            unskipped[-1] = num_frames - 1  # ensure last pose is used

        # ----------------------------------------------------------------
        # 2) Build new "interpolated" lists for front and side
        # ----------------------------------------------------------------
        new_front = [None]*num_frames
        new_side  = [None]*num_frames

        for k in range(len(unskipped)-1):
            idxA = unskipped[k]
            idxB = unskipped[k+1]
            # T_A => T_B
            T_Af = all_hand_poses_world[idxA]
            T_Bf = all_hand_poses_world[idxB]
            T_As = all_hand_poses_world_from_side[idxA]
            T_Bs = all_hand_poses_world_from_side[idxB]

            length = idxB - idxA
            # fill frames from idxA..idxB
            for local_i in range(length+1):
                frac = 0.0 if length==0 else (local_i / float(length))
                # front
                new_front[idxA + local_i] = interpolate_pose(T_Af, T_Bf, frac)
                # side
                new_side[idxA + local_i]  = interpolate_pose(T_As, T_Bs, frac)

        # If there's only 1 "unskipped" => do nothing or just copy
        if len(unskipped) == 1:
            # trivial => all are the same pose or None
            new_front[0] = all_hand_poses_world[0]
            new_side[0]  = all_hand_poses_world_from_side[0]

        # Now we have 'new_front', 'new_side' as interpolated poses
        # We'll do the EXACT same loop as your original code, but referencing these new lists.

        # 3) Precompute "position_x" from side_frames_list
        position_x = []
        for i in range(num_frames):
            side_frame = side_frames_list[i]
            bbox = find_color_bounding_box(side_frame, color_name="green")
            u, v = bounding_box_center(bbox)
            # be sure to reference new_front or new_side for the transform
            # According to your code, you used "all_hand_poses_world[i]" for the sideview_R transform
            # We'll keep the same approach, but now using new_front[i] (the "primary"???)
            # Actually you used "all_hand_poses_world[i]", but let's remain consistent:
            # if the logic is the same, we do that. Or we might want new_front. We'll do new_front.
            # Because that means we consistently handle the newly interpolated front pose for extrinsic?
            # But your original code used sideview_R * all_hand_poses_world[i] as a partial? Actually it did:
            #   pose_sideview_frame = inv(sideview_R) @ all_hand_poses_world[i]
            # We'll replicate the same. So:
            if new_front[i] is None:
                position_x.append(0.0)
                continue
            pose_sideview_frame = np.linalg.inv(self.sideview_R) @ new_front[i]
            point_depth = pose_sideview_frame[2, 3]
            point_world = pixel_to_world(u, v, point_depth, self.sideview_K, self.sideview_R)
            position_x.append(point_world[0])

        # 4) Build actions
        actions = []
        for i in range(num_frames - 1):
            pose_i_front = new_front[i]
            pose_i1_front = new_front[i + 1]
            pose_i_side = new_side[i]
            pose_i1_side = new_side[i + 1]

            if (pose_i_front is None) or (pose_i1_front is None) or \
            (pose_i_side is None) or (pose_i1_side is None):
                actions.append(np.zeros(7, dtype=np.float32))
                continue

            # translation
            pos_i = pose_i_front[:3, 3]
            pos_i1 = pose_i1_front[:3, 3]
            dp = (pos_i1 - pos_i) * self.scale_translation

            # fused rotation
            delta_pose_front = np.linalg.inv(pose_i1_front) @ pose_i_front
            R_front_delta = delta_pose_front[:3, :3]

            delta_pose_side = np.linalg.inv(pose_i1_side) @ pose_i_side
            R_side_delta = delta_pose_side[:3, :3]

            euler_front = R.from_matrix(R_front_delta).as_euler('xyz', degrees=False)
            euler_side  = R.from_matrix(R_side_delta).as_euler('xyz', degrees=False)

            # x,z from front, y from side
            fused_euler = [
                euler_front[0],
                euler_front[1],
                euler_side[2],
            ]
            R_fused = R.from_euler('xyz', fused_euler, degrees=False)
            axis_angle_vec = R_fused.as_rotvec()

            # multiply angle
            axis_angle_vec *= 9.5

            # finger distance
            finger_distance1 = all_fingers_distances[i]
            finger_distance2 = all_fingers_distances[i + 1]
            delta_finger_distance = finger_distance2 - finger_distance1

            action = np.zeros(7, dtype=np.float32)
            action[:3] = dp
            action[3:-1] = axis_angle_vec

            # Overwrite x with side-based position
            pos_x = position_x[i]
            pos_x_next = position_x[i + 1]
            action_x = (pos_x_next - pos_x) * self.scale_translation
            action[0] = action_x

            if i > 20 and delta_finger_distance < 0.02:
                action[-1] = 1
            else:
                action[-1] = -np.sign(delta_finger_distance)

            actions.append(action)

        return actions


    def compute_actions(
        self,
        all_hand_poses_world: list[np.ndarray],
        all_fingers_distances: list[float],
        side_frames_list: list[np.ndarray],
    ) -> list[np.ndarray]:
        """
        Replicates your script's logic to produce (n-1) actions in shape (7,).
        For each i from 0..(n-2):
          - If either frame i or i+1 is None => action=0
          - else => delta position * scale_translation, plus a sign-based 'gripper' coordinate in last dim
        Returns a list of length (n-1) with each action of shape (7,).
        """
        num_frames = len(all_hand_poses_world)
        if num_frames < 2:
            return []

        """The x-axis actions in global frame are computed separately"""
        # First we compute the x-axis POSITIONS
        position_x = []
        for i in range(num_frames):
            side_frame = side_frames_list[i]
            bbox = find_color_bounding_box(side_frame, color_name="green")
            u, v = bounding_box_center(bbox)
            pose_sideview_frame = np.linalg.inv(self.sideview_R) @ all_hand_poses_world[i]
            point_depth = pose_sideview_frame[2, 3]
            point_world = pixel_to_world(u, v, point_depth, self.sideview_K, self.sideview_R)
            position_x.append(point_world[0])

        actions = []
        for i in range(num_frames - 1):
            pose_i = all_hand_poses_world[i]
            pose_i1 = all_hand_poses_world[i + 1]
            if (pose_i is None) or (pose_i1 is None):
                actions.append(np.zeros(7, dtype=np.float32))
                continue

            pos_i = pose_i[:3, 3]
            pos_i1 = pose_i1[:3, 3]
            dp = (pos_i1 - pos_i) * self.scale_translation
            
            delta_pose = np.linalg.inv(pose_i1) @ pose_i

            # 2) Extract the 3×3 rotation from delta_pose
            R_delta = delta_pose[:3, :3]

            # 3) Convert rotation matrix to axis-angle
            #    as_rotvec() returns a 3D vector whose direction is the rotation axis
            #    and whose magnitude is the rotation angle (in radians).
            axis_angle_vec = R.from_matrix(R_delta).as_rotvec() 

            finger_distance1 = all_fingers_distances[i]
            finger_distance2 = all_fingers_distances[i + 1]
            
            delta_finger_distance = finger_distance2 - finger_distance1

            action = np.zeros(7, dtype=np.float32)
            action[:3] = dp
            action[3:-1] = axis_angle_vec * 9.5
            # action[3:-1] = add_angle_to_dominant_axis(axis_angle_vec, deg=15.0)
            # action[3:-1] = axis_angle_vec
            
            # Compute x-axis actions separately
            pos_x = position_x[i]
            pos_x_next = position_x[i + 1]
            action_x = (pos_x_next - pos_x) * self.scale_translation
            action[0] = action_x
            
            if i > 20 and delta_finger_distance < 0.02:
                # Gripping
                action[-1] = 1
            else:
                action[-1] = -np.sign(delta_finger_distance)
            actions.append(action)

        return actions



# -----------------------------------------------------------------------------
# Minimal "main" Demo
# -----------------------------------------------------------------------------

def main():
    """Demo showing basic usage of pose estimation pipeline."""
    """
    Example usage:
      python refactored_inference.py

    We'll demonstrate how to build `detections` from an in-memory bounding box,
    load the mesh from `action_extractor/megapose/panda_hand_mesh`,
    and run inference on a dummy image.
    """

    # 1. Let's pretend we have a 128x128 RGB image
    H, W = 128, 128
    image_rgb = np.zeros((H, W, 3), dtype=np.uint8)
    image_rgb[:] = (128, 255, 128)  # a random greenish color

    # 2. Intrinsics matrix K (example)
    K = np.array([
        [100.,    0., W/2.],
        [   0., 100., H/2.],
        [   0.,   0.,   1. ],
    ], dtype=np.float32)

    # 3. Make a bounding box detection for one object
    # If your original code expects [x1, y1, w, h], or [x1, y1, x2, y2],
    # adapt accordingly. We'll assume [x1, y1, w, h].
    from megapose.datasets.scene_dataset import ObjectData
    object_data = [
        ObjectData(
            label="panda-hand", 
            bbox_modal=[52, 52, 70, 59], 
        )
    ]

    # 4. Build a DetectionsType the same way your code does
    detections = make_detections_from_object_data(object_data)  # CPU so far

    # 5. Create a RigidObjectDataset from your mesh folder
    mesh_dir = Path("/home/yilong/Documents/action_extractor/action_extractor/megapose/panda_hand_mesh/")
    object_dataset = make_object_dataset_from_folder(mesh_dir)

    # 6. Run inference
    model_name = "megapose-1.0-RGB-multi-hypothesis"
    output_dir = None  # or None if you don't need to save
    
    model_info = NAMED_MODELS[model_name]
    logger.info(f"Loading model {model_name} ...")
    pose_estimator = load_named_model(model_name, object_dataset).cuda()

    pose_estimates = estimate_pose(
        image_rgb, K,
        detections,
        pose_estimator,
        model_info,
        depth=None,
        output_dir=output_dir,
    )

    print("Pose estimates:", pose_estimates)


if __name__ == "__main__":
    main()
