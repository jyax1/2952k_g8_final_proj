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

from megapose.utils.tensor_collection import PandasTensorCollection

from megapose.datasets.object_dataset import RigidObject, RigidObjectDataset
from megapose.datasets.scene_dataset import CameraData, ObjectData
from megapose.inference.types import DetectionsType, ObservationTensor, PoseEstimatesType
from megapose.inference.utils import make_detections_from_object_data
from megapose.lib3d.transform import Transform
from megapose.utils.load_model import NAMED_MODELS, load_named_model
from megapose.utils.logging import get_logger
from megapose.inference.pose_estimator import PoseEstimator

from scipy.spatial.transform import Rotation

logger = get_logger(__name__)

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

def estimate_pose_batched(
    list_of_images: list[np.ndarray],
    list_of_bboxes: list[list[dict]],
    K: np.ndarray,
    pose_estimator,
    model_info: dict,
    depth_list: Optional[list[np.ndarray]] = None,
) -> tuple[list[PoseEstimatesType], list[np.ndarray]]:
    """
    Runs pose estimation on multiple images in a single forward pass.
    
    Returns:
      - results_per_image: A list of length N, where results_per_image[i] 
        is a PoseEstimatesType containing all poses estimated for image i.
      - scores_per_image: A list of length N, where scores_per_image[i] is 
        a NumPy array containing the corresponding 'pose_score' for each row 
        in results_per_image[i].infos (in the same order).
    """
    assert len(list_of_images) == len(list_of_bboxes), (
        "list_of_images and list_of_bboxes must have same length"
    )
    N = len(list_of_images)

    # ----------------------------------------------------------------
    # 1) Build one big ObservationTensor with shape [N, C, H, W]
    # ----------------------------------------------------------------
    images_torch = []
    for i in range(N):
        rgb = list_of_images[i]  # shape (H,W,3) in uint8
        rgb_torch = torch.from_numpy(rgb).float() / 255.0  # [H,W,3], values in [0,1]
        rgb_torch = rgb_torch.permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)

        if depth_list is not None:
            # depth is shape (H,W) in uint8, for example
            depth_np = depth_list[i]
            depth_torch = torch.from_numpy(depth_np).float() / 255.0
            depth_torch = depth_torch.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
            # Concatenate along channel dimension => shape (1,4,H,W)
            img_4ch = torch.cat([rgb_torch, depth_torch], dim=1)
            images_torch.append(img_4ch)
        else:
            images_torch.append(rgb_torch)

    # Stack images into a single batch => shape (N, C, H, W)
    images_batched = torch.cat(images_torch, dim=0)

    # Replicate K => shape (N,3,3)
    K_torch = torch.from_numpy(K).float().unsqueeze(0).expand(N, -1, -1).clone()

    # Create ObservationTensor and move to GPU
    from megapose.inference.types import ObservationTensor
    observation = ObservationTensor(images_batched, K_torch).cuda()

    # ----------------------------------------------------------------
    # 2) Build a single DetectionsType containing all bounding boxes
    # ----------------------------------------------------------------
    rows = []
    bboxes_list_np = []
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
            bboxes_list_np.append(box_)

    if len(rows) == 0:
        # no bounding boxes => empty
        empty_df = pd.DataFrame([], columns=["batch_im_id", "label", "instance_id"])
        empty_boxes = torch.empty((0,4), dtype=torch.float32)
        detections_batched = PandasTensorCollection(empty_df, bboxes=empty_boxes)
    else:
        df = pd.DataFrame(rows)
        boxes_np = np.array(bboxes_list_np, dtype=np.float32)  # (#boxes,4)
        boxes_torch = torch.from_numpy(boxes_np)
        detections_batched = PandasTensorCollection(df, bboxes=boxes_torch)

    detections_batched = detections_batched.cuda()

    # ----------------------------------------------------------------
    # 3) Single forward pass for all frames
    #    => run_inference_pipeline returns final pose estimates
    # ----------------------------------------------------------------
    pose_est_batched, _ = pose_estimator.run_inference_pipeline(
        observation,
        detections=detections_batched,
        **model_info["inference_parameters"],
    )

    # ----------------------------------------------------------------
    # 4) Split results per image, along with pose scores
    # ----------------------------------------------------------------
    results_per_image: list[PoseEstimatesType] = [None]*N
    scores_per_image: list[np.ndarray] = [None]*N

    df_res = pose_est_batched.infos
    for i in range(N):
        mask = (df_res["batch_im_id"] == i)
        if not mask.any():
            # no detections for that image => empty slice
            empty_slice = pose_est_batched[[]]
            results_per_image[i] = empty_slice
            scores_per_image[i] = np.array([], dtype=np.float32)
        else:
            # indices that correspond to this image
            idx = mask[mask].index
            subset = pose_est_batched[idx]  # PoseEstimatesType for image i
            results_per_image[i] = subset

            # If pose_score is present, extract it; otherwise create an empty array
            if "pose_score" in subset.infos.columns:
                scores_per_image[i] = subset.infos["pose_score"].to_numpy(dtype=np.float32)
            else:
                scores_per_image[i] = np.array([], dtype=np.float32)

    return results_per_image, scores_per_image


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

def rotate_with_unwrap(rvec, scale):
    """
    Approach C: direct multiply then unwrap angle to (-pi, pi].
    """
    angle = np.linalg.norm(rvec)
    if angle < 1e-8:
        return rvec * scale
    axis = rvec / angle
    angle_new = angle * scale
    while angle_new > np.pi:
        angle_new -= 2*np.pi
    while angle_new <= -np.pi:
        angle_new += 2*np.pi
    return axis * angle_new


class ActionIdentifierMegapose:
    """Processes video frames (optionally with depth) to extract hand poses and finger distances."""

    def __init__(
        self,
        pose_estimator,
        frontview_R: np.ndarray,         # extrinsics from get_camera_extrinsic_matrix
        frontview_K: np.ndarray,         # intrinsics from get_camera_intrinsic_matrix
        sideview_R: np.ndarray,          # extrinsics from get_camera_extrinsic_matrix
        sideview_K: np.ndarray,          # intrinsics from get_camera_intrinsic_matrix
        model_info: dict,      # from NAMED_MODELS[model_name]
        batch_size: int = 40,
        scale_translation: float = 80.0,
    ):
        """
        :param pose_estimator: The PoseEstimator for 'panda-hand' (already loaded).
        :param R: A 4×4 (or 3×4) matrix from get_camera_extrinsic_matrix(...).
                  Must be expanded to 4×4 if only 3×3 is available.
        :param K: (3×3) intrinsics.
        :param model_info: dict with "inference_parameters" etc.
        :param batch_size: how many frames to process at once in estimate_pose_batched.
        :param scale_translation: multiplier for global translation deltas (80).
        """
        self.pose_estimator = pose_estimator
        self.model_info = model_info
        self.batch_size = batch_size
        self.scale_translation = scale_translation
        self.frontview_K = frontview_K
        self.sideview_K = sideview_K
        self.frontview_R = frontview_R
        self.sideview_R = sideview_R
    
    def get_all_hand_poses_finger_distances(
        self,
        front_frames_list: list[np.ndarray],
        front_depth_list: Optional[list[np.ndarray]] = None,
    ) -> tuple[list[Optional[np.ndarray]], list[float]]:
        """
        Given a list of frames (each shape [H,W,3] in np.uint8) and optionally a matching
        list of depth images (each shape [H,W]), replicates your script logic to:

          1) chunk the frames (and depth) in 'batch_size' steps,
          2) find bounding boxes for 'green' (the hand),
          3) call estimate_pose_batched for them,
          4) measure finger distances (cyan vs magenta),
          5) transform camera->object to world->object using R,
          6) Return:
               all_hand_poses_world : a list of length n, each an np.ndarray(4,4) or None
               all_fingers_distances: same length n, each a float with finger distance

        :param front_frames_list: List of RGB frames, each shape (H,W,3).
        :param front_depth_list:  Optional list of depth frames, each shape (H,W). If None, depth is not used.
        """
        num_frames = len(front_frames_list)
        all_hand_poses_world = [None] * num_frames
        all_fingers_distances = [0.0] * num_frames

        # Process in chunks
        for chunk_start in range(0, num_frames, self.batch_size):
            chunk_end = min(chunk_start + self.batch_size, num_frames)
            images_chunk = []
            bboxes_chunk = []
            depth_chunk = None

            if front_depth_list is not None:
                # Slice the depth frames for this batch
                depth_chunk = front_depth_list[chunk_start:chunk_end]

            # Gather bounding boxes for 'panda-hand' (green)
            for idx in range(chunk_start, chunk_end):
                img = front_frames_list[idx]
                box_hand = find_color_bounding_box(img, color_name="green")
                if box_hand is not None:
                    # instance_id=0 for single object
                    bboxes_chunk.append([{"label": "panda-hand", "bbox": box_hand, "instance_id": 0}])
                else:
                    bboxes_chunk.append([])
                images_chunk.append(img)

            # Batched pose estimation for chunk (with optional depth)
            chunk_results = estimate_pose_batched(
                list_of_images=images_chunk,
                list_of_bboxes=bboxes_chunk,
                K=self.frontview_K,
                pose_estimator=self.pose_estimator,
                model_info=self.model_info,
                depth_list=depth_chunk,
            )

            # measure finger distances (cyan vs magenta)
            finger_dist_chunk = []                
            for idx in range(chunk_start, chunk_end):
                img = front_frames_list[idx]
                bbox_cyan = find_color_bounding_box(img, "cyan")
                bbox_magenta = find_color_bounding_box(img, "magenta")
                d = 0.0

                offset = idx - chunk_start  # position within this chunk
                pose_est_for_frame = chunk_results[offset]

                if (bbox_cyan is not None) and (bbox_magenta is not None) and (len(pose_est_for_frame) > 0):
                    # Get the transform from the *front camera* to the object
                    # (assuming your 'pose_est_for_frame.poses[0]' is T_cam_obj)
                    pose_frontview_frame = pose_est_for_frame.poses[0].cpu().numpy()

                    # The 'depth' is just the z-translation of this frontview frame
                    point_depth = pose_frontview_frame[2, 3]

                    # Then compute the finger distance in world, or do a pixel_to_world, etc.
                    d = finger_distance_in_world(
                        bbox_cyan,
                        bbox_magenta,
                        point_depth,
                        self.frontview_K,
                        self.frontview_R
                    )

                finger_dist_chunk.append(d)

            # Store results in the global arrays
            for offset, global_idx in enumerate(range(chunk_start, chunk_end)):
                pose_est_for_frame = chunk_results[offset]
                all_fingers_distances[global_idx] = finger_dist_chunk[offset]
                if len(pose_est_for_frame) < 1:
                    # no detection => None
                    all_hand_poses_world[global_idx] = None
                else:
                    # get the first detection
                    T_cam_obj = pose_est_for_frame.poses[0].cpu().numpy()
                    T_world_obj = self.frontview_R @ T_cam_obj
                    all_hand_poses_world[global_idx] = T_world_obj

        return all_hand_poses_world, all_fingers_distances    

    def get_all_hand_poses_finger_distances_with_side(
        self,
        front_frames_list: list[np.ndarray],
        front_depth_list: Optional[list[np.ndarray]] = None,
        side_frames_list: Optional[list[np.ndarray]] = None,
    ) -> tuple[list[Optional[np.ndarray]], list[float], list[Optional[np.ndarray]]]:
        """
        Given a list of *front* frames (each shape [H,W,3] in np.uint8) and optionally a matching
        list of depth images (each shape [H,W]), plus an optional list of *side* frames:

          1) Chunk the front frames (and optional front-depth) in 'batch_size' steps,
          2) Find bounding boxes for 'green' (the hand) in the front frames,
          3) Call estimate_pose_batched for them (front camera) => front poses,
          4) Measure finger distances (using the front frames, 'cyan' vs 'magenta'),
          5) Transform front camera->object to world->object using frontview_R,
          6) Do the same logic for the side frames if provided, using sideview_R and sideview_K,
          7) Return three lists (all length n):
               all_hand_poses_world        : np.ndarray(4,4) or None
               all_fingers_distances       : float
               all_hand_poses_world_from_side : np.ndarray(4,4) or None

        :param front_frames_list: List of RGB frames from the front camera, each shape (H,W,3).
        :param front_depth_list:  Optional list of depth frames, each shape (H,W). If None, depth is not used.
        :param side_frames_list:  Optional list of side-camera frames, each shape (H,W,3). If None, side poses are not computed.
        :return: (all_hand_poses_world, all_fingers_distances, all_hand_poses_world_from_side)
        """
        num_frames = len(front_frames_list)
        all_hand_poses_world = [None] * num_frames
        all_fingers_distances = [0.0] * num_frames
        
        # Prepare a list for side poses (will remain all None if side_frames_list is None)
        all_hand_poses_world_from_side = [None] * num_frames

        # -----------------
        # FRONT FRAMES LOGIC
        # -----------------
        for chunk_start in range(0, num_frames, self.batch_size):
            chunk_end = min(chunk_start + self.batch_size, num_frames)
            images_chunk = []
            bboxes_chunk = []
            depth_chunk = None

            # Slice the depth frames for this batch if provided
            if front_depth_list is not None:
                depth_chunk = front_depth_list[chunk_start:chunk_end]

            # Gather bounding boxes for 'panda-hand' (green) in front frames
            for idx in range(chunk_start, chunk_end):
                img = front_frames_list[idx]
                box_hand = find_color_bounding_box(img, color_name="green")
                if box_hand is not None:
                    # instance_id=0 for single object
                    bboxes_chunk.append([{"label": "panda-hand", "bbox": box_hand, "instance_id": 0}])
                else:
                    bboxes_chunk.append([])
                images_chunk.append(img)

            # Batched pose estimation for chunk (with optional depth)
            chunk_results = estimate_pose_batched(
                list_of_images=images_chunk,
                list_of_bboxes=bboxes_chunk,
                K=self.frontview_K,
                pose_estimator=self.pose_estimator,
                model_info=self.model_info,
                depth_list=depth_chunk,
            )

            # measure finger distances (cyan vs magenta) using front frames
            finger_dist_chunk = []
            for idx in range(chunk_start, chunk_end):
                img = front_frames_list[idx]
                bbox_cyan = find_color_bounding_box(img, "cyan")
                bbox_magenta = find_color_bounding_box(img, "magenta")
                d = 0.0

                offset = idx - chunk_start  # position within this chunk
                pose_est_for_frame = chunk_results[offset]

                if (bbox_cyan is not None) and (bbox_magenta is not None) and (len(pose_est_for_frame) > 0):
                    # Get the transform from the *front camera* to the object
                    # (assuming your 'pose_est_for_frame.poses[0]' is T_cam_obj)
                    pose_frontview_frame = pose_est_for_frame.poses[0].cpu().numpy()

                    # The 'depth' is just the z-translation of this frontview frame
                    point_depth = pose_frontview_frame[2, 3]

                    # Then compute the finger distance in world, or do a pixel_to_world, etc.
                    d = finger_distance_in_world(
                        bbox_cyan,
                        bbox_magenta,
                        point_depth,
                        self.frontview_K,
                        self.frontview_R
                    )

                finger_dist_chunk.append(d)

            # Store results for front
            for offset, global_idx in enumerate(range(chunk_start, chunk_end)):
                pose_est_for_frame = chunk_results[offset]
                all_fingers_distances[global_idx] = finger_dist_chunk[offset]
                if len(pose_est_for_frame) < 1:
                    # no detection => None
                    all_hand_poses_world[global_idx] = None
                else:
                    # get the first detection
                    T_cam_obj = pose_est_for_frame.poses[0].cpu().numpy()
                    T_world_obj = self.frontview_R @ T_cam_obj
                    all_hand_poses_world[global_idx] = T_world_obj

        # ----------------
        # SIDE FRAMES LOGIC
        # ----------------
        if side_frames_list is not None:
            # Process side frames in chunks (same approach)
            for chunk_start in range(0, num_frames, self.batch_size):
                chunk_end = min(chunk_start + self.batch_size, num_frames)
                side_images_chunk = side_frames_list[chunk_start:chunk_end]
                side_bboxes_chunk = []

                # Gather bounding boxes in side frames
                for idx in range(chunk_start, chunk_end):
                    side_img = side_frames_list[idx]
                    box_hand_side = find_color_bounding_box(side_img, color_name="green")
                    if box_hand_side is not None:
                        side_bboxes_chunk.append([{"label": "panda-hand", "bbox": box_hand_side, "instance_id": 0}])
                    else:
                        side_bboxes_chunk.append([])

                # Batched pose estimation for side chunk
                side_chunk_results = estimate_pose_batched(
                    list_of_images=side_images_chunk,
                    list_of_bboxes=side_bboxes_chunk,
                    K=self.sideview_K,
                    pose_estimator=self.pose_estimator,
                    model_info=self.model_info,
                    depth_list=None,  # we don't have side-depth in your code
                )

                # Store results for side
                for offset, global_idx in enumerate(range(chunk_start, chunk_end)):
                    pose_est_for_frame_side = side_chunk_results[offset]
                    if len(pose_est_for_frame_side) < 1:
                        all_hand_poses_world_from_side[global_idx] = None
                    else:
                        T_cam_obj_side = pose_est_for_frame_side.poses[0].cpu().numpy()
                        T_world_obj_side = self.sideview_R @ T_cam_obj_side
                        all_hand_poses_world_from_side[global_idx] = T_world_obj_side

        return all_hand_poses_world, all_fingers_distances, all_hand_poses_world_from_side

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

            euler_front = Rotation.from_matrix(R_front_delta).as_euler('xyz', degrees=False)
            euler_side  = Rotation.from_matrix(R_side_delta).as_euler('xyz', degrees=False)

            fused_euler = [
                euler_front[0],
                euler_front[1],
                euler_side[2],
            ]
            R_fused = Rotation.from_euler('xyz', fused_euler, degrees=False)
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

            euler_front = Rotation.from_matrix(R_front_delta).as_euler('xyz', degrees=False)
            euler_side  = Rotation.from_matrix(R_side_delta).as_euler('xyz', degrees=False)

            fused_euler = [
                euler_front[0],
                euler_front[1],
                euler_side[2],
            ]
            R_fused = Rotation.from_euler('xyz', fused_euler, degrees=False)
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

            euler_front = Rotation.from_matrix(R_front_delta).as_euler('xyz', degrees=False)
            euler_side  = Rotation.from_matrix(R_side_delta).as_euler('xyz', degrees=False)

            fused_euler = [ euler_front[0], euler_front[1], euler_side[2] ]
            R_fused = Rotation.from_euler('xyz', fused_euler, degrees=False)
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
        from scipy.spatial.transform import Rotation as R

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
            axis_angle_vec = Rotation.from_matrix(R_delta).as_rotvec() 

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
