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

def estimate_pose_batched(list_of_images: list[np.ndarray], list_of_bboxes: list[list[dict]],
                         K: np.ndarray, pose_estimator: PoseEstimator, model_info: dict,
                         depth_list: list[np.ndarray] = None) -> list[PoseEstimatesType]:
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
            depth_torch = torch.from_numpy(depth_np).float() / 255.0  # (1,1,H,W)
            depth_torch = depth_torch.permute(2,0,1).unsqueeze(0) # (1,1,H,W)
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

class ActionIdentifierMegapose:
    """Processes video frames (optionally with depth) to extract hand poses and finger distances."""

    def __init__(
        self,
        pose_estimator,
        R: np.ndarray,         # extrinsics from get_camera_extrinsic_matrix
        K: np.ndarray,         # intrinsics from get_camera_intrinsic_matrix
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

        # Ensure R is at least 4×4 if user gave 3×3 or 3×4
        if R.shape == (4, 4):
            self.R = R
        else:
            # minimal fix if the user only provided a rotation or 3×4
            R_4x4 = np.eye(4, dtype=np.float32)
            R_4x4[:R.shape[0], :R.shape[1]] = R
            self.R = R_4x4

        self.K = K

    def get_all_hand_poses_finger_distances(
        self,
        frames_list: list[np.ndarray],
        depth_list: Optional[list[np.ndarray]] = None,
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

        :param frames_list: List of RGB frames, each shape (H,W,3).
        :param depth_list:  Optional list of depth frames, each shape (H,W). If None, depth is not used.
        """
        num_frames = len(frames_list)
        all_hand_poses_world = [None] * num_frames
        all_fingers_distances = [0.0] * num_frames

        # Process in chunks
        for chunk_start in range(0, num_frames, self.batch_size):
            chunk_end = min(chunk_start + self.batch_size, num_frames)
            images_chunk = []
            bboxes_chunk = []
            depth_chunk = None

            if depth_list is not None:
                # Slice the depth frames for this batch
                depth_chunk = depth_list[chunk_start:chunk_end]

            # Gather bounding boxes for 'panda-hand' (green)
            for idx in range(chunk_start, chunk_end):
                img = frames_list[idx]
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
                K=self.K,
                pose_estimator=self.pose_estimator,
                model_info=self.model_info,
                depth_list=depth_chunk,
            )

            # measure finger distances (cyan vs magenta)
            finger_dist_chunk = []
            for idx in range(chunk_start, chunk_end):
                img = frames_list[idx]
                bbox_cyan = find_color_bounding_box(img, color_name="cyan")
                bbox_magenta = find_color_bounding_box(img, color_name="magenta")
                d = 0.0
                # If bounding boxes are None => bounding_box_distance would fail
                if (bbox_cyan is not None) and (bbox_magenta is not None):
                    d = bounding_box_distance(bbox_cyan, bbox_magenta)
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
                    T_world_obj = self.R @ T_cam_obj
                    all_hand_poses_world[global_idx] = T_world_obj

        return all_hand_poses_world, all_fingers_distances

    def compute_actions(
        self,
        all_hand_poses_world: list[Optional[np.ndarray]],
        all_fingers_distances: list[float],
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

        actions = []
        for i in range(num_frames - 1):
            pose_i = all_hand_poses_world[i]
            pose_i1 = all_hand_poses_world[i + 1]
            if (pose_i is None) or (pose_i1 is None):
                actions.append(np.zeros(7, dtype=np.float32))
                continue

            pos_i = pose_i[:3, 3]
            pos_i1 = pose_i1[:3, 3]
            dp = self.scale_translation * (pos_i1 - pos_i)

            finger_distance1 = all_fingers_distances[i]
            finger_distance2 = all_fingers_distances[i + 1]
            
            delta_finger_distance = finger_distance2 - finger_distance1

            action = np.zeros(7, dtype=np.float32)
            action[:3] = dp
            print('finger_distance:', finger_distance1)
            if finger_distance1 <= 19:
                action[-1] = -1
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
