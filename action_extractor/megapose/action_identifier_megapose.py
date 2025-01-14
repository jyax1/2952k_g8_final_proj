import os
import json
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

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
    """
    Creates a RigidObjectDataset by scanning `mesh_dir`
    for .obj or .ply mesh files.
    
    If there's only one mesh (panda_hand_new.ply), we give it label="panda-hand".
    If there are multiple meshes, you can generate unique labels per file name.
    """
    rigid_objects = []
    mesh_units = "mm"

    # Suppose you only have 1 or very few files in mesh_dir. 
    # We will create a label from the filename stem or a fixed label.
    for fn in mesh_dir.glob("*"):
        if fn.suffix in {".obj", ".ply"}:
            label = fn.stem  # e.g. "panda_hand_new"
            # or just hardcode label="panda-hand" if you prefer
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

def save_predictions(
    output_dir: Path,
    pose_estimates: PoseEstimatesType,
) -> None:
    """
    Saves the pose_estimates as an object_data.json in `output_dir / outputs`.
    """
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

def run_inference_on_data(
    image_rgb: np.ndarray,          # shape (H, W, 3), uint8
    K: np.ndarray,                  # shape (3, 3) camera intrinsics
    detections: DetectionsType,     # bounding boxes + labels, etc.
    pose_estimator: PoseEstimator,
    model_info: dict,
    object_dataset: RigidObjectDataset,
    requires_depth: bool = False,
    depth: Optional[np.ndarray] = None,   # shape (H, W) or None
    output_dir: Optional[Path] = None,    # if you want to save predictions
) -> PoseEstimatesType:
    """
    Runs the pose estimation pipeline on a single image + intrinsics + detections,
    without reading from any 'example_dir'.
    """
    # 1) Construct an ObservationTensor
    observation = ObservationTensor.from_numpy(
        image_rgb,  # (H, W, 3)
        depth,      # (H, W) or None
        K           # (3, 3)
    ).cuda()  # Move to GPU

    # 2) Load the named model
    # Done outside this function

    # 3) Run inference
    logger.info("Running inference pipeline...")
    output, _ = pose_estimator.run_inference_pipeline(
        observation,
        detections=detections.cuda(),
        **model_info["inference_parameters"],
    )

    # 4) Optionally save predictions
    if output_dir is not None:
        save_predictions(output_dir, output)

    return output


# -----------------------------------------------------------------------------
# Minimal "main" Demo
# -----------------------------------------------------------------------------

def main():
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
    requires_depth = False
    output_dir = None  # or None if you don't need to save
    
    model_info = NAMED_MODELS[model_name]
    logger.info(f"Loading model {model_name} ...")
    pose_estimator = load_named_model(model_name, object_dataset).cuda()

    pose_estimates = run_inference_on_data(
        image_rgb, K,
        detections,
        pose_estimator,
        model_info,
        object_dataset,
        requires_depth=requires_depth,
        depth=None,
        output_dir=output_dir,
    )

    print("Pose estimates:", pose_estimates)


if __name__ == "__main__":
    main()
