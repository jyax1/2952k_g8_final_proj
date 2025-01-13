import os
import json
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

# If these imports are from your internal code, keep them:
# from megapose.config import LOCAL_DATA_DIR
from megapose.datasets.object_dataset import RigidObjectDataset
from megapose.datasets.scene_dataset import CameraData, ObjectData
from megapose.inference.types import DetectionsType, ObservationTensor, PoseEstimatesType
from megapose.inference.utils import make_detections_from_object_data
from megapose.lib3d.transform import Transform
from megapose.panda3d_renderer import Panda3dLightData
from megapose.panda3d_renderer.panda3d_scene_renderer import Panda3dSceneRenderer
from megapose.utils.load_model import NAMED_MODELS, load_named_model
from megapose.utils.logging import get_logger
# from megapose.visualization.bokeh_plotter import BokehPlotter  # if needed
# from megapose.visualization.utils import make_contour_overlay  # if needed

logger = get_logger(__name__)

# -----------------------------------------------------------------------------
# Helper / Old Functions Remain Largely the Same
# -----------------------------------------------------------------------------

def make_object_dataset(example_dir: Path) -> RigidObjectDataset:
    """
    Creates a RigidObjectDataset by scanning `example_dir / 'meshes'`
    for .obj or .ply mesh files. If you need to further refactor so that
    meshes are provided in memory, adapt this function similarly.
    """
    from megapose.datasets.object_dataset import RigidObject  # ensure local import
    rigid_objects = []
    mesh_units = "mm"
    object_dirs = (example_dir / "meshes").iterdir()
    for object_dir in object_dirs:
        label = object_dir.name
        mesh_path = None
        for fn in object_dir.glob("*"):
            if fn.suffix in {".obj", ".ply"}:
                assert not mesh_path, f"There are multiple meshes in the {label} directory"
                mesh_path = fn
        assert mesh_path, f"Couldn't find a .obj or .ply mesh for {label}"
        rigid_objects.append(RigidObject(label=label, mesh_path=mesh_path, mesh_units=mesh_units))
    rigid_object_dataset = RigidObjectDataset(rigid_objects)
    return rigid_object_dataset


def save_predictions(
    output_dir: Path,
    pose_estimates: PoseEstimatesType,
) -> None:
    """
    Saves the pose_estimates as an object_data.json in `output_dir / outputs`.
    Similar to your original `save_predictions`, but references `output_dir` 
    instead of `example_dir`.
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
# New Refactored Inference Code
# -----------------------------------------------------------------------------

def run_inference_on_data(
    image_rgb: np.ndarray,          # shape (H, W, 3), uint8
    K: np.ndarray,                  # shape (3, 3) camera intrinsics
    detections: DetectionsType,     # bounding boxes + labels, etc.
    model_name: str,
    object_dataset: RigidObjectDataset,
    requires_depth: bool = False,
    depth: Optional[np.ndarray] = None,   # shape (H, W) or None
    output_dir: Optional[Path] = None,    # if you want to save predictions
) -> PoseEstimatesType:
    """
    Runs the pose estimation pipeline on a single image + intrinsics + detections,
    without reading from any example directory.

    :param image_rgb: (H, W, 3) input image in uint8
    :param K: (3, 3) camera intrinsics
    :param detections: the bounding box / label info, as a DetectionsType
    :param model_name: e.g. "megapose-1.0-RGB-multi-hypothesis"
    :param object_dataset: a RigidObjectDataset referencing all possible objects
    :param requires_depth: whether the chosen model needs depth
    :param depth: (H, W) float array if needed, else None
    :param output_dir: optional path to save predictions in object_data.json
    :return: PoseEstimatesType containing the poses for each detection
    """

    # 1) Construct an ObservationTensor
    observation = ObservationTensor.from_numpy(
        image_rgb,  # (H, W, 3)
        depth,      # (H, W) or None
        K           # (3, 3)
    ).cuda()

    # 2) Load the named model
    model_info = NAMED_MODELS[model_name]
    logger.info(f"Loading model {model_name} ...")
    pose_estimator = load_named_model(model_name, object_dataset).cuda()

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
    Example usage. 
    We'll demonstrate how to build `detections` from an in-memory bounding box,
    and how to call `run_inference_on_data`.
    """

    # Let's pretend we have a 128x128 RGB image
    H, W = 128, 128
    image_rgb = np.zeros((H, W, 3), dtype=np.uint8)
    image_rgb[:] = (128, 255, 128)  # just a random green-ish color

    # Intrinsics matrix K (example)
    K = np.array([
        [100.,    0., W/2.],
        [   0., 100., H/2.],
        [   0.,   0.,   1. ],
    ], dtype=np.float32)

    # Let's make a bounding box detection for one object
    # Suppose the bounding box is [x1, y1, w, h]
    # or [x1, y1, x2, y2], depending on your code's convention.
    # In your snippet, it seems to be [x1, y1, w, h].
    object_data = [
        ObjectData(
            label="panda-hand",
            bbox_modal=[52, 52, 70, 59],  # e.g., x=52, y=52, w=70, h=59
        )
    ]

    # Build a DetectionsType the same way `make_detections_from_object_data` does
    # (You mentioned that code uses PandasTensorCollection, etc.)
    detections = make_detections_from_object_data(object_data)  # normal CPU
    # Typically you'd do detections = detections.cuda() inside run_inference_on_data

    # Suppose we still have a folder with "panda-hand" mesh. 
    # We'll pass that to `make_object_dataset`.
    example_dir = Path("/some/path/to/panda-hand-example/")
    object_dataset = make_object_dataset(example_dir)

    # Now run inference
    model_name = "megapose-1.0-RGB-multi-hypothesis"
    requires_depth = False
    output_dir = Path("/tmp/pose_results")  # or None if you don't want to save

    pose_estimates = run_inference_on_data(
        image_rgb, K,
        detections,
        model_name,
        object_dataset,
        requires_depth=requires_depth,
        depth=None,
        output_dir=output_dir,
    )

    # Do something with `pose_estimates` here
    print("Pose estimates:", pose_estimates)


# For command-line usage
if __name__ == "__main__":
    main()
