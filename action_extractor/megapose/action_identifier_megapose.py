import logging
import time
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any

from megapose.config import LOCAL_DATA_DIR
from megapose.datasets.scene_dataset import ObjectData
from megapose.datasets.object_dataset import RigidObjectDataset, RigidObject
from megapose.inference.types import ObservationTensor, PoseEstimatesType
from megapose.inference.utils import make_detections_from_object_data
from megapose.utils.load_model import NAMED_MODELS, load_named_model
from megapose.lib3d.transform import Transform

import os
import logging

# Set required environment variables before importing Panda3D
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU
os.environ["DISPLAY"] = ":0"  # Set display
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

def get_pose_from_inputs(
    rgb: np.ndarray,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    mesh_path: str,
    bounding_box: tuple,
    model_name: str = "megapose-1.0-RGBD"
) -> PoseEstimatesType:
    """
    Given an RGB(+D) image, camera intrinsics, a bounding box, and a mesh,
    regress the object pose in the camera frame using MegaPose.
    """

    # 1. (Optional) set environment variables for headless usage, if necessary
    os.environ["DISPLAY"] = ":0"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["EGL_VISIBLE_DEVICES"] = "0"
    
    # 2. Create observation tensor (similar to run_inference_on_example.py)
    observation = ObservationTensor.from_numpy(
        rgb=rgb,
        depth=depth,  # might be None
        K=intrinsics
    ).cuda()

    # 3. Create object data + bounding box detection
    #    (like load_detections does, but directly from bounding_box)
    object_data = ObjectData(
        label="custom_object",
        bbox_modal=np.array(bounding_box)
    )
    detections = make_detections_from_object_data([object_data]).cuda()

    # 4. Create a minimal object dataset (just one object)
    rigid_object = RigidObject(
        label="custom_object",
        mesh_path=Path(mesh_path),
        mesh_units="m"  # adjust if your mesh is in different units
    )
    object_dataset = RigidObjectDataset([rigid_object])

    # 5. Load the MegaPose model WITHOUT the unused renderer_kwargs
    #    and run inference similarly to run_inference_on_example.py
    model_info = NAMED_MODELS[model_name]
    pose_estimator = load_named_model(model_name, object_dataset).cuda()

    # 6. Run the inference pipeline
    #    If you want to remove 'n_workers' or other keys, you can filter:
    inference_kwargs = {
        k: v for k, v in model_info["inference_parameters"].items()
        if k != "n_workers"  # or remove any other unsupported keys
    }

    # 7. PoseEstimatesType is returned by run_inference_pipeline
    predictions, _ = pose_estimator.run_inference_pipeline(
        observation,
        detections=detections,
        # you could also just do **model_info["inference_parameters"]
        # if none of them cause errors
        **inference_kwargs
    )
    
    return predictions

def main():
    """Example usage to demonstrate get_pose_from_inputs()."""
    # Create test inputs
    test_rgb = np.ones((128, 128, 3), dtype=np.uint8) * 255
    test_depth = None  # Or a valid depth array if your model needs it
    test_intrinsics = np.array([
        [500,   0, 64],
        [  0, 500, 64],
        [  0,   0,  1]
    ])
    test_bbox = (32, 32, 96, 96)
    
    # Update mesh_path to a valid local mesh (OBJ, PLY, etc.)
    mesh_path = "/home/yilong/Documents/action_extractor/action_extractor/megapose/mesh/panda_hand.obj"
    
    try:
        predictions = get_pose_from_inputs(
            rgb=test_rgb,
            depth=test_depth,
            intrinsics=test_intrinsics,
            mesh_path=mesh_path,
            bounding_box=test_bbox,
            model_name="megapose-1.0-RGBD"  # or e.g. "megapose-1.0-RGB"
        )
        print(f"Predicted poses: {predictions}")
    except Exception as e:
        print(f"Error in pose estimation: {e}")

if __name__ == "__main__":
    main()
