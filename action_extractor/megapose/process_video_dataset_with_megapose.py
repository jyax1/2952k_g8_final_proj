#!/usr/bin/env python3

"""
process_video_dataset_with_megapose.py

Overwrites the existing 'actions' in your Robomimic / zarr dataset with newly
inferred actions from the Megapose-based ActionIdentifierMegapose.
"""

import os
import sys
import math
import shutil
import logging
import numpy as np
import torch
import cv2
import zarr

from pathlib import Path
from glob import glob
from tqdm import tqdm
from zarr import ZipStore, DirectoryStore

# Silence some logs
logging.getLogger("action_extractor.megapose.action_identifier_megapose").setLevel(logging.WARNING)

# Robomimic / Env utilities
import robomimic.utils.obs_utils as ObsUtils
from robomimic.utils.file_utils import get_env_metadata_from_dataset
from robomimic.utils.env_utils import create_env_from_metadata

# Megapose + your utilities
from megapose.utils.logging import get_logger
logger = get_logger(__name__)

from megapose.utils.load_model import NAMED_MODELS, load_named_model
from action_extractor.megapose.action_identifier_megapose import (
    ActionIdentifierMegapose, 
    make_object_dataset_from_folder,
)

from action_extractor.utils.dataset_utils import (
    hdf5_to_zarr_parallel_with_progress,
    directorystore_to_zarr_zip,
)
from robosuite.utils.camera_utils import get_camera_extrinsic_matrix, get_camera_intrinsic_matrix


###############################################################################
# Additional local helper functions (optional)
###############################################################################

def bounding_box_center(bbox):
    """
    Given a bbox = [x_min, y_min, x_max, y_max],
    return its center (cx, cy).
    """
    x_min, y_min, x_max, y_max = bbox
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    return cx, cy

def bounding_box_distance(bbox1, bbox2):
    """
    Compute the Euclidean distance between the centers of two bounding boxes.
    """
    cx1, cy1 = bounding_box_center(bbox1)
    cx2, cy2 = bounding_box_center(bbox2)
    return math.hypot(cx2 - cx1, cy2 - cy1)


###############################################################################
# Main script function
###############################################################################

def process_video_dataset_with_megapose(
    dataset_path: str,
    hand_mesh_dir: str,
    output_log_dir: str,
    num_demos: int = 100,
    batch_size: int = 40,
):
    """
    1) Finds all .hdf5 in dataset_path, converts them to .zarr.zip if needed.
    2) Loads a Megapose model once.
    3) Iterates over each demo in each zarr archive,
       computes new actions via ActionIdentifierMegapose,
       and overwrites root_z["data"][demo]["actions"] in place.
    4) Logs results in a text file in output_log_dir.

    :param dataset_path: e.g. "/home/yilong/Documents/policy_data/lift/raw/1736991916_9054875/test"
    :param hand_mesh_dir: path to folder with your .obj / .ply of 'panda-hand'
    :param output_log_dir: where to write e.g. "megapose_processed_actions.txt"
    :param num_demos: how many demos per zarr file to process (0 => all).
    :param batch_size: chunk size for inference in ActionIdentifierMegapose
    """
    # 0) Prepare output directory
    os.makedirs(output_log_dir, exist_ok=True)
    log_filename = os.path.join(output_log_dir, "megapose_processed_actions.txt")

    # 1) Possibly do HDF5 => Zarr if not done
    sequence_dirs = glob(f"{dataset_path}/**/*.hdf5", recursive=True)
    for seq_dir in sequence_dirs:
        ds_dir = seq_dir.replace(".hdf5", ".zarr")
        zarr_path = seq_dir.replace(".hdf5", ".zarr.zip")
        if not os.path.exists(zarr_path):
            logger.info(f"Converting {seq_dir} -> {zarr_path}")
            hdf5_to_zarr_parallel_with_progress(seq_dir, max_workers=16)
            store_tmp = DirectoryStore(ds_dir)
            root_tmp = zarr.group(store_tmp, overwrite=False)
            store_tmp.close()
            directorystore_to_zarr_zip(ds_dir, zarr_path)
            shutil.rmtree(ds_dir)

    # 2) Collect all .zarr.zip
    zarr_files = glob(f"{dataset_path}/**/*.zarr.zip", recursive=True)
    if not zarr_files:
        logger.warning(f"No .zarr.zip files found in {dataset_path}")
        return

    # 3) Load object_dataset + model once
    logger.info("Loading hand mesh dataset...")
    hand_object_dataset = make_object_dataset_from_folder(Path(hand_mesh_dir))
    model_name = "megapose-1.0-RGB-multi-hypothesis"
    model_info = NAMED_MODELS[model_name]
    logger.info(f"Loading Megapose model '{model_name}'...")
    hand_pose_estimator = load_named_model(model_name, hand_object_dataset).cuda()

    # 4) We'll create a dummy environment to read intrinsics/extrinsics from
    logger.info("Creating dummy environment to extract camera data (frontview/sideview).")
    env_meta = get_env_metadata_from_dataset(dataset_path=sequence_dirs[0])
    obs_modality_specs = {
        "obs": {
            "rgb": ["frontview_image", "sideview_image"],   # or your camera names
            "depth": ["frontview_depth", "sideview_depth"], # optional
        }
    }
    ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs)

    # Create environment to read camera properties
    env_camera_dummy = create_env_from_metadata(env_meta=env_meta, render_offscreen=True)

    # read an example image shape from the first dataset
    store_check = ZipStore(zarr_files[0], mode='r')
    root_check = zarr.group(store_check)
    first_demo_name = list(root_check["data"].keys())[0]
    example_image = root_check["data"][first_demo_name]["obs"]["frontview_image"][0]
    store_check.close()

    camera_height, camera_width = example_image.shape[:2]

    frontview_K = get_camera_intrinsic_matrix(
        env_camera_dummy.env.sim,
        camera_name="frontview",
        camera_height=camera_height,
        camera_width=camera_width,
    )
    sideview_K = get_camera_intrinsic_matrix(
        env_camera_dummy.env.sim,
        camera_name="sideview",
        camera_height=camera_height,
        camera_width=camera_width,
    )
    frontview_R = get_camera_extrinsic_matrix(env_camera_dummy.env.sim, camera_name="frontview")
    sideview_R = get_camera_extrinsic_matrix(env_camera_dummy.env.sim, camera_name="sideview")

    # 5) Build the ActionIdentifierMegapose
    action_identifier = ActionIdentifierMegapose(
        pose_estimator=hand_pose_estimator,
        frontview_R=frontview_R,
        frontview_K=frontview_K,
        sideview_R=sideview_R,
        sideview_K=sideview_K,
        model_info=model_info,
        batch_size=batch_size,
        scale_translation=80.0,
    )

    # 6) Process each .zarr.zip
    all_logs = []
    for zarr_file in zarr_files:
        logger.info(f"Processing {zarr_file}")
        # Open in read+write (r+) so we can overwrite the "actions" dataset
        store = ZipStore(zarr_file, mode='r+')
        root_z = zarr.group(store, overwrite=False)

        # Gather demos
        demos = list(root_z["data"].keys())
        if num_demos > 0:
            demos = demos[:num_demos]

        for demo in tqdm(demos, desc=f"Overwriting actions in {zarr_file}"):
            obs_group = root_z["data"][demo]["obs"]
            # shape (#frames, H, W, 3)
            num_samples = obs_group["frontview_image"].shape[0]

            # Gather front frames & depths
            front_frames_list = [obs_group["frontview_image"][i] for i in range(num_samples)]
            # If you have front_depth available:
            # front_depth_list = [obs_group["frontview_depth"][i] for i in range(num_samples)]
            # or set to None if not used
            front_depth_list = None

            # Gather side frames
            side_frames_list = [obs_group["sideview_image"][i] for i in range(num_samples)]

            # 6A) Get the (n) poses in world frame + finger distances
            all_hand_poses_world, all_fingers_distances = action_identifier.get_all_hand_poses_finger_distances(
                front_frames_list, 
                front_depth_list=front_depth_list,
            )

            # 6B) Compute the (n-1) actions
            actions_for_demo = action_identifier.compute_actions(
                all_hand_poses_world, 
                all_fingers_distances,
                side_frames_list,
            )
            actions_for_demo = np.array(actions_for_demo, dtype=np.float32)  # shape (n-1, 7) e.g.

            # 6C) Overwrite root_z["data"][demo]["actions"]
            # We assume it originally has shape (n-1, action_dim).
            old_shape = root_z["data"][demo]["actions"].shape
            new_shape = actions_for_demo.shape
            if new_shape != old_shape:
                logger.warning(f"demo {demo} in {zarr_file} has old actions shape={old_shape}, new={new_shape}. Attempting partial overwrite if possible.")
            
            # If new_shape is smaller or the same, we can do partial overwrite:
            # e.g. root_z["data"][demo]["actions"][:new_shape[0], :new_shape[1]] = actions_for_demo
            # For exact match:
            min_frames = min(new_shape[0], old_shape[0])
            min_dim    = min(new_shape[1], old_shape[1])
            
            root_z["data"][demo]["actions"][:min_frames, :min_dim] = actions_for_demo[:min_frames, :min_dim]
            
            # Log
            all_logs.append(f"{zarr_file} / {demo}: Overwrote actions with shape {new_shape}")

        # Close store
        store.close()

    # 7) Write logs
    with open(log_filename, "w") as f:
        f.write("\n".join(all_logs))
    logger.info(f"Done. Wrote logs to {log_filename}")


###############################################################################
# main entry
###############################################################################

if __name__ == "__main__":
    """
    Example usage:
      python process_video_dataset_with_megapose.py \
        --dataset-path /path/to/dataset \
        --hand-mesh-dir /path/to/mesh \
        --output-log-dir /path/for/logs \
        --num-demos 100 \
        --batch-size 40
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to your dataset containing .hdf5 or .zarr.zip files")
    parser.add_argument("--hand-mesh-dir", type=str, required=True, help="Folder with .obj or .ply mesh for the 'panda-hand'")
    parser.add_argument("--output-log-dir", type=str, default="megapose_logs", help="Where to write final logs")
    parser.add_argument("--num-demos", type=int, default=100, help="max demos per file; 0 => process all demos")
    parser.add_argument("--batch-size", type=int, default=40, help="Batch size for chunked Megapose inference")
    args = parser.parse_args()

    process_video_dataset_with_megapose(
        dataset_path=args.dataset_path,
        hand_mesh_dir=args.hand_mesh_dir,
        output_log_dir=args.output_log_dir,
        num_demos=args.num_demos,
        batch_size=args.batch_size,
    )
