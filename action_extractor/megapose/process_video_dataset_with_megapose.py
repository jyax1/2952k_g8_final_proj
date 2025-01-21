#!/usr/bin/env python3

"""
process_video_dataset_with_megapose.py

1) Copies your original HDF5 file to a new file (so the original is untouched).
2) Loads the Megapose-based ActionIdentifierMegapose.
3) Iterates over each demo in the new file, infers actions from frames, 
   and overwrites the `root["data"][demo]["actions"]` dataset in place.

Example usage:
  python process_video_dataset_with_megapose.py \
    --input-file /path/to/original.hdf5 \
    --output-file /path/to/copied_and_updated.hdf5 \
    --hand-mesh-dir /path/to/panda_hand_mesh \
    --num-demos 100 \
    --batch-size 40
"""

import os
import shutil
import logging
import math
import argparse

import numpy as np
import h5py
import torch
import cv2
from tqdm import tqdm
from pathlib import Path

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

from robosuite.utils.camera_utils import get_camera_extrinsic_matrix, get_camera_intrinsic_matrix


def process_video_dataset_with_megapose(
    input_file: str,
    output_file: str,
    hand_mesh_dir: str,
    num_demos: int = 100,
    batch_size: int = 40,
):
    """
    :param input_file: Path to the original HDF5 file you want to copy & process
    :param output_file: Where to save the new HDF5 with updated actions
    :param hand_mesh_dir: Directory containing .obj or .ply mesh(es) for the 'panda-hand'
    :param num_demos: Max demos to process (0 => all)
    :param batch_size: Chunk size for ActionIdentifierMegapose
    """

    # ----------------------------------------------------------------
    # 0) Copy the original file to a new file so we can overwrite it
    # ----------------------------------------------------------------
    if os.path.abspath(input_file) == os.path.abspath(output_file):
        raise ValueError("Output file must differ from input file, so we can safely copy.")
    logger.info(f"Copying {input_file} => {output_file}")
    shutil.copy(input_file, output_file)

    # ----------------------------------------------------------------
    # 1) Create a "dummy" environment to retrieve camera intrinsics & extrinsics
    # ----------------------------------------------------------------
    logger.info(f"Reading env metadata from {input_file}")
    env_meta = get_env_metadata_from_dataset(input_file)
    obs_modality_specs = {
        "obs": {
            "rgb": ["frontview_image", "sideview_image"],   # adapt if needed
            "depth": ["frontview_depth", "sideview_depth"], # adapt if needed or remove
        }
    }
    ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs)
    env_camera_dummy = create_env_from_metadata(env_meta=env_meta, render_offscreen=True)

    # We'll read an example image shape from the input file
    # to get camera_height, camera_width
    with h5py.File(input_file, "r") as f_in:
        # Grab the first demo's first frame
        first_demo = list(f_in["data"].keys())[0]
        example_image = f_in["data"][first_demo]["obs"]["frontview_image"][0]
        camera_height, camera_width = example_image.shape[:2]

    # Then gather intrinsics / extrinsics for frontview / sideview
    logger.info("Gathering camera intrinsics/extrinsics (frontview, sideview).")
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

    # ----------------------------------------------------------------
    # 2) Load Megapose model once
    # ----------------------------------------------------------------
    logger.info("Loading hand mesh dataset...")
    hand_object_dataset = make_object_dataset_from_folder(Path(hand_mesh_dir))

    model_name = "megapose-1.0-RGB-multi-hypothesis"
    model_info = NAMED_MODELS[model_name]
    logger.info(f"Loading Megapose model '{model_name}'...")
    hand_pose_estimator = load_named_model(model_name, hand_object_dataset).cuda()

    # Build the ActionIdentifier
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

    # ----------------------------------------------------------------
    # 3) Open the newly copied file in r+ mode and overwrite the actions
    # ----------------------------------------------------------------
    with h5py.File(output_file, "r+") as f_out:
        demos = list(f_out["data"].keys())
        if num_demos > 0:
            demos = demos[:num_demos]

        logger.info(f"Processing {len(demos)} demos in {output_file} ...")
        for demo in tqdm(demos, desc="Updating actions"):
            obs_group = f_out["data"][demo]["obs"]
            num_samples = obs_group["frontview_image"].shape[0]

            # Gather front frames
            front_frames_list = obs_group["frontview_image"][:]  # shape (N, H, W, 3)
            # If you have depth:
            # front_depth_list = obs_group["frontview_depth"][:]  # shape (N, H, W)
            front_depth_list = None

            # side frames
            side_frames_list = obs_group["sideview_image"][:]  # shape (N, H, W, 3)

            # Convert them to list-of-frames if your action_identifier expects that
            front_frames_list = [frame for frame in front_frames_list]
            side_frames_list  = [frame for frame in side_frames_list]
            # If needed for depth:
            # front_depth_list = [d for d in front_depth_list]

            # 3A) get poses + finger distances
            all_hand_poses_world, all_fingers_distances = action_identifier.get_all_hand_poses_finger_distances(
                front_frames_list,
                front_depth_list=front_depth_list,
            )

            # 3B) compute actions
            actions_for_demo = action_identifier.compute_actions(
                all_hand_poses_world,
                all_fingers_distances,
                side_frames_list
            )
            actions_for_demo = np.array(actions_for_demo, dtype=np.float32)  # shape (N-1, 7) typically

            # 3C) Overwrite the existing actions
            old_shape = f_out["data"][demo]["actions"].shape
            new_shape = actions_for_demo.shape
            min_frames = min(new_shape[0], old_shape[0])
            min_dim    = min(new_shape[1], old_shape[1])
            f_out["data"][demo]["actions"][:min_frames, :min_dim] = actions_for_demo[:min_frames, :min_dim]

    logger.info(f"Done. Overwrote actions in {output_file}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, default="/home/yilong/Documents/policy_data/lift/raw/1736991916_9054875/test/lift_1000_obs.hdf5",
                        help="Path to the original HDF5 file")
    parser.add_argument("--output-file", type=str, default="/home/yilong/Documents/policy_data/lift/raw/1736991916_9054875/test/lift_300_obs_megapose.hdf5",
                        help="Where to save the new HDF5 file with updated actions")
    parser.add_argument("--hand-mesh-dir", type=str, default="/home/yilong/Documents/action_extractor/action_extractor/megapose/panda_hand_mesh",
                        help="Folder with .obj or .ply mesh for the 'panda-hand'")
    parser.add_argument("--num-demos", type=int, default=300, help="Max demos per file; 0 => all demos")
    parser.add_argument("--batch-size", type=int, default=40, help="Chunk size for Megapose inference")
    args = parser.parse_args()

    process_video_dataset_with_megapose(
        input_file=args.input_file,
        output_file=args.output_file,
        hand_mesh_dir=args.hand_mesh_dir,
        num_demos=args.num_demos,
        batch_size=args.batch_size,
    )
