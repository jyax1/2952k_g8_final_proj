#!/usr/bin/env python
"""
label_dataset_with_pseudo_actions.py

For each demo in a SINGLE .hdf5 file:
  1) Reconstruct hand poses from point clouds.
  2) Convert those poses into absolute (or delta) actions.
  3) Copy the original 'actions' dataset to 'pseudo_actions' (if it doesn't exist yet).
  4) Overwrite the 'actions' dataset with these newly estimated actions.
     (Shape and dtype are enforced to match the existing 'actions' dataset exactly.)
"""

import os
import argparse
import h5py
from tqdm import tqdm

import robomimic.utils.obs_utils as ObsUtils
from robomimic.utils.file_utils import get_env_metadata_from_dataset
from robomimic.utils.env_utils import create_env_from_metadata

from action_extractor.utils.angles_utils import load_ground_truth_poses
from action_extractor.point_cloud.generate_point_clouds_from_dataset import reconstruct_pointclouds_from_obs_group
from action_extractor.utils.poses_to_actions import (
    poses_to_absolute_actions,
    poses_to_delta_actions
)
from action_extractor.point_cloud.action_identifier_point_cloud import (
    get_poses_from_pointclouds
)
from action_extractor.point_cloud.config import POSITIONAL_OFFSET


def label_dataset_with_pseudo_actions(
    hdf5_path,
    hand_mesh,
    cameras,
    output_hdf5_path,
    num_demos=None,
    absolute_actions=True,
    ground_truth=False,
    smooth=False,
    verbose=True,
    offset=None,
    icp_method="multiscale",
):
    """
    Given a single .hdf5 dataset file, for each demo:
     - Reconstruct point clouds from the specified cameras.
     - Estimate hand poses (ICP or ground-truth).
     - Convert those poses to actions.
     - Copy the old actions dataset into "pseudo_actions" if it doesn't already exist.
     - Overwrite the existing 'actions' dataset (shape, dtype) with these new actions.

    Args:
        hdf5_path (str): Path to the input .hdf5 file.
        hand_mesh (str): Path to the hand mesh .ply for ICP alignment.
        cameras (list of str): Camera observation keys (e.g. ["frontview_image"]).
        output_hdf5_path (str): Where to save the updated HDF5 file. If the path is
                           the same as hdf5_path, we overwrite in-place. If it's
                           different, we copy the file and modify the copy.
        num_demos (int or None): Max number of demos to process (if None, process all).
        absolute_actions (bool): If True, use poses_to_absolute_actions; else delta actions.
        ground_truth (bool): If True, use ground-truth poses from the dataset (no ICP).
        policy_freq (int): The "policy frequency" used for upsampling actions.
        smooth (bool): Whether to smooth the end-effector path.
        verbose (bool): Print debug info, if desired.
        offset (list or np.ndarray): 3-element translation offset for ICP. Default None -> [0,0,0].
        icp_method (str): Which ICP method to use. Default = "multiscale".
    """
    if offset is None:
        offset = [0, 0, 0]

    # 1) Copy or open the file in write mode. If output_hdf5_path == None, we modify in-place.
    if output_hdf5_path == None:
        # Overwrite the file in-place
        print(f"Overwriting dataset in-place: {hdf5_path}")
        with h5py.File(hdf5_path, "r+") as root_h:
            _process_file_and_write_actions(
                root_h,
                hdf5_path,
                hand_mesh,
                cameras,
                num_demos,
                absolute_actions,
                ground_truth,
                smooth,
                verbose,
                offset,
                icp_method
            )
    else:
        # If output_hdf5_path differs, copy the file first
        import shutil
        print(f"Copying {hdf5_path} -> {output_hdf5_path} ...")
        shutil.copy2(hdf5_path, output_hdf5_path)
        with h5py.File(output_hdf5_path, "r+") as root_h:
            _process_file_and_write_actions(
                root_h,
                output_hdf5_path,
                hand_mesh,
                cameras,
                num_demos,
                absolute_actions,
                ground_truth,
                smooth,
                verbose,
                offset,
                icp_method
            )

    print(f"Done labeling dataset with pseudo-actions.\nSaved to: {output_hdf5_path}")


def _process_file_and_write_actions(
    root_h,
    file_path,
    hand_mesh,
    cameras,
    num_demos,
    absolute_actions,
    ground_truth,
    smooth,
    verbose,
    offset,
    icp_method
):
    """
    Helper that loops over demos in an open h5py.File handle, copies the original
    "actions" dataset (if not already copied), and overwrites it with the newly
    computed pseudo-actions.
    """

    # 2) Create environment from metadata (for 'poses_to_absolute_actions', if needed)
    print(f"Retrieving env metadata from: {file_path}")
    env_meta = get_env_metadata_from_dataset(file_path)
    if absolute_actions:
        env_meta["env_kwargs"]["controller_configs"]["control_delta"] = False
        env_meta["env_kwargs"]["controller_configs"]["type"] = "OSC_POSE"
    env = create_env_from_metadata(env_meta=env_meta, render_offscreen=True)
    control_freq = env.env.control_freq  # used for up/downsampling
    policy_freq = control_freq

    # 3) Prepare obs utils
    camera_names = [cam.split("_")[0] for cam in cameras]
    depth_list = [f"{cam.split('_')[0]}_depth" for cam in cameras]
    obs_modality_specs = {
        "obs": {
            "rgb": cameras,
            "depth": depth_list,
        }
    }
    ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs)

    # 4) Gather demos
    all_demos = list(root_h["data"].keys())
    if num_demos:
        all_demos = all_demos[:num_demos]
    if len(all_demos) == 0:
        print("No demos found in this file!")
        return

    # Grab an example image to figure out camera dims
    example_demo = all_demos[0]
    example_image = root_h["data"][example_demo]["obs"][cameras[0]][0]
    camera_height, camera_width = example_image.shape[:2]

    # 5) Loop over demos and update actions
    for demo in tqdm(all_demos, desc="Labeling demos with pseudo-actions"):
        obs_group = root_h["data"][demo]["obs"]
        old_actions_ds = root_h["data"][demo]["actions"]  # existing HDF5 dataset
        num_samples = old_actions_ds.shape[0]

        # 5A) Copy original actions to a backup dataset if not already present
        if "pseudo_actions" not in root_h["data"][demo]:
            # Make a copy so we preserve the original data
            root_h["data"][demo].create_dataset(
                "pseudo_actions",
                data=old_actions_ds[:],
                dtype=old_actions_ds.dtype
            )
        else:
            if verbose:
                print(f"Demo {demo} already has an 'pseudo_actions' dataset. Not overwriting it.")

        # 6) Build pointcloud from each camera frame, exactly as in the 'visualize' script
        point_clouds_points, point_clouds_colors = reconstruct_pointclouds_from_obs_group(
            obs_group,
            env.env,  # pass the underlying env for camera intrinsics/extrinsics
            camera_names,
            camera_height,
            camera_width,
            verbose=verbose,
        )

        # 7) Get poses from ground truth or ICP
        if ground_truth:
            all_hand_poses = load_ground_truth_poses(obs_group)
        else:
            all_hand_poses = get_poses_from_pointclouds(
                point_clouds_points,
                point_clouds_colors,
                hand_mesh,
                verbose=verbose,
                offset=offset,
                debug_dir=None,  # or some debug directory if you want
                icp_method=icp_method,
            )

        # 8) Convert poses -> actions
        #    We keep the original "gripper_actions" from old_actions_ds but
        #    update the x,y,z,quat portion from all_hand_poses.
        gripper_actions = [old_actions_ds[i][-1] for i in range(num_samples)]
        if absolute_actions:
            new_actions = poses_to_absolute_actions(
                poses=all_hand_poses,
                gripper_actions=gripper_actions,
                env=env,
                control_freq=control_freq,
                policy_freq=policy_freq,
                smooth=smooth
            )
        else:
            new_actions = poses_to_delta_actions(
                poses=all_hand_poses,
                gripper_actions=gripper_actions,
                smooth=False,
                translation_scaling=80.0,
                rotation_scaling=9.0,
            )

        # 9) Ensure shape & dtype match. Then overwrite the original 'actions' dataset.
        if new_actions.shape != old_actions_ds.shape:
            raise ValueError(
                f"New actions shape {new_actions.shape} != original {old_actions_ds.shape} for demo {demo}"
            )
        new_actions = new_actions.astype(old_actions_ds.dtype, copy=False)

        # 10) Overwrite the old "actions" data
        old_actions_ds[:] = new_actions


def main():
    parser = argparse.ArgumentParser(
        description="Rewrite 'actions' in a single .hdf5 dataset with pseudo-actions (estimated from point clouds)."
    )

    parser.add_argument("--hdf5_path", type=str, required=True,
                        help="Path to a single .hdf5 dataset file")
    parser.add_argument("--output_hdf5_path", type=str, default=None,
                        help="Path to output the updated .hdf5. "
                             "If None, modifies in-place; else copies first.")
    parser.add_argument("--num_demos", type=int, default=None,
                        help="Number of demos to process (if None, do all).")
    parser.add_argument("--absolute_actions", action="store_true",
                        help="If set, use poses_to_absolute_actions instead of delta actions.")
    parser.add_argument("--ground_truth", action="store_true",
                        help="If set, use ground-truth poses from the dataset instead of ICP.")
    parser.add_argument("--smooth", action="store_true",
                        help="If set, attempts to smooth the resulting action path.")
    parser.add_argument("--verbose", action="store_true",
                        help="If set, print debug information.")
    parser.add_argument("--icp_method", type=str, default="multiscale", choices=["multiscale", "updown"],
                        help="ICP method used for pose estimation.")
    parser.add_argument('--cameras',
                        type=str,
                        nargs='+',
                        default=[
                            'squared0view_image', 'squared0view2_image',
                            'squared0view3_image', 'squared0view4_image',
                            'frontview_image', 'fronttableview_image',
                            'sidetableview_image', 'sideview2_image',
                            'backview_image'
                        ],
                        help='Space separated list of cameras for pointcloud reconstruction. All must be available in the dataset.')

    args = parser.parse_args()

    label_dataset_with_pseudo_actions(
        hdf5_path        = args.hdf5_path,
        hand_mesh        = "*/data/meshes/panda_hand_mesh/panda-hand.ply",
        cameras          = args.cameras,
        output_hdf5_path = args.output_hdf5_path,
        num_demos        = args.num_demos,
        absolute_actions = args.absolute_actions,
        ground_truth     = args.ground_truth,
        smooth           = args.smooth,
        verbose          = args.verbose,
        offset           = POSITIONAL_OFFSET,
        icp_method       = args.icp_method
    )


if __name__ == "__main__":
    main()
