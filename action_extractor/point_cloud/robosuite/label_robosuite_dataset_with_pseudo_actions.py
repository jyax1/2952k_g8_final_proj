'''
action_extractor/point_cloud/robosuite/label_robosuite_dataset_with_pseudo_actions.py

For each demo in a SINGLE .hdf5 file:
  1) Reconstruct hand poses from point clouds.
  2) Convert those poses into absolute (or delta) actions.
  3) Copy the original 'actions' dataset to 'pseudo_actions' (if it doesn't exist yet).
  4) Overwrite the 'actions' dataset with these newly estimated actions.
     (Shape and dtype are enforced to match the existing 'actions' dataset exactly.)
'''

import os
import argparse
import h5py
import json
from tqdm import tqdm
import numpy as np
from copy import deepcopy

import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.env_utils as EnvUtils
from robomimic.utils.file_utils import get_env_metadata_from_dataset

from action_extractor.utils.angles_utils import load_ground_truth_poses
from action_extractor.point_cloud.generate_point_clouds_from_dataset import reconstruct_pointclouds_from_obs_group
from action_extractor.utils.poses_to_actions import (
    poses_to_absolute_actions,
    poses_to_delta_actions
)
from action_extractor.point_cloud.action_identifier_point_cloud import (
    get_poses_from_pointclouds
)
from action_extractor.point_cloud.config import *
from action_extractor.utils.robosuite_data_processing_utils import *

def exclude_cameras_from_obs(traj, camera_names):
    if len(camera_names) > 0:
        for cam in camera_names:
            del traj['obs'][f"{cam}_image"]
            del traj['obs'][f"{cam}_depth"]
            del traj['obs'][f"{cam}_rgbd"]

def extract_trajectory(
    env_meta,
    camera_names,
    initial_state, 
    states, 
    actions,
):
    """
    Helper function to extract observations, rewards, and dones along a trajectory using
    the simulator environment.

    Args:
        env (instance of EnvBase): environment
        initial_state (dict): initial simulation state to load
        states (np.array): array of simulation states to load to extract information
        actions (np.array): array of actions
        done_mode (int): how to write done signal. If 0, done is 1 whenever s' is a 
            success state. If 1, done is 1 at the end of each trajectory. 
            If 2, do both.
    """
    done_mode = 0
    env = EnvUtils.create_env_for_data_processing(
        env_meta=env_meta,
        camera_names=camera_names, 
        camera_height=CAMERA_HEIGHT, 
        camera_width=CAMERA_WIDTH, 
        reward_shaping=False,
    )
    assert states.shape[0] == actions.shape[0]

    # load the initial state
    env.reset()
    insert_camera_info(initial_state)
    obs = env.reset_to(initial_state)

    traj = dict(
        obs=[], 
        next_obs=[], 
        rewards=[], 
        dones=[], 
        actions=np.array(actions), 
        states=np.array(states), 
        initial_state_dict=initial_state,
    )
    traj_len = states.shape[0]
    # iteration variable @t is over "next obs" indices
    for t in range(1, traj_len + 1):

        # get next observation
        if t == traj_len:
            # play final action to get next observation for last timestep
            next_obs, _, _, _ = env.step(actions[t - 1])
        else:
            # reset to simulator state to get observation
            next_obs = env.reset_to({"states" : states[t]})

        # infer reward signal
        # note: our tasks use reward r(s'), reward AFTER transition, so this is
        #       the reward for the current timestep
        r = env.get_reward()

        # infer done signal
        done = False
        if (done_mode == 1) or (done_mode == 2):
            # done = 1 at end of trajectory
            done = done or (t == traj_len)
        if (done_mode == 0) or (done_mode == 2):
            # done = 1 when s' is task success state
            done = done or env.is_success()["task"]
        done = int(done)

        # collect transition
        traj["obs"].append(obs)
        traj["next_obs"].append(next_obs)
        traj["rewards"].append(r)
        traj["dones"].append(done)

        # update for next iter
        obs = deepcopy(next_obs)

    # convert list of dict to dict of list for obs dictionaries (for convenient writes to hdf5 dataset)
    traj["obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["obs"])
    traj["next_obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["next_obs"])

    # list to numpy array
    for k in traj:
        if k == "initial_state_dict":
            continue
        if isinstance(traj[k], dict):
            for kp in traj[k]:
                traj[k][kp] = np.array(traj[k][kp])
        else:
            traj[k] = np.array(traj[k])

    return traj

def worker(x):
    env_meta, camera_names, initial_state, states, actions = x
    traj = extract_trajectory(
        env_meta=env_meta,
        camera_names=camera_names,
        initial_state=initial_state, 
        states=states, 
        actions=actions,
    )
    return traj

def label_dataset_with_pseudo_actions(args: argparse.Namespace) -> None:
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
        offset (list or np.ndarray): 3-element translation offset for ICP. Default [0,0,0].
        icp_method (str): Which ICP method to use. Default = "multiscale".
    """
    
    print(f"Retrieving env metadata from: {args.hdf5_path}")
    env_meta = get_env_metadata_from_dataset(args.hdf5_path)
    if args.absolute_actions:
        env_meta["env_kwargs"]["controller_configs"]["control_delta"] = False
        env_meta["env_kwargs"]["controller_configs"]["type"] = "OSC_POSE"
    env = EnvUtils.create_env_for_data_processing(
        env_meta=env_meta,
        camera_names=CAMERAS_FOR_POLICY + ADDITIONAL_CAMERAS_FOR_POINT_CLOUD, 
        camera_height=CAMERA_HEIGHT, 
        camera_width=CAMERA_WIDTH, 
        reward_shaping=False,
    )
    control_freq = env.env.control_freq  # used for up/downsampling
    policy_freq = control_freq
    
    is_robosuite_env = EnvUtils.is_robosuite_env(env_meta)

    # list of all demonstration episodes (sorted in increasing number order)
    f = h5py.File(args.hdf5_path, "r")
    demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    # maybe reduce the number of demonstrations to playback
    if args.num_demos is not None:
        demos = demos[:args.num_demos]

    # output file in same directory as input file
    output_path = os.path.join(os.path.dirname(args.hdf5_path), args.output_hdf5_path)
    f_out = h5py.File(output_path, "w")
    data_grp = f_out.create_group("data")
    print("input file: {}".format(args.hdf5_path))
    print("output file: {}".format(output_path))

    total_samples = 0
    
    for i in range(0, len(demos), args.num_workers):
        end = min(i + args.num_workers, len(demos))
        initial_state_list = []
        states_list = []
        actions_list = []
        for j in range(i, end):
            ep = demos[j]
            # prepare initial state to reload from
            states = f["data/{}/states".format(ep)][()]
            initial_state = dict(states=states[0])
            if is_robosuite_env:
                xml_str = f["data/{}".format(ep)].attrs["model_file"]
                xml_str = replace_all_lights(xml_str)
                xml_str = recolor_gripper(xml_str)
                initial_state["model"] = xml_str
            actions = f["data/{}/actions".format(ep)][()]

            initial_state_list.append(initial_state)
            states_list.append(states)
            actions_list.append(actions)

        with multiprocessing.Pool(args.num_workers) as pool:
            trajs = pool.map(worker, [[env_meta, CAMERAS_FOR_POLICY + ADDITIONAL_CAMERAS_FOR_POINT_CLOUD, initial_state_list[j], states_list[j], actions_list[j]] for j in range(len(initial_state_list))]) 

        for j, ind in enumerate(range(i, end)):
            ep = demos[ind]
            traj = trajs[j]
            exclude_cameras_from_obs(traj, ADDITIONAL_CAMERAS_FOR_POINT_CLOUD)
            # maybe copy reward or done signal from source file
            # if args.copy_rewards:
            #     traj["rewards"] = f["data/{}/rewards".format(ep)][()]
            # if args.copy_dones:
            #     traj["dones"] = f["data/{}/dones".format(ep)][()]

            # store transitions

            # IMPORTANT: keep name of group the same as source file, to make sure that filter keys are
            #            consistent as well
            ep_data_grp = data_grp.create_group(ep)
            ep_data_grp.create_dataset("actions", data=np.array(traj["actions"]))
            ep_data_grp.create_dataset("states", data=np.array(traj["states"]))
            ep_data_grp.create_dataset("rewards", data=np.array(traj["rewards"]))
            ep_data_grp.create_dataset("dones", data=np.array(traj["dones"]))
            for k in traj["obs"]:
                if args.compress:
                    ep_data_grp.create_dataset("obs/{}".format(k), data=np.array(traj["obs"][k]), compression="gzip")
                else:
                    ep_data_grp.create_dataset("obs/{}".format(k), data=np.array(traj["obs"][k]))
                if not args.exclude_next_obs:
                    if args.compress:
                        ep_data_grp.create_dataset("next_obs/{}".format(k), data=np.array(traj["next_obs"][k]), compression="gzip")
                    else:
                        ep_data_grp.create_dataset("next_obs/{}".format(k), data=np.array(traj["next_obs"][k]))

            # episode metadata
            if is_robosuite_env:
                ep_data_grp.attrs["model_file"] = traj["initial_state_dict"]["model"] # model xml for this episode
            ep_data_grp.attrs["num_samples"] = traj["actions"].shape[0] # number of transitions in this episode
            total_samples += traj["actions"].shape[0]
            print("ep {}: wrote {} transitions to group {}".format(ind, ep_data_grp.attrs["num_samples"], ep))
        
        del trajs

    # copy over all filter keys that exist in the original hdf5
    if "mask" in f:
        f.copy("mask", f_out)

    # global metadata
    data_grp.attrs["total"] = total_samples
    data_grp.attrs["env_args"] = json.dumps(env.serialize(), indent=4) # environment info
    print("Wrote {} trajectories to {}".format(len(demos), output_path))

    f.close()
    f_out.close()

    # # 5) Loop over demos and update actions
    # for demo in tqdm(all_demos, desc="Labeling demos with pseudo-actions"):
    #     obs_group = root_h["data"][demo]["obs"]
    #     old_actions_ds = root_h["data"][demo]["actions"]  # existing HDF5 dataset
    #     num_samples = old_actions_ds.shape[0]

    #     # 5A) Copy original actions to a backup dataset if not already present
    #     if "pseudo_actions" not in root_h["data"][demo]:
    #         # Make a copy so we preserve the original data
    #         root_h["data"][demo].create_dataset(
    #             "pseudo_actions",
    #             data=old_actions_ds[:],
    #             dtype=old_actions_ds.dtype
    #         )
    #     else:
    #         if verbose:
    #             print(f"Demo {demo} already has an 'pseudo_actions' dataset. Not overwriting it.")

    #     # 6) Build pointcloud from each camera frame, exactly as in the 'visualize' script
    #     point_clouds_points, point_clouds_colors = reconstruct_pointclouds_from_obs_group(
    #         obs_group,
    #         env.env,  # pass the underlying env for camera intrinsics/extrinsics
    #         camera_names,
    #         camera_height,
    #         camera_width,
    #         verbose=verbose,
    #     )

    #     # 7) Get poses from ground truth or ICP
    #     if ground_truth:
    #         all_hand_poses = load_ground_truth_poses(obs_group)
    #     else:
    #         all_hand_poses = get_poses_from_pointclouds(
    #             point_clouds_points,
    #             point_clouds_colors,
    #             hand_mesh,
    #             verbose=verbose,
    #             offset=offset,
    #             debug_dir=None,  # or some debug directory if you want
    #             icp_method=icp_method,
    #         )

    #     # 8) Convert poses -> actions
    #     #    We keep the original "gripper_actions" from old_actions_ds but
    #     #    update the x,y,z,quat portion from all_hand_poses.
    #     gripper_actions = [old_actions_ds[i][-1] for i in range(num_samples)]
    #     if absolute_actions:
    #         new_actions = poses_to_absolute_actions(
    #             poses=all_hand_poses,
    #             gripper_actions=gripper_actions,
    #             env=env,
    #             control_freq=control_freq,
    #             policy_freq=policy_freq,
    #             smooth=smooth
    #         )
    #     else:
    #         new_actions = poses_to_delta_actions(
    #             poses=all_hand_poses,
    #             gripper_actions=gripper_actions,
    #             smooth=False,
    #             translation_scaling=80.0,
    #             rotation_scaling=9.0,
    #         )

    #     # 9) Ensure shape & dtype match. Then overwrite the original 'actions' dataset.
    #     if new_actions.shape != old_actions_ds.shape:
    #         raise ValueError(
    #             f"New actions shape {new_actions.shape} != original {old_actions_ds.shape} for demo {demo}"
    #         )
    #     new_actions = new_actions.astype(old_actions_ds.dtype, copy=False)

    #     # 10) Overwrite the old "actions" data
    #     old_actions_ds[:] = new_actions
    

    print(f"Done labeling dataset with pseudo-actions.\nSaved to: {args.output_hdf5_path}")


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
    parser.add_argument("--num_workers", type=int, default=16, # Maximum num_workers suitable for my machine
                        help="Number of workers for parallel saving")
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

    args = parser.parse_args()

    label_dataset_with_pseudo_actions(args)


if __name__ == "__main__":
    main()