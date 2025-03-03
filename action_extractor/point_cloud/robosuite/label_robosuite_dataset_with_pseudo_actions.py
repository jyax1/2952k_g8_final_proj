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

from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder # For debugging

import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.env_utils as EnvUtils
from robomimic.utils.env_utils import create_env_from_metadata
from robomimic.utils.file_utils import get_env_metadata_from_dataset
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
from action_extractor.utils.rollout_debug_utils import save_point_clouds_as_ply

def exclude_cameras_from_obs(traj, camera_names):
    if len(camera_names) > 0:
        for cam in camera_names:
            del traj['obs'][f"{cam}_image"]
            del traj['obs'][f"{cam}_depth"]
            del traj['obs'][f"{cam}_rgbd"]
    del traj['obs']['pointcloud_points']
    del traj['obs']['pointcloud_colors']
    
def roll_out(env, actions_for_demo, verbose=False, file_name = 'roll_out.mp4') -> bool:
    '''
    Given a list of actions, an environment, roll out the actions in the environment and return success.
    '''
    if verbose:
        debug_dir = 'debug/label_robosuite_dataset'
        os.makedirs(debug_dir, exist_ok=True)
        env.file_path = env.file_path = os.path.join(debug_dir, file_name)
        env.step_count = 0
    
    for action in actions_for_demo:
        env.step(action)
        
    if verbose:
        env.video_recoder.stop()
        print(f"Rollout complete. Video saved to {env.file_path}")
    
    return env.is_success()["task"]

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
    # insert_camera_info(initial_state) Don't need if we already have obs dataset
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
            next_obs, _, _, _ = env.step(actions[t - 1], get_point_clouds=True)
        else:
            # reset to simulator state to get observation
            next_obs = env.reset_to({"states" : states[t]}, get_point_clouds=True)

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
        args (argparse.Namespace): parsed arguments
    """
    
    print(f"Retrieving env metadata from: {args.hdf5_path}")
    env_meta = get_env_metadata_from_dataset(args.hdf5_path)
    if not args.delta_actions:
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
    
    if args.verbose or args.count_rollout_success:
        env_rollout = create_env_from_metadata(env_meta=env_meta, render_offscreen=True)
        env_rollout = VideoRecordingWrapper(
            env_rollout,
            video_recoder=VideoRecorder.create_h264(fps=20, codec="h264", input_pix_fmt="rgb24", crf=22),
            steps_per_render=1,
            width=CAMERA_WIDTH,
            height=CAMERA_HEIGHT,
            mode="rgb_array",
            camera_name='fronttableview',
        )
        
        results_file_path = os.path.join("logs", "labeling_rollouts", "rollouts_success.txt")
        os.makedirs(os.path.dirname(results_file_path), exist_ok=True)
        with open(results_file_path, "w") as results_txt:
            results_txt.write("Trajectory results:\n")

    # list of all demonstration episodes (sorted in increasing number order)
    f = h5py.File(args.hdf5_path, "r")
    demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    # maybe reduce the number of demonstrations to playback
    if args.num_demos is not None:
        demos = demos[:args.num_demos]
        
    if args.debug_demo is not None:
        demos = [demos[args.debug_demo]]

    # output file in same directory as input file
    output_path = os.path.join(os.path.dirname(args.hdf5_path), args.output_hdf5_name)
    f_out = h5py.File(output_path, "w")
    data_grp = f_out.create_group("data")
    print("input file: {}".format(args.hdf5_path))
    print("output file: {}".format(output_path))

    total_samples = 0
    
    n_success = 0
    total_n = 0
    
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
            
        tasks = [
            [env_meta, CAMERAS_FOR_POLICY + ADDITIONAL_CAMERAS_FOR_POINT_CLOUD, 
             initial_state_list[j], states_list[j], actions_list[j]]
            for j in range(len(initial_state_list))
        ]
        
        if args.num_workers == 1:
            # Process tasks sequentially if only one worker is specified
            trajs = [worker(task) for task in tasks]
        else:
            with multiprocessing.Pool(args.num_workers) as pool:
                trajs = pool.map(worker, tasks)

        for j, ind in enumerate(range(i, end)):
            ep = demos[ind]
            traj = trajs[j]
            
            pointcloud_points = traj['obs']['pointcloud_points']
            pointcloud_colors = traj['obs']['pointcloud_colors']
            
            if args.verbose:
                save_point_clouds_as_ply(pointcloud_points, pointcloud_colors, output_dir=os.path.join('debug/label_robosuite_dataset', f"pointclouds_{j}"))
                
            all_hand_poses = get_poses_from_pointclouds(
                [pointcloud_points[i] for i in range(len(pointcloud_points))],
                [pointcloud_colors[i] for i in range(len(pointcloud_colors))],
                model_path = "data/meshes/panda_hand_mesh/panda-hand.ply",
                verbose=args.verbose, # debug this function with visualize_pseudo_actions_rollouts.py
                offset=POSITIONAL_OFFSET,
                debug_dir='debug',
                icp_method=args.icp_method,
            )
            
            if not args.delta_actions:
                actions_for_demo = poses_to_absolute_actions(
                    poses=all_hand_poses,
                    gripper_actions=[traj['actions'][i][-1] for i in range(len(traj['actions']))],
                    control_freq = control_freq,
                    policy_freq = policy_freq,
                    smooth=args.smooth
                )
            else:
                actions_for_demo = poses_to_delta_actions(
                    poses=all_hand_poses,
                    gripper_actions=[traj['actions'][i][-1] for i in range(len(traj['actions']))],
                    smooth=False,
                    translation_scaling=80.0,
                    rotation_scaling=9.0,
                )
                
            if args.verbose or args.count_rollout_success:
                env_rollout.reset()
                env_rollout.reset_to(initial_state_list[j])
                success = roll_out(env_rollout, actions_for_demo, verbose=args.verbose or args.save_rollout_videos, file_name=f"roll_out_{ind}.mp4")
                
                if success:
                    n_success += 1
                total_n += 1
                
                result_str = f"demo_{ind}: {f'success' if success else 'failure'}"
                
                print(result_str)
                with open(results_file_path, "a") as results_txt:
                    results_txt.write(result_str + "\n")
            
            traj["actions"] = actions_for_demo
            
            exclude_cameras_from_obs(traj, ADDITIONAL_CAMERAS_FOR_POINT_CLOUD)
            # store transitions

            # IMPORTANT: keep name of group the same as source file, to make sure that filter keys are
            #            consistent as well
            ep_data_grp = data_grp.create_group(ep)
            ep_data_grp.create_dataset("actions", data=np.array(traj["actions"]))
            ep_data_grp.create_dataset("states", data=np.array(traj["states"]))
            ep_data_grp.create_dataset("rewards", data=np.array(traj["rewards"]))
            ep_data_grp.create_dataset("dones", data=np.array(traj["dones"]))
            # for k in traj["obs"]:
            #     if args.compress:
            #         ep_data_grp.create_dataset("obs/{}".format(k), data=np.array(traj["obs"][k]), compression="gzip")
            #     else:
            #         ep_data_grp.create_dataset("obs/{}".format(k), data=np.array(traj["obs"][k]))

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
    
    if args.count_rollout_success:
        success_rate = (n_success / total_n)*100 if total_n else 0
        summary_str = f"\nFinal Success Rate: {n_success}/{total_n} => {success_rate:.2f}%"
        print(summary_str)
        with open(results_file_path, "a") as results_txt:
            results_txt.write(summary_str + "\n")

    print(f"Done labeling dataset with pseudo-actions.\nSaved to: {args.output_hdf5_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Rewrite 'actions' in a single .hdf5 dataset with pseudo-actions (estimated from point clouds)."
    )

    parser.add_argument("--hdf5_path", type=str, required=True,
                        help="Path to a single .hdf5 dataset file")
    parser.add_argument("--output_hdf5_name", type=str, default=None,
                        help="Name of the psuedo-labeled .hdf5. file"
                             "If None, modifies in-place; else copies first.")
    parser.add_argument("--num_demos", type=int, default=None,
                        help="Number of demos to process (if None, do all).")
    parser.add_argument("--num_workers", type=int, default=16, # Maximum num_workers suitable for my machine
                        help="Number of workers for parallel saving")
    parser.add_argument("--delta_actions", action="store_true",
                        help="If set, use poses_to_delta_actions instead of absolute actions.")
    parser.add_argument("--smooth", action="store_true",
                        help="If set, attempts to smooth the resulting action path.")
    parser.add_argument("--verbose", action="store_true",
                        help="If set, visualize for debugging purposes.")
    parser.add_argument("--count_rollout_success", action="store_true",
                        help="If set, count the number of successful rollouts with pseudo actions.")
    parser.add_argument("--save_rollout_videos", action="store_true",
                        help="If set, save videos of rollouts with pseudo actions.")
    parser.add_argument("--debug_demo", type=int, default=None,
                        help="If not None, process the specified demo for debugging purposes.")
    parser.add_argument("--icp_method", type=str, default="multiscale", choices=["multiscale", "updown"],
                        help="ICP method used for pose estimation.")
    parser.add_argument("--compress", action='store_true',
                        help="Compress observations with gzip option in hdf5")

    args = parser.parse_args()

    label_dataset_with_pseudo_actions(args)


if __name__ == "__main__":
    main()