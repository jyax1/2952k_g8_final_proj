#!/usr/bin/env python3
"""
Script to convert a dataset to a new robot specification.
Given an input HDF5 dataset (with groups demo_0, demo_1, …),
this script updates the environment metadata to use a new robot,
then replays each demonstration by resetting the environment to the
initial state and stepping through the stored actions. The new simulation
states (flattened) are recorded in an output HDF5 file with the same structure.
If --verbose is set, videos of the frontview observations are saved to debug/robot_conversion.
 
The structure of the output dataset is:
    data (group)
        total (attribute): total number of state-action samples
        env_args (attribute): JSON string of environment metadata
        demo_i (group):
            num_samples (attribute): number of transitions in this demo
            model_file (attribute): MJCF XML model string
            states (dataset): new simulation states (flattened) for each timestep
            actions (dataset): copied from the original dataset
"""

import os
import json
import h5py
import argparse
import numpy as np
from copy import deepcopy
import imageio

from scipy.spatial.transform import Rotation as R

# Import robomimic/robosuite utilities.
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils

from action_extractor.utils.angles_utils import *
from action_extractor.utils.rollout_utils import *

from xml.etree import ElementTree as ET
from action_extractor.utils.robosuite_data_processing_utils import *

def get_abs_action(demo_grp_in, index, policy_freq=20):
    repetition = 20 // policy_freq
    goal_pos_list = [pos for pos in demo_grp_in['obs']['robot0_eef_pos'] for _ in range(repetition)]
    goal_ori_quat_list = [quat for quat in demo_grp_in['obs']['robot0_eef_quat'] for _ in range(repetition)]
    gripper_action_list = [action[-1] for action in demo_grp_in['actions'] for _ in range(repetition)]
    
    goal_pos = goal_pos_list[index]
    goal_ori_quat = goal_ori_quat_list[index] # rotated by 90 degrees around the z axis counter clockwise
    goal_ori_quat = (R.from_euler('z', 90, degrees=True) * R.from_quat(goal_ori_quat)).as_quat() # correct rotation
    
    goal_ori_axisangle = quat2axisangle(goal_ori_quat)
    goal_pose_axisangle = np.concatenate((goal_pos, goal_ori_axisangle), axis=0)
    
    gripper_action = gripper_action_list[index]
    return np.concatenate((goal_pose_axisangle, np.array([gripper_action])), axis=0)

def save_state_mask(diff_array: np.ndarray, filename: str = "robot_state_mask.npy", directory: str = "action_extractor/utils", threshold: float = 1e-4):
    """
    Process the difference array into a binary mask (zeros and ones) and save it as an .npy file.
    
    Args:
        diff_array (np.ndarray): Difference array (initial_state['states'] - new_state).
        filename (str): Filename for the saved mask.
        directory (str): Directory where the mask file will be saved.
        threshold (float): Threshold to consider a change significant.
        
    Returns:
        np.ndarray: The binary mask array.
    """
    # Create the binary mask: 1 if the absolute difference is greater than threshold, else 0.
    mask = (np.abs(diff_array) > threshold).astype(np.uint8)
    
    # Ensure the target directory exists.
    os.makedirs(directory, exist_ok=True)
    
    # Build the full path.
    filepath = os.path.join(directory, filename)
    
    # Save the mask to an .npy file.
    np.save(filepath, mask)
    print(f"Mask saved to {filepath}")
    
    return mask

def convert_dataset(args):
    # Load the original environment metadata from the dataset.
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
    # Update the robot type in the environment kwargs.
    env_meta['env_kwargs']['gripper_types'] = 'PandaGripper'
    env_meta['env_kwargs']['robots'] = args.new_robot

    # Create the new environment for data processing.
    # Even though we don’t need images for the conversion, we set a camera name
    # so that we can record the 'frontview_image' if --verbose is enabled.
    env_meta["env_kwargs"]["controller_configs"]["control_delta"] = False
    new_env = EnvUtils.create_env_for_data_processing(
        env_meta=env_meta,
        camera_names=['frontview'],
        camera_height=args.camera_height,
        camera_width=args.camera_width,
        reward_shaping=False,
    )
    env_meta["env_kwargs"]["controller_configs"]["control_delta"] = False
    initialization_env = EnvUtils.create_env_for_data_processing(
        env_meta=env_meta,
        camera_names=['frontview'],
        camera_height=args.camera_height,
        camera_width=args.camera_width,
        reward_shaping=False,
    )
    
    # If verbose is set, prepare the debug directory.
    if args.verbose:
        debug_dir = os.path.join("debug", "robot_conversion")
        os.makedirs(debug_dir, exist_ok=True)

    # Open the input dataset.
    f_in = h5py.File(args.dataset, "r")
    demos = list(f_in["data"].keys())
    # Sort demos by numeric index (assuming names like demo_0, demo_1, etc.)
    demos = sorted(demos, key=lambda x: int(x.split("_")[-1]))

    # Create the output file in the same directory as the input.
    output_path = os.path.join(os.path.dirname(args.dataset), args.output_name)
    f_out = h5py.File(output_path, "w")
    data_grp = f_out.create_group("data")

    total_samples = 0
    
    n_success = 0
    
    # <<-- ADDED: Initialize the conversion status dictionary for each demo.
    conversion_status = {}
    # -->> 

    # Loop over each demonstration.
    if args.num_demos is not None:
        demos = demos[:args.num_demos]
    for demo in demos:
        demo_grp_in = f_in["data"][demo]
        old_states = demo_grp_in["states"][()]  # shape (T, state_dim)
        actions = copy.deepcopy(demo_grp_in["actions"][()])      # shape (T, action_dim)
        T = actions.shape[0]

        # Prepare video recording if verbose.
        if args.verbose:
            frames = []

        # Reset the environment using the first state.
        initial_state = {"states": old_states[0]}
        xml_str = demo_grp_in.attrs["model_file"]
        xml_str = replace_all_lights(xml_str)
        xml_str = recolor_robot(xml_str)
        xml_str = recolor_gripper(xml_str)
        initial_state["model"] = xml_str
        
        initialization_env.reset()
        
        # Update the state with the new robot configuration.
        initial_state = convert_robot_in_state(initial_state, initialization_env)
        new_model_file = initial_state['model']
        obs = initialization_env.reset_to(initial_state)
        
        if args.verbose:
            # Save the initial frame.
            frames.append(obs['frontview_image'])
            
        current_pose_quat = initialization_env.env.robots[0].recent_ee_pose.last

        # initial_goal_pos = current_pos # preserve initial gripper position
        initial_goal_pos = demo_grp_in['obs']['robot0_eef_pos'][0]
        # initial_goal_ori_quat = initialization_env.env.sim.data.get_body_xquat("gripper0_right_gripper") # preserve initial gripper orientation
        initial_goal_ori_quat = demo_grp_in['obs']['robot0_eef_quat'][0] # rotated by 90 degrees around the z axis counter clockwise
        initial_goal_ori_quat = (R.from_euler('z', 90, degrees=True) * R.from_quat(initial_goal_ori_quat)).as_quat() # correct rotation
        
        initial_goal_pose_quat = np.concatenate((initial_goal_pos, initial_goal_ori_quat), axis=0)
        
        initial_goal_ori_axisangle = quat2axisangle(initial_goal_ori_quat)
        initial_goal_pose_axisangle = np.concatenate((initial_goal_pos, initial_goal_ori_axisangle), axis=0)
        abs_action = np.concatenate((initial_goal_pose_axisangle, np.array([-1])), axis=0)
        
        stationary_count = 0
        stationary_threshold = 10  # number of iterations to consider as "stationary"
        prev_pose_quat = None

        while not np.allclose(current_pose_quat, initial_goal_pose_quat, atol=0.00005):
            # Compute delta command in the axis-angle space for the orientation
            obs, _, _, _ = initialization_env.step(abs_action)
            
            if args.verbose:
                frames.append(obs['frontview_image'])
            
            # Check if current_pose_quat has changed significantly from the previous iteration
            if prev_pose_quat is not None:
                # Adjust the tolerance (e.g., 1e-6) as needed based on your precision requirements
                if np.allclose(current_pose_quat, prev_pose_quat, atol=1e-6):
                    stationary_count += 1
                else:
                    stationary_count = 0  # reset counter if a significant change is detected
            
            # Update the previous pose for the next iteration
            prev_pose_quat = current_pose_quat.copy()
            
            # Break if the pose has remained stationary for a number of iterations
            if stationary_count >= stationary_threshold:
                break
            
            # Update the current command
            current_pose_quat = initialization_env.env.robots[0].recent_ee_pose.last
            
        new_state = initialization_env.env.sim.get_state().flatten()
        
        new_env.reset()
        
        success = False
        
        policy_freq = 20

        max_attempts = 4
        while not success and max_attempts > 0:
            max_attempts -= 1
        # Replay actions.
            actions = np.array([action for action in demo_grp_in["actions"][()] for _ in range(20 // policy_freq)])
            obs = new_env.reset_to(initial_state)
            new_states_list = []
            # new_states_list.append(initial_state['states'])
            if args.verbose:
                    frames.append(obs['frontview_image'])
            policy_T = T * (20 // policy_freq)
            for t in range(policy_T):
                new_state = new_env.env.sim.get_state().flatten()
                new_states_list.append(new_state)
                obs, _, _, _ = new_env.step(get_abs_action(demo_grp_in, t, policy_freq))
                if args.verbose:
                    frames.append(obs['frontview_image'])

            success = new_env.is_success()["task"]
            if success:
                print(f"success for demo {demo} with policy frequency {policy_freq}")
            else:
                policy_freq = change_policy_freq(policy_freq, random_choice=False) 
                if max_attempts > 0:
                    print(f"retrying with policy frequency {policy_freq} for demo {demo}")
            
        if not success:
            print(f"Failed to convert demo {demo} after multiple attempts.")
        else:
            n_success += 1

        # <<-- ADDED: Record the conversion status for this demo.
        conversion_status[demo] = bool(success)
        # -->> 

        new_states = np.stack(new_states_list)  # shape (T+1, state_dim)
        num_samples = actions.shape[0]
        total_samples += num_samples

        # Write the demo to the output dataset.
        demo_grp_out = data_grp.create_group(demo)
        demo_grp_out.create_dataset("actions", data=actions)
        demo_grp_out.create_dataset("states", data=new_states)
        demo_grp_out.attrs["model_file"] = new_model_file
        demo_grp_out.attrs["num_samples"] = num_samples

        print(f"Processed {demo}: {num_samples} transitions.\n")

        # If verbose, save the video.
        if args.verbose:
            video_path = os.path.join(debug_dir, f"{demo}.mp4")
            with imageio.get_writer(video_path, fps=20) as writer:
                for frame in frames:
                    writer.append_data(frame)
            print(f"Saved video for {demo} to {video_path}")

    # Set global attributes.
    data_grp.attrs["total"] = total_samples
    data_grp.attrs["env_args"] = json.dumps(new_env.serialize(), indent=4)
    
    # <<-- ADDED: Save the conversion status dictionary as an attribute.
    data_grp.attrs["conversion_status"] = json.dumps(conversion_status)
    # -->> 
    
    print(f"Wrote {len(demos)} demos with total {total_samples} samples to {output_path}")
    
    print(f"Success rate: {n_success / len(demos) * 100:.2f}%")

    f_in.close()
    f_out.close()

def main():
    parser = argparse.ArgumentParser(
        description="Convert a dataset to a new robot specification by re-running the demos."
    )
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to input HDF5 dataset")
    parser.add_argument("--output_name", type=str, required=True,
                        help="Name of output HDF5 dataset")
    parser.add_argument("--num_demos", type=int, default=None,
                        help="Number of demos to process (default: all)")
    parser.add_argument("--new_robot", type=str, required=True,
                        help="New robot name to use (will override env_meta['env_kwargs']['robots'])")
    parser.add_argument("--camera_height", type=int, default=480,
                        help="Camera height (required by env creation)")
    parser.add_argument("--camera_width", type=int, default=640,
                        help="Camera width (required by env creation)")
    parser.add_argument("--verbose", action="store_true",
                        help="If set, save a video of frontview observations for each demo to debug/robot_conversion")
    args = parser.parse_args()
    convert_dataset(args)

if __name__ == "__main__":
    main()