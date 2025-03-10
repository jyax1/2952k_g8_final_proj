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

# Import robomimic/robosuite utilities.
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils

from xml.etree import ElementTree as ET
from action_extractor.utils.robosuite_data_processing_utils import *

def convert_dataset(args):
    # Load the original environment metadata from the dataset.
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
    # Update the robot type in the environment kwargs.
    env_meta['env_kwargs']['gripper_types'] = 'PandaGripper'
    env_meta['env_kwargs']['robots'] = args.new_robot

    # Create the new environment for data processing.
    # Even though we don’t need images for the conversion, we set a camera name
    # so that we can record the 'frontview_image' if --verbose is enabled.
    new_env = EnvUtils.create_env_for_data_processing(
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

    # Loop over each demonstration.
    for demo in demos:
        demo_grp_in = f_in["data"][demo]
        old_states = demo_grp_in["states"][()]  # shape (T, state_dim)
        actions = demo_grp_in["actions"][()]      # shape (T, action_dim)
        T = actions.shape[0]
        new_states_list = []

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
        
        # Update the state with the new robot configuration.
        initial_state = convert_robot_in_state(initial_state, new_env)

        new_env.reset()  # Ensure the environment is ready
        obs = new_env.reset_to(initial_state)
        # If reset_to() returns None, then get observation from get_obs().
        if obs is None:
            obs = new_env.get_obs()

        if args.verbose:
            # Save the initial frame.
            frames.append(obs['frontview_image'])

        # Record the initial state from the new environment.
        new_state = new_env.env.sim.get_state().flatten()
        new_states_list.append(new_state)

        # Replay actions.
        for t in range(T):
            obs, _, done, _ = new_env.step(actions[t])
            new_state = new_env.env.sim.get_state().flatten()
            new_states_list.append(new_state)
            if args.verbose:
                frames.append(obs['frontview_image'])
            # Optionally, you could break out of the loop if done is True.

        success = new_env.is_success()["task"]
        print(f"success: {success} for demo {demo}")
        new_states = np.stack(new_states_list)  # shape (T+1, state_dim)
        num_samples = actions.shape[0]
        total_samples += num_samples

        # Write the demo to the output dataset.
        demo_grp_out = data_grp.create_group(demo)
        demo_grp_out.create_dataset("actions", data=actions)
        demo_grp_out.create_dataset("states", data=new_states)
        if "model_file" in demo_grp_in.attrs:
            demo_grp_out.attrs["model_file"] = demo_grp_in.attrs["model_file"]
        demo_grp_out.attrs["num_samples"] = num_samples

        print(f"Processed {demo}: {num_samples} transitions.")

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
    print(f"Wrote {len(demos)} demos with total {total_samples} samples to {output_path}")

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