#!/usr/bin/env python3
"""
Script to convert a dataset to a new robot specification.
Given an input HDF5 dataset (with groups demo_0, demo_1, …),
this script updates the environment metadata to use a new robot,
then replays each demonstration by resetting the environment to the
initial state and stepping through the stored actions. The new simulation
states (flattened) are recorded in an output HDF5 file with the same structure.

The structure of the output dataset is:
    data (group)
        total (attribute): total number of state-action samples
        env_args (attribute): json string of environment metadata
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

# Import robomimic/robosuite utilities.
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils

from xml.etree import ElementTree as ET

from action_extractor.utils.robosuite_data_processing_utils import convert_robot_in_state

# You may need to import additional functions if needed, for instance:
# from action_extractor.utils.robosuite_data_processing_utils import recolor_robot, recolor_gripper
# (if you want to update the XML string)

def convert_dataset(args):
    # Load the original environment metadata from the dataset.
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
    # Update the robot type in the environment kwargs.
    env_meta['env_kwargs']['gripper_types'] = 'PandaGripper'
    env_meta['env_kwargs']['robots'] = args.new_robot

    # Create the new environment for data processing.
    # We do not need image processing here, so we can set camera_names to an empty list.
    new_env = EnvUtils.create_env_for_data_processing(
        env_meta=env_meta,
        camera_names=[],  # not using cameras here
        camera_height=args.camera_height,
        camera_width=args.camera_width,
        reward_shaping=False,
    )

    # Open the input dataset.
    f_in = h5py.File(args.dataset, "r")
    demos = list(f_in["data"].keys())
    # Sort demos by their numeric index (assuming names like demo_0, demo_1, ...)
    demos = sorted(demos, key=lambda x: int(x.split("_")[-1]))

    # Create the output file in the same directory as input.
    output_path = os.path.join(os.path.dirname(args.dataset), args.output_name)
    f_out = h5py.File(output_path, "w")
    data_grp = f_out.create_group("data")

    total_samples = 0

    # Loop over each demonstration.
    for demo in demos:
        demo_grp_in = f_in["data"][demo]
        # Load actions and states from the input demo.
        old_states = demo_grp_in["states"][()]  # assume shape (T, state_dim)
        actions = demo_grp_in["actions"][()]      # assume shape (T, action_dim)

        T = actions.shape[0]
        # Prepare a list for new states.
        new_states_list = []

        # Reset the environment using the first state from the demo.
        # The initial state is stored in old_states[0]. In our case, we assume that
        # the state is a flattened MuJoCo state that can be fed to env.reset_to().
        initial_state = {"states": old_states[0]}
        # (If the dataset also stores a model XML in attributes, you can update it here.)
        # For instance:
        if "model_file" in demo_grp_in.attrs:
            initial_xml = demo_grp_in.attrs["model_file"]
            # Optionally update xml string (e.g., recolor_robot) here.
            initial_state["model"] = initial_xml
            
        initial_state = convert_robot_in_state(initial_state, new_env)

        new_env.reset()  # make sure the environment is ready
        new_env.reset_to(initial_state)
        # Record the initial state from the new environment.
        # We assume new_env.get_state() returns a flat state vector.
        new_state = new_env.get_state()  # Adjust if needed, e.g., new_env.sim.get_state()
        new_states_list.append(new_state)

        # Now, iterate over the actions to update the state.
        for t in range(T):
            # Execute action t in the new environment.
            # If env.step returns (obs, reward, done, info) then we discard everything except the state.
            _, _, done, _ = new_env.step(actions[t])
            new_state = new_env.get_state()
            new_states_list.append(new_state)
            # Optionally, you could check for 'done' and break early if desired.

        new_states = np.array(new_states_list)  # shape (T+1, state_dim)
        num_samples = actions.shape[0]
        total_samples += num_samples

        # Create a new group for this demo in the output file.
        demo_grp_out = data_grp.create_group(demo)
        demo_grp_out.create_dataset("actions", data=actions)
        demo_grp_out.create_dataset("states", data=new_states)
        # Copy attributes such as model_file and num_samples.
        if "model_file" in demo_grp_in.attrs:
            demo_grp_out.attrs["model_file"] = demo_grp_in.attrs["model_file"]
        demo_grp_out.attrs["num_samples"] = num_samples

        print(f"Processed {demo}: {num_samples} transitions.")

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
    parser.add_argument("--camera_height", type=int, default=84,
                        help="Camera height (if applicable; not used here but required by env creation)")
    parser.add_argument("--camera_width", type=int, default=84,
                        help="Camera width (if applicable; not used here but required by env creation)")
    # You can add other flags (e.g., number of demos to process) as needed.
    parser.add_argument("--n", type=int, default=None,
                        help="Stop after n trajectories are processed (optional)")

    args = parser.parse_args()

    convert_dataset(args)

if __name__ == "__main__":
    main()
