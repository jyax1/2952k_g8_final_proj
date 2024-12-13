'''
Modeled after collecta_human_demonstrations.py and collect_action_id_data_parallel.py
'''

import argparse
import datetime
import json
import os
import shutil
import time
from glob import glob

from concurrent.futures import ProcessPoolExecutor, as_completed
import random

import itertools

from tqdm import tqdm

import h5py
import numpy as np

import robosuite as suite
import robosuite.macros as macros
from robosuite import load_controller_config
from robosuite.utils.input_utils import input2action
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper

def collect_trajectory(env_info, arm, env_configuration, render, tmp_directory, gripper):
    """
    Function to run a single trajectory collection.
    This can be passed into the parallel pool executor.
    """
    # Create a new environment instance for each process
    

    # Collect the trajectory between points
    collect_lift_trajectory_without_rotation(env_info, gripper, tmp_directory, arm, env_configuration, render=render)


def parallel_collect_lift_trajectories(env_info, num_trajectories, arm, env_configuration, render, tmp_directory, gripper, num_workers=4):
    """
    Function to collect multiple trajectories in parallel.
    """
    futures = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for _ in range(num_trajectories):
            futures.append(
                executor.submit(collect_trajectory, env_info, arm, env_configuration, render, tmp_directory, gripper)
            )
        
        # Use tqdm to show progress as futures complete
        for future in tqdm(as_completed(futures), total=len(futures), desc="Collecting Demos in Parallel"):
            future.result()  # Blocking call to ensure each trajectory finishes


def collect_lift_trajectory_without_rotation(env_info, gripper, tmp_directory, arm, env_configuration, render=True):
    env = suite.make(
        **env_info,
        gripper_types = gripper,
        has_renderer=render,  # Renderer is generally not used in parallel processes
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
    )
    env = VisualizationWrapper(env)
    env = DataCollectionWrapper(env, tmp_directory)
    
    env.reset()

    goal_position = env.sim.data.body_xpos[env.cube_body_id]
    print(goal_position)
    gripper_action = np.array([-1.]).astype(float)

    if render:
        env.render()
        
    stage = 0 # translation stage
    
    task_completion_hold_count = -1

    i = 0
    while i < 100:
        # Set active robot
        active_robot = env.robots[0] if env_configuration == "bimanual" else env.robots[arm == "left"]
        previous_eef_pose = active_robot.recent_ee_pose.last
        previous_eef_position = previous_eef_pose[:3]
        
        if stage == 0:
            action_translation = (goal_position - previous_eef_position) / np.linalg.norm(goal_position - previous_eef_position)
        
        action_rotation = np.array([0., 0., 0.])
        action = np.concatenate([action_translation, action_rotation, gripper_action])
        
        env.step(action)
        if render:
            env.render()
        
        if np.allclose(previous_eef_position, goal_position, atol=0.05) and stage == 0:
            action_translation = np.array([0., 0., 0.])
            gripper_action = np.array([1.]).astype(float)
            stage = 1 # grasping stage
    
        current_robot_pose = active_robot.recent_ee_pose.last
        if np.allclose(previous_eef_pose, current_robot_pose, atol=0.001) and stage == 1: # end of grasping stage
            action_translation = np.array([0., 0., 1.])
            stage = 2 # lift stage
            
        if task_completion_hold_count == 0:
            break

        # state machine to check for having a success for 10 consecutive timesteps
        if env._check_success():
            if task_completion_hold_count > 0:
                task_completion_hold_count -= 1  # latched state, decrement count
            else:
                task_completion_hold_count = 3  # reset count on first success timestep
        else:
            task_completion_hold_count = -1  # null the counter if there's no success
            
        i += 1
        
    env.close()



def gather_demonstrations_as_hdf5(directory, out_dir, env_info, file_name):
    """
    Gathers the demonstrations saved in @directory into a
    single hdf5 file.

    The structure of the hdf5 file is as follows.

    data (group)
        date (attribute) - date of collection
        time (attribute) - time of collection
        repository_version (attribute) - repository version used during collection
        env (attribute) - environment name on which demos were collected

        demo1 (group) - every demonstration has a group
            model_file (attribute) - model xml string for demonstration
            states (dataset) - flattened mujoco states
            actions (dataset) - actions applied during demonstration

        demo2 (group)
        ...

    Args:
        directory (str): Path to the directory containing raw demonstrations.
        out_dir (str): Path to where to store the hdf5 file.
        env_info (str): JSON-encoded string containing environment information,
            including controller and robot info
    """

    hdf5_path = os.path.join(out_dir, file_name)
    
    # Open the HDF5 file in append mode ('a')
    with h5py.File(hdf5_path, "a") as f:
        # Check if the "data" group exists, if not, create it
        if "data" not in f:
            grp = f.create_group("data")

            # write dataset attributes (metadata)
            now = datetime.datetime.now()
            grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
            grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
            grp.attrs["repository_version"] = suite.__version__
            grp.attrs["env_info"] = env_info
        else:
            grp = f["data"]

        # Find the next available demonstration number
        num_eps = len(grp.keys())

        for ep_directory in os.listdir(directory):
            state_paths = os.path.join(directory, ep_directory, "state_*.npz")
            states = []
            actions = []
            success = False
            
            env_name = None  # Define env_name inside the loop

            for state_file in sorted(glob(state_paths)):
                dic = np.load(state_file, allow_pickle=True)
                env_name = str(dic["env"])

                states.extend(dic["states"])
                for ai in dic["action_infos"]:
                    actions.append(ai["actions"])
                success = success or dic["successful"]

            if len(states) == 0:
                continue
            
            if success:
                print("Demonstration is successful and has been saved")
                # Delete the last state. This is because when the DataCollector wrapper
                # recorded the states and actions, the states were recorded AFTER playing that action,
                # so we end up with an extra state at the end.
                del states[-1]
                assert len(states) == len(actions)

                ep_data_grp = grp.create_group("demo_{}".format(num_eps))
                num_eps += 1

                # store model xml as an attribute
                xml_path = os.path.join(directory, ep_directory, "model.xml")
                with open(xml_path, "r") as f:
                    xml_str = f.read()
                ep_data_grp.attrs["model_file"] = xml_str

                # write datasets for states and actions
                ep_data_grp.create_dataset("states", data=np.array(states))
                ep_data_grp.create_dataset("actions", data=np.array(actions))
            else:
                print("Demonstration is unsuccessful and has NOT been saved")
                
            now = datetime.datetime.now()
            grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
            grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
            grp.attrs["repository_version"] = suite.__version__
            grp.attrs["env"] = env_name
            grp.attrs["env_info"] = env_info

            f.close()


if __name__ == "__main__":
    
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=str, default="/home/yilong/Documents/policy_data/lift/raw")
    parser.add_argument("--file_name", type=str, default="lift.hdf5")
    parser.add_argument("--environment", type=str, default="Lift")
    parser.add_argument("--robots", nargs="+", type=str, default="Panda", choices=['Panda', 'Sawyer', 'IIWA', 'Jaco', 'Kinova3', 'UR5e', 'Baxter'], help="Which robot(s) to use in the env")
    parser.add_argument("--config", type=str, default="single-arm-opposed", help="Specified environment configuration if necessary")
    parser.add_argument("--arm", type=str, default="right", help="Which arm to control (eg bimanual) 'right' or 'left'")
    parser.add_argument("--camera", type=str, default="frontview", help="Which camera to use for collecting demos")
    parser.add_argument("--controller", type=str, default="OSC_POSE", help="Choice of controller. Can be 'IK_POSE' or 'OSC_POSE'")
    parser.add_argument("--gripper", type=str, default="PandaGripper")
    parser.add_argument("--pos-sensitivity", type=float, default=1.0, help="How much to scale position user inputs")
    parser.add_argument("--rot-sensitivity", type=float, default=1.0, help="How much to scale rotation user inputs")
    parser.add_argument("--num_trajectories", type=int, default=1000, help="Number of trajectories to collect")
    parser.add_argument("--render", action='store_true')
    parser.add_argument("--num_workers", type=int, default=1, help="Number of parallel workers")
    args = parser.parse_args()

    # Get controller config
    controller_config = load_controller_config(default_controller=args.controller)
    config = {
        "env_name": args.environment,
        "robots": args.robots,
        "controller_configs": controller_config,
    }
    
    tmp_directory = "/tmp/{}".format(str(time.time()).replace(".", "_"))
    os.makedirs(tmp_directory, exist_ok=True)

    parallel_collect_lift_trajectories(config, args.num_trajectories, args.arm, args.config, render=args.render, tmp_directory=tmp_directory, gripper=args.gripper, num_workers=args.num_workers)
    
    # Gather demonstrations into HDF5
    new_dir = os.path.join(args.directory, "{}_{}".format(*str(time.time()).split(".")))
    os.makedirs(new_dir, exist_ok=True)
    gather_demonstrations_as_hdf5(tmp_directory, new_dir, json.dumps(config), args.file_name)
