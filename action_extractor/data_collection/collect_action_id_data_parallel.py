'''
Built on collect_human_demonstrations.py
Collect MOVEMENT data for the action id task in 3D space
Given a 3d grid, collect trajectories between pairs of points
'''

'''
This script is used to collect data for pose estimation
Given a grid of 3D points covering the task action space, we collect trajectories between pairs of points
This gives us a mapping from images to gripper poses
'''

'''
TODO (David):
Collect pose-estimation/action-identification data for square_d0
How to get grid size:
- Go into square_d0 dataset (from mimicgen), look at ['data'][{demo}]['ob']['robot0_eef_pos'] for all demos
- Find the min and max values for x, y, z
- Right now, this script only does translation, no rotation.
- Think about how to do rotation.
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

def collect_trajectory(env_info, arm, env_configuration, start_point, end_point, render, tmp_directory, gripper):
    """
    Function to run a single trajectory collection.
    This can be passed into the parallel pool executor.
    """
    # Create a new environment instance for each process
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

    # Collect the trajectory between points
    collect_trajectory_between_points(env, arm, env_configuration, start_point, end_point, render=render)


def parallel_collect_trajectories(env_info, pairs, noisy_first_points, second_points, arm, env_configuration, render, tmp_directory, gripper, num_workers=4):
    """
    Function to collect multiple trajectories in parallel.
    """
    futures = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for i in range(len(pairs)):
            start_point = noisy_first_points[i]
            end_point = second_points[i]
            # Submit each trajectory collection as a separate task
            futures.append(
                executor.submit(collect_trajectory, env_info, arm, env_configuration, start_point, end_point, render, tmp_directory, gripper)
            )
        
        # Use tqdm to show progress as futures complete
        for future in tqdm(as_completed(futures), total=len(futures), desc="Collecting Demos in Parallel"):
            future.result()  # Blocking call to ensure each trajectory finishes


def collect_trajectory_between_points(env, arm, env_configuration, start_point, end_point, render=True):
    env.reset()  # Set to start_point

    # ID = 2 always corresponds to agentview
    if render:
        env.render()
        
    goal_ee_position = start_point
    
    i = 0
    gripper_opening = True  # Track the state of the gripper
    
    while i < 150:
        # Set active robot
        active_robot = env.robots[0] if env_configuration == "bimanual" else env.robots[arm == "left"]
        
        previous_ee_pose = active_robot.recent_ee_pose.last
        
        action_ee_rotation = np.array([0., 0., 0.])
        action_ee_gripper = np.array([-1.]).astype(float) if gripper_opening else np.array([1.]).astype(float)
        action_ee_position = (goal_ee_position - previous_ee_pose[:3]) / np.linalg.norm(goal_ee_position - previous_ee_pose[:3])
        action = np.concatenate([action_ee_position, action_ee_rotation, action_ee_gripper])

        env.step(action)
        
        current_ee_pose = active_robot.recent_ee_pose.last

        if np.allclose(current_ee_pose[:3], end_point, atol=0.009) and np.array_equal(goal_ee_position, end_point):
            # print(f"Position achieved after {i} iterations")
            
            def stop_in_place(env):
                action = np.array([0., 0., 0., 0., 0., 0., 0.])                
                env.step(action)
                
            def stop_in_place_and_move_gripper(env, gripper_opening):
                action = np.array([0., 0., 0., 0., 0., 0., 0.])
                action[-1] = -1. if gripper_opening else 1.
                env.step(action)
                
                if gripper_opening and np.allclose(abs(gripper_qpos), 0.04, atol=0.001):
                    gripper_opening = False
                elif not gripper_opening and np.allclose(abs(gripper_qpos), 0, atol=0.005):
                    gripper_opening = True
                    
                return gripper_opening
            
            # Stop in place for 10 steps
            for _ in range(10):
                stop_in_place(env)
                if render:
                    env.render()
                    
            # Move gripper while eef stays in place
            times_flipped = 0
            while times_flipped < 3:
                new_gripper_opening = stop_in_place_and_move_gripper(env, gripper_opening)
                if new_gripper_opening != gripper_opening:
                    times_flipped += 1
                    gripper_opening = new_gripper_opening
                if render:
                    env.render() 
            
            break
        
        elif np.allclose(previous_ee_pose[:3], current_ee_pose[:3], atol=1e-4) and i != 0:
            if np.array_equal(goal_ee_position, start_point):
                goal_ee_position = end_point
                second_trajectory_start_index = i
                # print(f"Start position can't be reached; converged to nearest start point after {i} iterations")
            elif np.array_equal(goal_ee_position, end_point) and i > second_trajectory_start_index + 20:
                # print(f"Goal position can't be reached; position converged after {i} iterations")
                break
        elif np.allclose(current_ee_pose[:3], start_point[:3], atol=0.01) and np.array_equal(goal_ee_position, start_point):
            goal_ee_position = end_point
            second_trajectory_start_index = i
            # print(f'Start position achieved after {i} iterations')

        # Access gripper joint positions
        gripper_qpos = env.sim.data.qpos[active_robot._ref_gripper_joint_pos_indexes]

        # Switch gripper state based on its position
        if gripper_opening and np.allclose(abs(gripper_qpos), 0.04, atol=0.001):
            gripper_opening = False
        elif not gripper_opening and np.allclose(abs(gripper_qpos), 0, atol=0.005):
            gripper_opening = True

        i += 1
        
        if render:
            env.render()
            
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
            
            env_name = None  # Define env_name inside the loop

            for state_file in sorted(glob(state_paths)):
                dic = np.load(state_file, allow_pickle=True)
                env_name = str(dic["env"])

                states.extend(dic["states"])
                for ai in dic["action_infos"]:
                    actions.append(ai["actions"])

            if len(states) == 0:
                continue

            # Save every episode
            del states[-1]
            assert len(states) == len(actions)

            # Create a new group for the current demonstration
            ep_data_grp = grp.create_group("demo_{}".format(num_eps))
            num_eps += 1

            # Store model xml as an attribute
            xml_path = os.path.join(directory, ep_directory, "model.xml")
            with open(xml_path, "r") as f:
                xml_str = f.read()
            ep_data_grp.attrs["model_file"] = xml_str

            # Write datasets for states and actions
            ep_data_grp.create_dataset("states", data=np.array(states))
            ep_data_grp.create_dataset("actions", data=np.array(actions))

            # If env_name is retrieved, update the environment attribute
            if env_name is not None:
                grp.attrs["env"] = env_name


if __name__ == "__main__":
    
    np.random.seed(2)
    random.seed(2)
    
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=str, default="/home/yilong/Documents/ae_data/random_processing/raw")
    parser.add_argument("--file_name", type=str, default="random.hdf5")
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
    parser.add_argument("--num_workers", type=int, default=128, help="Number of parallel workers")
    args = parser.parse_args()

    # Get controller config
    controller_config = load_controller_config(default_controller=args.controller)
    config = {
        "env_name": args.environment,
        "robots": args.robots,
        "controller_configs": controller_config,
    }

    # Create grid coordinates
    # General scale
    grid_length = 0.01
    x_min, x_max = -0.4, 0.2
    y_min, y_max = -0.4, 0.4
    z_min, z_max = 0.82, 1.2
    # 0.4 -> 0.39 to ensure visibility in agentview/sideagentview camera?
    
    ############################################################
    # Scale for lift
    # grid_length = 0.005
    # x_min, x_max = -0.12, 0.10
    # y_min, y_max = -0.02, 0.02
    # z_min, z_max = 0.81, 1.05
    
    ############################################################

    x_ = np.linspace(x_min, x_max, int((x_max - x_min) // grid_length))
    y_ = np.linspace(y_min, y_max, int((y_max - y_min) // grid_length))
    z_ = np.linspace(z_min, z_max, int((z_max - z_min) // grid_length))
    grid_coordinates = np.vstack(np.meshgrid(x_, y_, z_)).reshape(3, -1).T
    print(f"Grid contains {grid_coordinates.shape[0]} 3D points")

    n = grid_coordinates.shape[0]

    # Randomly generate unique pairs in place without creating all combinations
    selected_pairs = set()
    while len(selected_pairs) < args.num_trajectories:
        i, j = random.sample(range(n), 2)
        pair = tuple(sorted((i, j)))  # Ensure uniqueness of (i, j) and (j, i)
        selected_pairs.add(pair)

    # Convert set to list of pairs
    pairs = list(selected_pairs)

    first_points = np.array([grid_coordinates[i] for i, _ in pairs])
    noisy_first_points = first_points + np.random.normal(loc=0, scale=grid_length, size=first_points.shape)
    second_points = np.array([grid_coordinates[j] for _, j in pairs])
    
    tmp_directory = "/tmp/{}".format(str(time.time()).replace(".", "_"))
    os.makedirs(tmp_directory, exist_ok=True)

    parallel_collect_trajectories(config, pairs, noisy_first_points, second_points, args.arm, args.config, render=args.render, tmp_directory=tmp_directory, gripper=args.gripper, num_workers=args.num_workers)
    
    # Gather demonstrations into HDF5
    new_dir = os.path.join(args.directory, "{}_{}".format(*str(time.time()).split(".")))
    os.makedirs(new_dir, exist_ok=True)
    gather_demonstrations_as_hdf5(tmp_directory, new_dir, json.dumps(config), args.file_name)
