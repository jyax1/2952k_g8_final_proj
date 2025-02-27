'''
action_extractor/point_cloud/robosuite/visualize_pseudo_actions_rollouts.py

Given a dataset of video demonstrations, this script estimates
pseudo actions from the video demonstrations and rolls out the pseudo actions for visualization.

For a demo, run the script with default arguments:
python visualize_pseudo_actions_rollouts.py
'''

import os
import numpy as np
import imageio
from glob import glob
from tqdm import tqdm
import h5py

import robomimic.utils.obs_utils as ObsUtils
from robomimic.utils.file_utils import get_env_metadata_from_dataset
from robomimic.utils.env_utils import create_env_from_metadata

from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder

from action_extractor.utils.angles_utils import *
from action_extractor.utils.poses_to_actions import *
from action_extractor.point_cloud.action_identifier_point_cloud import *
from action_extractor.utils.rollout_video_utils import *
from action_extractor.utils.poses_utils import *
from action_extractor.utils.rollout_utils import *
from action_extractor.utils.rollout_debug_utils import *
from action_extractor.point_cloud.generate_point_clouds_from_dataset import *
from action_extractor.point_cloud.config import POLICY_FREQS, POSITIONAL_OFFSET
            
def roll_out_and_save_video(env, 
             actions_for_demo, 
             file_path, 
             policy_freq,
             verbose=False, 
             # for verbose usage:
             point_clouds_points=None, 
             point_clouds_colors=None, 
             hand_mesh=None, 
             output_dir=None,
             all_hand_poses=None,
             demo_id=None) -> None:
    '''
    Given a list of actions, an environment, and a policy frequency, roll out the actions in the environment and save the video.
    '''
    pos_array, quat_array = [], []             
    env.file_path = file_path
    env.step_count = 0
    pos_array.append(env.env.env._eef_xpos.astype(np.float32))
    quat_array.append(env.env.env._eef_xquat.astype(np.float32))
    for (i, action) in enumerate(actions_for_demo):
        env.step(action)
        if i % (env.env.env.control_freq // policy_freq) == 0:
            pos_array.append(env.env.env._eef_xpos.astype(np.float32))
            quat_array.append(env.env.env._eef_xquat.astype(np.float32))
    env.video_recoder.stop()
    env.file_path = None
    
    if verbose:
        effect_poses = get_4x4_poses(pos_array, quat_array)
        render_model_on_pointclouds_two_colors(
            point_clouds_points,
            point_clouds_colors,
            all_hand_poses, # Red
            effect_poses,   # Blue
            model=load_model_as_pointcloud(hand_mesh, model_in_mm=True),
            output_dir=os.path.join(output_dir, f"rendered_models_{demo_id}"),
            verbose=verbose
        )

def infer_actions_and_rollout(root_h,
                              demo,
                              env_camera0,
                              env_camera1,
                              point_clouds_points,
                              point_clouds_colors,
                              hand_mesh,
                              upper_right_video_path,
                              lower_right_video_path,
                              output_dir,
                              demo_id,
                              policy_freq=20,
                              smooth=False,
                              verbose=True,
                              num_samples=100,
                              absolute_actions=True,
                              ground_truth=False,
                              offset=[0,0,0],
                              icp_method="multiscale") -> None:
    '''
    infer actions from poses and roll out the actions in the environment
    '''
    initial_state = root_h["data"][demo]["states"][0]
    obs_group = root_h["data"][demo]["obs"]
    env_camera0.reset()
    env_camera0.reset_to({"states": initial_state})
    
    if ground_truth:
        all_hand_poses = load_ground_truth_poses(obs_group)
        if verbose:
            render_model_on_pointclouds(
                point_clouds_points,
                point_clouds_colors,
                all_hand_poses,
                model=load_model_as_pointcloud(hand_mesh,
                                            model_in_mm=True),
                output_dir=os.path.join(output_dir, f"rendered_frames_{demo_id}"),
                verbose=verbose
            )
    else:
        all_hand_poses = get_poses_from_pointclouds(
            point_clouds_points,
            point_clouds_colors,
            hand_mesh,
            verbose=verbose,
            offset=offset,
            debug_dir=os.path.join(output_dir, f"rendered_pose_estimations_{demo_id}"),
            icp_method=icp_method
        )
            
        if verbose:
            # Compare estimated poses with ground truth poses for debug/calibration
            all_hand_poses_gt = load_ground_truth_poses(obs_group)
            
            render_positions_on_pointclouds_two_colors(
                point_clouds_points,
                point_clouds_colors,
                all_hand_poses,
                all_hand_poses_gt,
                output_dir=os.path.join(output_dir, f"rendered_positions_{demo_id}"),
                verbose=verbose
            )
            
            render_model_on_pointclouds_two_colors(
                point_clouds_points,
                point_clouds_colors,
                all_hand_poses,
                all_hand_poses_gt,
                model=load_model_as_pointcloud(hand_mesh, model_in_mm=True),
                output_dir=os.path.join(output_dir, f"rendered_models_{demo_id}"),
                verbose=verbose
            )
    
    if absolute_actions:
        actions_for_demo = poses_to_absolute_actions(
            poses=all_hand_poses,
            gripper_actions=[root_h["data"][demo]['actions'][i][-1] for i in range(num_samples)],
            control_freq = env_camera0.env.env.control_freq,
            policy_freq = policy_freq,
            smooth=smooth
        )
    else:
        actions_for_demo = poses_to_delta_actions(
            poses=all_hand_poses,
            gripper_actions=[root_h["data"][demo]['actions'][i][-1] for i in range(num_samples)],
            smooth=False,
            translation_scaling=80.0,
            rotation_scaling=9.0,
        )

    roll_out_and_save_video(env_camera0, 
                actions_for_demo, 
                upper_right_video_path,
                policy_freq,
                verbose=verbose, 
                point_clouds_points=point_clouds_points, 
                point_clouds_colors=point_clouds_colors, 
                hand_mesh=hand_mesh,
                output_dir=output_dir, 
                all_hand_poses=all_hand_poses, 
                demo_id=demo_id)
    
    env_camera1.reset()
    env_camera1.reset_to({"states": initial_state})
    
    roll_out_and_save_video(env_camera1,
            actions_for_demo,
            lower_right_video_path,
            policy_freq,
            verbose=verbose,
            point_clouds_points=point_clouds_points,
            point_clouds_colors=point_clouds_colors,
            hand_mesh=hand_mesh,
            output_dir=output_dir,
            all_hand_poses=all_hand_poses,
            demo_id=demo_id)

def imitate_trajectory_with_action_identifier(
    dataset_path=None,
    hand_mesh=None,
    output_dir=None,
    num_demos=100,
    save_webp=False,
    absolute_actions=True,
    ground_truth=False,
    policy_freq=10,
    smooth=False,
    verbose=True,
    offset=[0,0,0],
    icp_method="multiscale",
    max_num_trials=10
) -> None:
    """
    Main function:
    1) Load the dataset.
    2) Initialize rendering environments.
    3) Estimate gripper poses from point clouds.
    4) Estimate pseudo actions from gripper poses.
    5) Roll out pseudo actions in the environment.
    6) Save the video.
    7) Repeat for all demos.
    8) Save the results to a text file.
    9) Optionally convert videos to webp format.
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    cameras = ['fronttableview', 'sidetableview']

    # 1) Load the dataset.
    hdf5_files = glob(f"{dataset_path}/**/*.hdf5", recursive=True)
    hdf5_roots = [h5py.File(fp, "r") for fp in hdf5_files]

    # 2) Initialize rendering environments.
    try:
        # Try using the first file found in hdf5_files
        env_meta = get_env_metadata_from_dataset(dataset_path=hdf5_files[0])
    except Exception as e:
        print(f"Failed to get environment metadata from {hdf5_files[0]}: {e}")

        # If it fails, switch to the parent directory of dataset_path
        parent_path = os.path.dirname(dataset_path)
        print(f"Using parent directory instead: {parent_path}")

        # Now gather .hdf5 files from this parent directory
        hdf5_files_parent = glob(f"{parent_path}/**/*.hdf5", recursive=True)
        if not hdf5_files_parent:
            raise RuntimeError(
                f"No .hdf5 files found in parent directory: {parent_path}"
            )

        # Attempt to get environment metadata again
        env_meta = get_env_metadata_from_dataset(dataset_path=hdf5_files_parent[0])
    if absolute_actions:
        env_meta['env_kwargs']['controller_configs']['control_delta'] = False
        env_meta['env_kwargs']['controller_configs']['type'] = 'OSC_POSE'

    # Setup observation modality specs.
    obs_modality_specs = {
        "obs": {
            "rgb": [f"{cam}_image" for cam in cameras],
            "depth": [f"{cam}_depth" for cam in cameras],
        }
    }
    ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs)

    # Create a rendering environment (we'll use it to obtain image dimensions and camera parameters).
    env_camera0 = create_env_from_metadata(env_meta=env_meta, render_offscreen=True)
    example_image = hdf5_roots[0]["data"]["demo_0"]["obs"][f"{cameras[0]}_image"][0]
    camera_height, camera_width = example_image.shape[:2]

    # 7) Initialize rendering environments for at least two cameras.
    # Use cameras[0] and cameras[1] for video recording.
    env_camera0 = VideoRecordingWrapper(
        env_camera0,
        video_recoder=VideoRecorder.create_h264(fps=20, codec="h264", input_pix_fmt="rgb24", crf=22),
        steps_per_render=1,
        width=camera_width,
        height=camera_height,
        mode="rgb_array",
        camera_name=cameras[0],
    )
    env_camera1 = create_env_from_metadata(env_meta=env_meta, render_offscreen=True)
    env_camera1 = VideoRecordingWrapper(
        env_camera1,
        video_recoder=VideoRecorder.create_h264(fps=20, codec="h264", input_pix_fmt="rgb24", crf=22),
        steps_per_render=1,
        width=camera_width,
        height=camera_height,
        mode="rgb_array",
        camera_name=cameras[1],
    )
    
    results_file_path = os.path.join(output_dir, "trajectory_results.txt")
    with open(results_file_path, "w") as f:
        f.write("Trajectory results:\n")
    
    n_success = 0
    total_n = 0

    # 7) Repeat for all demos.
    for root_h in hdf5_roots:
        demos = list(root_h["data"].keys())[:num_demos] if num_demos else list(root_h["data"].keys())

        for demo in tqdm(demos, desc="Processing demos"):
            demo_id = demo.replace("demo_", "")
            upper_left_video_path  = os.path.join(output_dir, f"{demo_id}_upper_left.mp4")
            upper_right_video_path = os.path.join(output_dir, f"{demo_id}_upper_right.mp4")
            lower_left_video_path  = os.path.join(output_dir, f"{demo_id}_lower_left.mp4")
            lower_right_video_path = os.path.join(output_dir, f"{demo_id}_lower_right.mp4")
            combined_video_path    = os.path.join(output_dir, f"{demo_id}_combined.mp4")

            obs_group = root_h["data"][demo]["obs"]
            num_samples = obs_group[f"{cameras[0]}_image"].shape[0]

            cameras_frames = {}
            for camera in cameras:
                cameras_frames[camera] = [obs_group[f"{camera}_image"][i] for i in range(num_samples)]
                    
            with imageio.get_writer(upper_left_video_path, fps=20) as writer:
                for frame in cameras_frames[cameras[0]]:
                    writer.append_data(frame)
            with imageio.get_writer(lower_left_video_path, fps=20) as writer:
                for frame in cameras_frames[cameras[1]]:
                    writer.append_data(frame)
            
            point_clouds_points = [points for points in obs_group[f"pointcloud_points"]]
            point_clouds_colors = [colors for colors in obs_group[f"pointcloud_colors"]]
            
            if verbose:
                save_point_clouds_as_ply(point_clouds_points, point_clouds_colors, output_dir=os.path.join(output_dir, f"pointclouds_{demo_id}"))
            
            success = False
            i = 0
            
            # 3) Estimate gripper poses from point clouds.
            # 4) Estimate pseudo actions from gripper poses.
            # 5) Roll out pseudo actions in the environment.
            while not success and i < max_num_trials:
                infer_actions_and_rollout(
                    root_h,
                    demo,
                    env_camera0,
                    env_camera1,
                    point_clouds_points,
                    point_clouds_colors,
                    hand_mesh,
                    upper_right_video_path,
                    lower_right_video_path,
                    output_dir,
                    demo_id,
                    policy_freq=policy_freq,
                    smooth=smooth,
                    verbose=verbose,
                    num_samples=num_samples,
                    absolute_actions=absolute_actions,
                    ground_truth=ground_truth,
                    offset=offset,
                    icp_method=icp_method
                )

                # Success check
                success = env_camera0.is_success()["task"]
                if success:
                    n_success += 1
                else:
                    policy_freq = change_policy_freq(policy_freq)
                    print(f"Retrying with policy frequency {policy_freq} Hz...")
                    
                i += 1
            
            total_n += 1
            
            result_str = f"demo_{demo_id}: {f'success after {i} trials' if success else f'fail after {i} trials'}"

            # Immediately append to the results file in "a" (append) mode
            with open(results_file_path, "a") as f:
                f.write(result_str + "\n")

            # 6) Save the video.
            # Combine videos from all cameras
            combine_videos_quadrants(
                upper_left_video_path,
                upper_right_video_path,
                lower_left_video_path,
                lower_right_video_path,
                combined_video_path
            )
            os.remove(upper_left_video_path)
            os.remove(upper_right_video_path)
            os.remove(lower_left_video_path)
            os.remove(lower_right_video_path)

    # 8) Save the results to a text file.
    success_rate = (n_success / total_n)*100 if total_n else 0
    summary_str = f"\nFinal Success Rate: {n_success}/{total_n} => {success_rate:.2f}%"
    print(summary_str)
    with open(results_file_path, "a") as f:
        f.write(summary_str + "\n")

    # 9) Optionally convert videos to webp format.
    if save_webp:
        print("Converting to webp...")
        mp4_files = glob(os.path.join(output_dir, "*.mp4"))
        for mp4_file in tqdm(mp4_files):
            webp_file = mp4_file.replace(".mp4", ".webp")
            try:
                convert_mp4_to_webp(mp4_file, webp_file)
                os.remove(mp4_file)
            except Exception as e:
                print(f"Error converting {mp4_file}: {e}")

    print(f"Wrote results to {os.path.join(output_dir, 'trajectory_results.txt')}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Estimate pseudo actions from video demonstrations, and roll-out the pseudo actions for visualization.")
    
    parser.add_argument('--dataset_path', type=str, default='data/manipulation_demos', help='Path to video dataset directory')
    parser.add_argument('--output_dir', type=str, default='pseudo_action_rollout_visualizations/example_rollout', help='Path to output directory')
    parser.add_argument('--num_demos', type=int, default=1, help='Number of demos to process')
    parser.add_argument('--save_webp', action='store_true', help='Store videos in webp format')
    parser.add_argument('--delta_actions', action='store_true', help='Use delta actions')
    parser.add_argument('--ground_truth', action='store_true', help='For debug: use ground truth poses instead of estimated poses')
    parser.add_argument('--policy_freq', type=int, default=20, choices=POLICY_FREQS, help='Policy frequency')
    parser.add_argument('--smooth', action='store_true', help='Smooth trajectory positions')
    parser.add_argument('--verbose', action='store_true', help='Print debug information and save debug visualizations')
    parser.add_argument('--icp_method', type=str, default='multiscale', choices=['multiscale', 'updown'], help='ICP method used for pose estimation')
    parser.add_argument('--max_num_trials', type=int, default=10, help='Maximum number of trials to attempt for each demo')
    
    args = parser.parse_args()
    
    imitate_trajectory_with_action_identifier(
        dataset_path     = args.dataset_path,
        hand_mesh        = "data/meshes/panda_hand_mesh/panda-hand.ply",
        output_dir       = args.output_dir,
        num_demos        = args.num_demos,
        save_webp        = args.save_webp,
        absolute_actions = not args.delta_actions,
        ground_truth     = args.ground_truth,
        policy_freq      = args.policy_freq,
        smooth           = args.smooth,
        verbose          = args.verbose,
        offset           = POSITIONAL_OFFSET,
        icp_method       = args.icp_method,
        max_num_trials   = args.max_num_trials,
    )