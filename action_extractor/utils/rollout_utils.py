from action_extractor.point_cloud.config import *
from action_extractor.utils.poses_utils import *
from action_extractor.utils.rollout_debug_utils import *
from action_extractor.point_cloud.action_identifier_point_cloud import *
from action_extractor.utils.poses_to_actions import *

import random
import numpy as np
import os


def change_policy_freq(policy_freq):
    current_policy_freq = policy_freq

    # Filter out the old policy_freq
    valid_options = [opt for opt in POLICY_FREQS if opt != current_policy_freq]

    # Randomly pick a new policy_freq
    return random.choice(valid_options)

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