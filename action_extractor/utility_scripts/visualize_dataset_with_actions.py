import os
import h5py
from tqdm import tqdm

import robomimic.utils.obs_utils as ObsUtils
from robomimic.utils.file_utils import get_env_metadata_from_dataset
from robomimic.utils.env_utils import create_env_from_metadata

from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from action_extractor.utils.rollout_utils import change_policy_freq

def visualize_dataset_trajectories_as_videos(args) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    
    cameras = ['fronttableview', 'sidetableview'] # For visualization
    
    env_meta = get_env_metadata_from_dataset(dataset_path=args.hdf5_path)
    
    if not args.delta_actions: # Using absolute actions
        env_meta['env_kwargs']['controller_configs']['control_delta'] = False
        env_meta['env_kwargs']['controller_configs']['type'] = 'OSC_POSE'
        
    obs_modality_specs = {
        "obs": {
            "rgb": [f"{cam}_image" for cam in cameras],
            "depth": [f"{cam}_depth" for cam in cameras],
        }
    }
    ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs)
    
    hdf5_root = h5py.File(args.hdf5_path, "r")
    
    env_camera0 = create_env_from_metadata(env_meta=env_meta, render_offscreen=True)
    example_image = hdf5_root["data"]["demo_0"]["obs"][f"{cameras[0]}_image"][0]
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
    
    demos = list(hdf5_root["data"].keys())[:args.num_demos]
    for demo in tqdm(demos, desc="Rolling out demos"):
        demo_id = demo.replace("demo_", "")
        video_path_cam1 = os.path.join(args.output_dir, f"{demo_id}_cam1.mp4")
        video_path_cam2 = os.path.join(args.output_dir, f"{demo_id}_cam2.mp4")
        
        obs_group = hdf5_root["data"][demo]["obs"]
        num_samples = obs_group[f"{cameras[0]}_image"].shape[0]
        initial_state = hdf5_root["data"][demo]["states"][0]
        
        env_camera0.reset()
        env_camera0.reset_to({"states": initial_state})
        
        
        

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Estimate pseudo actions from video demonstrations, and roll-out the pseudo actions for visualization.")
    
    parser.add_argument('--hdf5_path', type=str, default='data/manipulation_demos', help='Path to video dataset directory')
    parser.add_argument('--output_dir', type=str, default='visualization/dataset', help='Path to output directory')
    parser.add_argument('--num_demos', type=int, default=1, help='Number of demos to process')
    parser.add_argument('--save_webp', action='store_true', help='Store videos in webp format')
    parser.add_argument('--delta_actions', action='store_true', help='Use delta actions')
    
    args = parser.parse_args()
    
    visualize_dataset_trajectories_as_videos(
        args = args
    )