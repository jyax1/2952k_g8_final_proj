import os
import h5py
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm

import torch
from action_extractor.action_identifier import load_action_identifier

from robomimic.utils.file_utils import get_env_metadata_from_dataset
from robomimic.utils.env_utils import create_env_from_metadata
import robomimic.utils.obs_utils as ObsUtils
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from moviepy.editor import VideoFileClip, clips_array

latent_action = True

conv_path='/home/yilong/Documents/action_extractor/results/iiwa16168,lift1000-cropped_rgbd+color_mask-delta_position+gripper-frontside-cosine+mse-bs1632_resnet-46.pth'
mlp_path='/home/yilong/Documents/action_extractor/results/iiwa16168,lift1000-cropped_rgbd+color_mask-delta_position+gripper-frontside-cosine+mse-bs1632_mlp-46.pth'
cameras=["frontview_image", "sideview_image"]
stats_path='/home/yilong/Documents/ae_data/random_processing/iiwa16168/action_statistics_delta_action_norot.npz'

# Define the path to the HDF5 file and output directory
hdf5_file_path = '/home/yilong/Documents/ae_data/random_processing/iiwa200/action_extraction_IIWA200.hdf5'
output_dir = '/home/yilong/Documents/action_extractor/debug/D_movement_iiwa200'
output_format = 'mp4'  # Choose 'mp4' or 'webp'

latent_hdf5_file_path = '/home/yilong/Documents/ae_data/random_processing/iiwa16168_test/lift_200_cropped_rgbd+color_mask.hdf5'
latent_output_dir = '/home/yilong/Documents/action_extractor/debug/D_demo_latent_panda200_delta_pos_cosine+mse_model'

def visualize_action_dataset_as_videos():
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open the HDF5 file
    with h5py.File(hdf5_file_path, 'r') as root:
        
        i = 0
        # Iterate over each demo
        for demo in tqdm(root['data']):
            if i == 500:
                break
            
            # Retrieve the frontview_image for this demo
            frontview_images = root['data'][demo]['obs']['frontview_image'][:]
            
            if output_format == 'mp4':
                # Save as MP4 video
                # Get the number of frames, height, width, and channels
                n_frames, height, width, channels = frontview_images.shape
                
                # Define the video codec and create a VideoWriter object
                video_path = os.path.join(output_dir, f'{demo}.mp4')
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                fps = 30  # Adjust the frames per second if needed
                out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
                
                # Loop over frames and write each frame to the video
                for frame in frontview_images:
                    # Convert the frame from RGB to BGR for OpenCV
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    out.write(frame_bgr)
                
                # Release the VideoWriter
                out.release()
            
            elif output_format == 'webp':
                # Save as animated WebP
                # Convert images to a list of PIL Images
                pil_images = [Image.fromarray(frame) for frame in frontview_images]
                
                # Save as animated WebP
                output_path = os.path.join(output_dir, f'{demo}.webp')
                pil_images[0].save(
                    output_path, 
                    format='WEBP', 
                    save_all=True, 
                    append_images=pil_images[1:], 
                    duration=33,  # Duration between frames in milliseconds (30 FPS)
                    loop=0        # 0 means loop indefinitely
                )
            else:
                print(f"Unsupported output format: {output_format}")
                break

            i += 1
            
def visualize_latent_action_dataset_as_video():
    os.makedirs(latent_output_dir, exist_ok=True)
    
    env_meta = get_env_metadata_from_dataset(dataset_path=latent_hdf5_file_path)
    obs_modality_specs = {
        "obs": {
            "rgb": cameras,
            "depth": [f"{camera.split('_')[0]}_depth" for camera in cameras],
        }
    }
    ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs)

    # Create environments for both cameras
    env_camera0 = create_env_from_metadata(env_meta=env_meta, render_offscreen=True)
    env_camera0 = VideoRecordingWrapper(
        env_camera0,
        video_recoder=VideoRecorder.create_h264(
            fps=20,
            codec='h264',
            input_pix_fmt='rgb24',
            crf=22,
            thread_type='FRAME',
            thread_count=1
        ),
        steps_per_render=1,
        width=256,
        height=256,
        mode='rgb_array',
        camera_name=cameras[0].split('_')[0]
    )

    env_camera1 = create_env_from_metadata(env_meta=env_meta, render_offscreen=True)
    env_camera1 = VideoRecordingWrapper(
        env_camera1,
        video_recoder=VideoRecorder.create_h264(
            fps=20,
            codec='h264',
            input_pix_fmt='rgb24',
            crf=22,
            thread_type='FRAME',
            thread_count=1
        ),
        steps_per_render=1,
        width=256,
        height=256,
        mode='rgb_array',
        camera_name=cameras[1].split('_')[0]
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    action_identifier = load_action_identifier(
        conv_path=conv_path,
        mlp_path=mlp_path,
        resnet_version='resnet18',
        video_length=2,
        in_channels=len(cameras) * 6,
        action_length=1,
        num_classes=4,
        num_mlp_layers=3,
        stats_path=stats_path,
        coordinate_system='global',
        camera_name=cameras[0].split('_')[0]
    ).to(device)
    action_identifier.eval()

    # Add tracking variables
    n_success = 0
    total_n = 0
    results = []

    with h5py.File(latent_hdf5_file_path, 'r') as root:
        i = 0
        for demo in tqdm(root['data']):
            if i == 200:
                break

            latent_actions_dataset = root["data"][demo]["actions"][:-1]
            latent_actions_dataset = torch.from_numpy(latent_actions_dataset).float().to(device)
            latent_actions_dataset = latent_actions_dataset.unsqueeze(1)

            initial_state = root["data"][demo]["states"][0]
            env_camera0.reset()
            env_camera0.reset_to({"states": initial_state})

            env_camera1.reset()
            env_camera1.reset_to({"states": initial_state})

            # Set up video recording
            left_video_path = os.path.join(latent_output_dir, f'{demo}_left.mp4')
            env_camera0.file_path = left_video_path
            env_camera0.step_count = 0

            right_video_path = os.path.join(latent_output_dir, f'{demo}_right.mp4')
            env_camera1.file_path = right_video_path
            env_camera1.step_count = 0

            for j in range(len(latent_actions_dataset)):
                true_action = action_identifier.decode(latent_actions_dataset[j])
                true_action = true_action.detach().cpu().numpy()
                true_action = np.insert(true_action, [3, 3, 3], 0.0)
                # Remove the magnitude normalization
                true_action[-1] = np.sign(true_action[-1])
                
                env_camera0.step(true_action)
                env_camera1.step(true_action)

            env_camera0.video_recoder.stop()
            env_camera0.file_path = None

            env_camera1.video_recoder.stop()
            env_camera1.file_path = None

            # Check success after running actions
            success = env_camera0.is_success()['task'] and env_camera1.is_success()['task']
            if success:
                n_success += 1
            total_n += 1
            results.append(f"{demo}: {'success' if success else 'failed'}")

            # Combine the two videos side by side
            combined_video_path = os.path.join(latent_output_dir, f'{demo}.mp4')

            left_clip = VideoFileClip(left_video_path)
            right_clip = VideoFileClip(right_video_path)

            # Ensure both clips have the same duration
            min_duration = min(left_clip.duration, right_clip.duration)
            left_clip = left_clip.subclip(0, min_duration)
            right_clip = right_clip.subclip(0, min_duration)

            # Create a side-by-side video
            combined_clip = clips_array([[left_clip, right_clip]])

            # Write the combined video to file
            combined_clip.write_videofile(combined_video_path, fps=20)

            # Close the video clips
            left_clip.close()
            right_clip.close()
            combined_clip.close()

            # Remove the original left and right videos
            os.remove(left_video_path)
            os.remove(right_video_path)

            i += 1

        # Calculate and save results
        success_rate = (n_success/total_n)*100
        results.append(f"\nFinal Success Rate: {n_success}/{total_n}: {success_rate:.2f}%")
        
        results_path = os.path.join(latent_output_dir, "trajectory_results.txt")
        with open(results_path, "w") as f:
            f.write("\n".join(results))

        print(f"Success Rate: {success_rate:.2f}% ({n_success}/{total_n})")

if __name__ == '__main__':
    
    if latent_action:
        visualize_latent_action_dataset_as_video()
    else:
        visualize_action_dataset_as_videos()
    
    print("Videos saved successfully.")