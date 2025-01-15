import os
import h5py
import numpy as np
import torch
from tqdm import tqdm
import imageio
import cv2
import zarr
from glob import glob
from einops import rearrange
from zarr import DirectoryStore, ZipStore
import shutil

from action_extractor.action_identifier import load_action_identifier, VariationalEncoder
from action_extractor.utils.dataset_utils import hdf5_to_zarr_parallel_with_progress, preprocess_data_parallel, directorystore_to_zarr_zip
from robomimic.utils.file_utils import get_env_metadata_from_dataset
from robomimic.utils.env_utils import create_env_from_metadata
from action_extractor.utils.dataset_utils import pose_inv, frontview_K, frontview_R, sideview_K, sideview_R, agentview_K, agentview_R, sideagentview_K, sideagentview_R
import robomimic.utils.obs_utils as ObsUtils
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder


variational = False

right_video_mode = 'inferred_actions'
dataset_path = "/home/yilong/Documents/policy_data/lift/obs_policy_c"
dataset_path = "/home/yilong/Documents/policy_data/lift/lift_smaller_2000"
dataset_path = "/home/yilong/Documents/ae_data/random_processing/lift_1000"
# dataset_path = "/home/yilong/Documents/policy_data/lift/raw/1736557347_6730564/test"
# dataset_path = "/home/yilong/Documents/ae_data/random_processing/iiwa16168_test"
# conv_path='/home/yilong/Documents/action_extractor/results/iiwa16168,lift1000-cropped_rgbd+color_mask-delta_position+gripper-frontside-bs1632_resnet-49-353.pth'
# mlp_path='/home/yilong/Documents/action_extractor/results/iiwa16168,lift1000-cropped_rgbd+color_mask-delta_position+gripper-frontside-bs1632_mlp-49-353.pth'

# non-variational settings
conv_path = '/home/yilong/Documents/action_extractor/results/resnet_global_average_pooling-lift1000-cropped_rgbd+color_mask-delta_position+gripper-frontside-cosine+mse-bs1632_resnet-45.pth'
mlp_path = '/home/yilong/Documents/action_extractor/results/resnet_global_average_pooling-lift1000-cropped_rgbd+color_mask-delta_position+gripper-frontside-cosine+mse-bs1632_mlp-45.pth'

n = 100
save_webp = False

deterministic=True

# variational settings
# conv_path = '/home/yilong/Documents/action_extractor/results/variational-iiwa16168,lift1000-cropped_rgbd+color_mask-delta_position+gripper-frontside-cosine+mse-bs1632_resnet-73.pth'
# mlp_path = '/home/yilong/Documents/action_extractor/results/variational-iiwa16168,lift1000-cropped_rgbd+color_mask-delta_position+gripper-frontside-cosine+mse-bs1632_mlp-73.pth'
if variational:
    fc_mu_path = '/home/yilong/Documents/action_extractor/results/variational-iiwa16168,lift1000-cropped_rgbd+color_mask-delta_position+gripper-frontside-cosine+mse-bs1632_fc_mu-73.pth'
    fc_logvar_path = '/home/yilong/Documents/action_extractor/results/variational-iiwa16168,lift1000-cropped_rgbd+color_mask-delta_position+gripper-frontside-cosine+mse-bs1632_fc_logvar-73.pth'
else:
    fc_mu_path = None
    fc_logvar_path = None
    
output_dir = "/home/yilong/Documents/action_extractor/debug/resnet_test_1000"

# checkpoint_path = '/home/yilong/Documents/action_extractor/results/S_variational-lift1000-cropped_rgbd+color_mask-delta_position+gripper-frontside-cosine+mse-bs1632*8-rejection_checkpoint.pth'
# arch_type = 'resnet'
checkpoint_path = '/home/yilong/Documents/action_extractor/results/resnet_global_average_pooling-lift1000-cropped_rgbd+color_mask-delta_position+gripper-frontside-cosine+mse-bs1632_checkpoint_latest.pth'
checkpoint_path = None
arch_type = 'resnet'

# conv_path='/home/yilong/Documents/action_extractor/results/iiwa16168-cropped_rgbd+color_mask-delta_position+gripper-frontside-bs1632_resnet-50-300.pth'
# mlp_path='/home/yilong/Documents/action_extractor/results/iiwa16168-cropped_rgbd+color_mask-delta_position+gripper-frontside-bs1632_mlp-50-300.pth'

def convert_mp4_to_webp(input_path, output_path, quality=80):
    """Convert mp4 video to webp format"""
    import subprocess
    import shutil
    
    # Find ffmpeg executable path
    ffmpeg_path = shutil.which('ffmpeg')
    if ffmpeg_path is None:
        raise RuntimeError("ffmpeg not found. Please install it with: sudo apt-get install ffmpeg")
        
    cmd = [
        ffmpeg_path,  # Use full path
        '-i', input_path,
        '-c:v', 'libwebp',
        '-quality', str(quality),
        '-lossless', '0',
        '-compression_level', '6',
        '-qmin', '0',
        '-qmax', '100',
        '-preset', 'default',
        '-loop', '0',
        '-vsync', '0',
        '-f', 'webp',
        output_path
    ]
    subprocess.run(cmd, check=True)

def imitate_trajectory_with_action_identifier(
    dataset_path=dataset_path,
    output_dir=output_dir,
    conv_path=conv_path,
    mlp_path=mlp_path,
    # stats_path='/home/yilong/Documents/ae_data/random_processing/lift_1000/action_statistics_delta_position+gripper.npz',
    stats_path='/home/yilong/Documents/ae_data/random_processing/iiwa16168/action_statistics_delta_position+gripper_oscar.npz',
    n_demos=None,
    data_modality='cropped_rgbd+color_mask',
    cameras=["frontview_image", "sideview_image"],
    right_video_mode=right_video_mode,  # New parameter to select the rollout type
    save_webp=save_webp  # New parameter
):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
            
    # Preprocess dataset
    sequence_dirs = glob(f"{dataset_path}/**/*.hdf5", recursive=True)
    for seq_dir in sequence_dirs:
        ds_dir = seq_dir.replace('.hdf5', '.zarr')        # e.g. /path/to/file.zarr
        zarr_path = seq_dir.replace('.hdf5', '.zarr.zip')
        # 1) Convert HDF5 -> DirectoryStore if needed
        if not os.path.exists(zarr_path):
            hdf5_to_zarr_parallel_with_progress(seq_dir)
        
            # 2) Preprocess data in the DirectoryStore 
            store = DirectoryStore(ds_dir)
            root = zarr.group(store, overwrite=False)  # or mode='r+' in older Zarr versions
            # run your preprocess_data_parallel on the directory store
            # but you must adapt: currently it expects a "ZipStore"? 
            # Actually it just needs a 'root' with shape/dtype. 
            # So it should be the same code, just pass root, not ZipStore
            all_cameras = ['frontview_image', 'sideview_image', 'agentview_image', 'sideagentview_image']
            for i in range(len(all_cameras)):
                camera_name = all_cameras[i].split('_')[0]
                obs_group = root['data']['demo_0']['obs']
                mask_key = f"{camera_name}_maskdepth"
                if mask_key not in obs_group:
                    # Call the preprocessing function if any data is missing
                    preprocess_data_parallel(root, camera_name, frontview_R)

            store.close()

            # 3) Now convert DirectoryStore -> .zarr.zip
            directorystore_to_zarr_zip(ds_dir, zarr_path)

            # 4) Clean up the .zarr directory if you only want the final .zarr.zip
            shutil.rmtree(ds_dir)
                
        for i in range(len(cameras)):
            if data_modality == 'color_mask_depth':
                cameras[i] = cameras[i].split('_')[0] + '_maskdepth'
            elif 'cropped_rgbd' in data_modality:
                cameras[i] = cameras[i].split('_')[0] + '_rgbdcrop'
            
    # Collect all Zarr files
    zarr_files = glob(f"{dataset_path}/**/*.zarr.zip", recursive=True)
    stores = [ZipStore(zarr_file, mode='r') for zarr_file in zarr_files]
    roots = [zarr.group(store) for store in stores]

    # Initialize the ActionIdentifier model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    action_identifier = load_action_identifier(
        checkpoint_path=checkpoint_path,
        conv_path=conv_path,
        mlp_path=mlp_path,
        resnet_version='resnet18',
        video_length=2,
        in_channels=len(cameras) * 6,  # Adjusted for multiple cameras
        action_length=1,
        num_classes=4,
        num_mlp_layers=3,
        fc_mu_path=fc_mu_path,
        fc_logvar_path=fc_logvar_path,
        stats_path=stats_path,
        coordinate_system='global',
        camera_name=cameras[0].split('_')[0],  # Use the first camera for initialization
        deterministic=deterministic,
        arch_type=arch_type
    ).to(device)
    action_identifier.eval()

    # Initialize observation utilities
    env_meta = get_env_metadata_from_dataset(dataset_path=sequence_dirs[0])
    obs_modality_specs = {
        "obs": {
            "rgb": cameras,
            "depth": [f"{camera.split('_')[0]}_depth" for camera in cameras],
        }
    }
    ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs)

    # Create the environment
    env_camera0 = create_env_from_metadata(
        env_meta=env_meta,
        render_offscreen=True
    )
    # Wrap with VideoRecordingWrapper
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
        steps_per_render=1,  # Ensure every step renders a frame
        width=128,           # Specify the desired width
        height=128,          # Specify the desired height
        mode='rgb_array',    # Ensure the render mode is set correctly
        camera_name=cameras[0].split('_')[0]  # Specify the camera name if needed
    )
    
    # Create the environment
    env_camera1 = create_env_from_metadata(
        env_meta=env_meta,
        render_offscreen=True
    )
    # Wrap with VideoRecordingWrapper
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
        steps_per_render=1,  # Ensure every step renders a frame
        width=128,           # Specify the desired width
        height=128,          # Specify the desired height
        mode='rgb_array',    # Ensure the render mode is set correctly
        camera_name=cameras[1].split('_')[0]  # Specify the camera name if needed
    )
    
    n_success = 0
    total_n = 0
    results = []

    # Process each demo
    for root in roots:
        if n is not None:
            demos = list(root["data"].keys())[:n]
        else:
            demos = list(root["data"].keys())
            
        for demo in tqdm(demos, desc="Processing demos"):
            demo_id = demo.replace("demo_", "")
            upper_left_video_path = os.path.join(output_dir, f"{demo_id}_upper_left.mp4")
            upper_right_video_path = os.path.join(output_dir, f"{demo_id}_upper_right.mp4")
            lower_left_video_path = os.path.join(output_dir, f"{demo_id}_lower_left.mp4")
            lower_right_video_path = os.path.join(output_dir, f"{demo_id}_lower_right.mp4")
            combined_video_path = os.path.join(output_dir, f"{demo_id}_combined.mp4")

            # Extract observations for the left video
            obs_group = root["data"][demo]["obs"]
            num_samples = obs_group[cameras[0]].shape[0]
            upper_left_frames = [obs_group[cameras[0].split('_')[0] + '_image'][i] for i in range(num_samples)]
            lower_left_frames = [obs_group[cameras[1].split('_')[0] + '_image'][i] for i in range(num_samples)]

            # Save the left video
            with imageio.get_writer(upper_left_video_path, fps=20) as writer:
                for frame in upper_left_frames:
                    writer.append_data(frame)
                    
            with imageio.get_writer(lower_left_video_path, fps=20) as writer:
                for frame in lower_left_frames:
                    writer.append_data(frame)

            # Prepare to collect inferred actions or position differences
            if right_video_mode == 'inferred_actions':
                
                actions_dataset = root["data"][demo]["actions"][:-1]
                
                # Infer actions using the model
                inferred_actions = []
                
                for i in range(num_samples - 1):
                    # Preprocess and concatenate observations from all cameras
                    obs_seq = []
                    for j in range(2):
                        frames = []
                        for camera in cameras:                           
                            obs = root['data'][demo]['obs'][camera][i+j] / 255.0
                            mask_depth_camera = '_'.join([camera.split('_')[0], "maskdepth"])
                            mask_depth = root['data'][demo]['obs'][mask_depth_camera][i+j] / 255.0
                            
                            if data_modality == 'cropped_rgbd+color_mask':
                                mask_depth = mask_depth[:, :, :2]
                            
                            obs = np.concatenate((obs, mask_depth), axis=2)
                            frames.append(obs)
                            
                        obs_seq.append(np.concatenate(frames, axis=2))
                        
                    obs_seq = [torch.from_numpy(rearrange(obs, "h w c -> c h w")).float() for obs in obs_seq]
                    
                    obs_tensor = torch.cat(obs_seq, dim=0).to(device)
                    obs_tensor = obs_tensor.unsqueeze(0)

                    # Infer action
                    with torch.no_grad():
                        if deterministic:
                            action = action_identifier(obs_tensor)
                        else:
                            action, _, _ = action_identifier(obs_tensor)
                    inferred_actions.append(action.cpu().numpy().squeeze())
                    
                # Reset environment to initial state from dataset
                initial_state = root["data"][demo]["states"][0]
                env_camera0.reset()
                env_camera0.reset_to({"states": initial_state})

                # Set up video recording for the right video
                env_camera0.file_path = upper_right_video_path  # Set the file path to start recording
                env_camera0.step_count = 0  # Reset the step counter

                # Run the inferred actions in the environment
                for i, action in enumerate(inferred_actions):
                    # Insert three zeros into the fourth, fifth, and sixth positions
                    action = np.insert(action, [3, 3, 3], 0.0)
                    # action[:3] /= (np.linalg.norm(action[:3]) + 1e-8)
                    # # action *= 
                    action *= 80
                    action[-1] = np.sign(action[-1])
                    
                    env_camera0.step(action)

                # Stop recording and reset the file path
                env_camera0.video_recoder.stop()  # Ensure the video recorder stops
                env_camera0.file_path = None      # Reset the file path
                
                env_camera1.reset()
                env_camera1.reset_to({"states": initial_state})
                
                env_camera1.file_path = lower_right_video_path  # Set the file path to start recording
                env_camera1.step_count = 0  # Reset the step counter
                
                # Run the inferred actions in the environment
                for i, action in enumerate(inferred_actions):
                    # Insert three zeros into the fourth, fifth, and sixth positions
                    action = np.insert(action, [3, 3, 3], 0.0)
                    # action[:3] /= (np.linalg.norm(action[:3]) + 1e-8)
                    # # action *= 60
                    action *= 80
                    action[-1] = np.sign(action[-1])

                    env_camera1.step(action)
                    
                env_camera1.video_recoder.stop()
                env_camera1.file_path = None
                
                success = env_camera0.is_success()['task'] and env_camera1.is_success()['task']
                if success:
                    n_success += 1
                total_n += 1
                results.append(f"{demo}: {'success' if success else 'failed'}")

            elif right_video_mode == 'ground_truth':
                # Extract end-effector positions
                eef_positions = root["data"][demo]["obs"]["robot0_eef_pos"]
                actions_dataset = root["data"][demo]["actions"][:-1]  # Extract actions from the dataset
                position_differences = np.diff(eef_positions, axis=0)

                # Ensure that actions and position_differences have the same length
                assert len(actions_dataset) == len(position_differences), "Mismatch in lengths"

                # Reset environment to initial state from dataset
                initial_state = root["data"][demo]["states"][0]
                env_camera0.reset()
                env_camera0.reset_to({"states": initial_state})

                # Set up video recording for the right video
                env_camera0.file_path = upper_right_video_path  # Set the file path to start recording
                env_camera0.step_count = 0  # Reset the step counter

                # Apply position differences as actions
                for i, delta_pos in enumerate(position_differences):
                    # Create action vector
                    action = np.zeros(7)
                    action[:3] = delta_pos  # Set positional deltas

                    # Normalize the first three dimensions of action
                    action_magnitude = np.linalg.norm(action[:3])
                    dataset_action_magnitude = np.linalg.norm(actions_dataset[i][:3])
                    if action_magnitude != 0:  # Avoid division by zero
                        scaling_factor = dataset_action_magnitude / action_magnitude
                        action[:3] *= scaling_factor

                    action[-1] = actions_dataset[i][-1]  # Set the final dimension from the dataset
                    env_camera0.step(action)

                # Stop recording and reset the file path
                env_camera0.video_recoder.stop()  # Ensure the video recorder stops
                env_camera0.file_path = None      # Reset the file path
                
                env_camera1.reset()
                env_camera1.reset_to({"states": initial_state})
                
                env_camera1.file_path = lower_right_video_path  # Set the file path to start recording
                env_camera1.step_count = 0  # Reset the step counter
                
                # Apply position differences as actions
                for i, delta_pos in enumerate(position_differences):
                    # Create action vector
                    action = np.zeros(7)
                    action[:3] = delta_pos
                    
                    # Normalize the first three dimensions of action
                    action_magnitude = np.linalg.norm(action[:3])
                    dataset_action_magnitude = np.linalg.norm(actions_dataset[i][:3])
                    if action_magnitude != 0:
                        scaling_factor = dataset_action_magnitude / action_magnitude
                        action[:3] *= scaling_factor
                        
                    action[-1] = actions_dataset[i][-1]
                    env_camera1.step(action)
                    
                env_camera1.video_recoder.stop()
                env_camera1.file_path = None

            else:
                raise ValueError(f"Invalid right_video_mode: {right_video_mode}")

            # Combine left and right videos side by side
            combine_videos_quadrants(upper_left_video_path, upper_right_video_path, lower_left_video_path, lower_right_video_path, combined_video_path)

            # Remove individual videos if desired
            os.remove(upper_left_video_path)
            os.remove(upper_right_video_path)
            os.remove(lower_left_video_path)
            os.remove(lower_right_video_path)
    
    if right_video_mode == 'inferred_actions':
        success_rate = (n_success/total_n)*100
        results.append(f"\nFinal Success Rate: {n_success}/{total_n}: {success_rate:.2f}%")
        
        results_path = os.path.join(output_dir, "trajectory_results.txt")
        with open(results_path, "w") as f:
            f.write("\n".join(results))

    # After all videos are processed
    if save_webp:
        print("Converting videos to webp format...")
        mp4_files = glob(os.path.join(output_dir, "*.mp4"))
        for mp4_file in tqdm(mp4_files, desc="Converting to webp"):
            webp_file = mp4_file.replace('.mp4', '.webp')
            try:
                convert_mp4_to_webp(mp4_file, webp_file)
                os.remove(mp4_file)  # Remove original mp4 after successful conversion
            except Exception as e:
                print(f"Error converting {mp4_file}: {e}")

def preprocess_cropped_rgbd_color_mask(obs_group, camera_name, index):
    """
    Preprocess the observations to create cropped_rgbd+color_mask data.
    """
    camera_name = camera_name.split('_')[0]  # Remove the suffix
    # Get the RGB image and depth image
    rgb_image = obs_group[f"{camera_name}_image"][index] / 255.0
    depth_image = obs_group[f"{camera_name}_depth"][index] / 255.0

    # Convert RGB image to HSV
    hsv_image = cv2.cvtColor((rgb_image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)

    # Define color ranges in HSV for green and cyan
    green_lower, green_upper = np.array([40, 40, 90]), np.array([80, 255, 255])
    cyan_lower, cyan_upper = np.array([80, 40, 100]), np.array([100, 255, 255])

    # Create masks for green and cyan
    green_mask = cv2.inRange(hsv_image, green_lower, green_upper)
    cyan_mask = cv2.inRange(hsv_image, cyan_lower, cyan_upper)

    # Union of green and cyan masks
    combined_mask = cv2.bitwise_or(green_mask, cyan_mask)

    # Create the mask-depth array by stacking green and cyan masks
    maskdepth_array = np.zeros((rgb_image.shape[0], rgb_image.shape[1], 2), dtype=np.uint8)
    maskdepth_array[..., 0] = green_mask
    maskdepth_array[..., 1] = cyan_mask

    # Create the RGBD image
    rgbd_image = np.concatenate((rgb_image, depth_image), axis=2)  # Shape: (128, 128, 4)

    # Calculate bounding box for the combined mask
    x, y, w, h = cv2.boundingRect(combined_mask)

    # Create a bounding box mask
    bbox_mask = np.zeros((rgb_image.shape[0], rgb_image.shape[1]), dtype=np.uint8)
    bbox_mask[y:y+h, x:x+w] = 1

    # Expand bbox_mask to match the RGB-D image shape
    bbox_mask_expanded = bbox_mask[..., np.newaxis]  # Shape: (128, 128, 1)

    # Mask the RGB-D image with the bounding box mask, setting regions outside the bounding box to zero
    rgbd_cropped = rgbd_image * bbox_mask_expanded

    # Concatenate RGBD cropped image and mask-depth array
    obs = np.concatenate((rgbd_cropped, maskdepth_array / 255.0), axis=2)
    return obs

def combine_videos_quadrants(top_left_video_path, top_right_video_path, bottom_left_video_path, bottom_right_video_path, output_path):
    # Read videos
    top_left_reader = imageio.get_reader(top_left_video_path)
    top_right_reader = imageio.get_reader(top_right_video_path)
    bottom_left_reader = imageio.get_reader(bottom_left_video_path)
    bottom_right_reader = imageio.get_reader(bottom_right_video_path)
    fps = top_left_reader.get_meta_data()["fps"]

    top_left_frames = [frame for frame in top_left_reader]
    top_right_frames = [frame for frame in top_right_reader]
    bottom_left_frames = [frame for frame in bottom_left_reader]
    bottom_right_frames = [frame for frame in bottom_right_reader]

    # Ensure same number of frames
    min_length = min(len(top_left_frames), len(top_right_frames), len(bottom_left_frames), len(bottom_right_frames))
    top_left_frames = top_left_frames[:min_length]
    top_right_frames = top_right_frames[:min_length]
    bottom_left_frames = bottom_left_frames[:min_length]
    bottom_right_frames = bottom_right_frames[:min_length]

    # Combine frames into quadrants
    combined_frames = [
        np.vstack([
            np.hstack([top_left, top_right]),
            np.hstack([bottom_left, bottom_right])
        ])
        for top_left, top_right, bottom_left, bottom_right in zip(top_left_frames, top_right_frames, bottom_left_frames, bottom_right_frames)
    ]

    # Save combined video
    with imageio.get_writer(output_path, fps=fps) as writer:
        for frame in combined_frames:
            writer.append_data(frame)
            
def compute_direction_difference(actions_dataset, inferred_actions):
    """
    Compute the difference in direction for each axis between corresponding vectors
    in actions_dataset and inferred_actions.

    Parameters:
    actions_dataset (np.ndarray): Array of shape (N, 7) containing the actions from the dataset.
    inferred_actions (np.ndarray): Array of shape (N, 4) containing the inferred actions.

    Returns:
    np.ndarray: Array of shape (N, 3) containing the difference in direction for each axis.
    """
    # Extract the first three dimensions
    actions_dataset_vectors = actions_dataset[:, :3]
    inferred_actions_vectors = inferred_actions[:, :3]

    # Compute norms
    actions_dataset_norms = np.linalg.norm(actions_dataset_vectors, axis=1, keepdims=True)
    inferred_actions_norms = np.linalg.norm(inferred_actions_vectors, axis=1, keepdims=True)

    # Avoid division by zero by setting norms to 1 where they are zero
    actions_dataset_norms = np.where(actions_dataset_norms == 0, 1, actions_dataset_norms)
    inferred_actions_norms = np.where(inferred_actions_norms == 0, 1, inferred_actions_norms)

    # Normalize the vectors to get their direction
    actions_dataset_directions = actions_dataset_vectors / actions_dataset_norms
    inferred_actions_directions = inferred_actions_vectors / inferred_actions_norms

    # Set directions to zero where norms were zero
    actions_dataset_directions = np.where(actions_dataset_norms == 1, 0, actions_dataset_directions)
    inferred_actions_directions = np.where(inferred_actions_norms == 1, 0, inferred_actions_directions)
    
    inferred_actions_directions[:, 2] += 1.0

    # Compute the difference in direction for each axis
    direction_difference = actions_dataset_directions - inferred_actions_directions

    return direction_difference

if __name__ == "__main__":
    imitate_trajectory_with_action_identifier()