import os
import shutil
import h5py
import numpy as np
import torch
from tqdm import tqdm
from einops import rearrange
import zarr

from action_extractor.action_identifier import load_action_identifier
from action_extractor.utils.dataset_utils import hdf5_to_zarr_parallel, preprocess_data_parallel
from action_extractor.utils.dataset_utils import frontview_R


def process_dataset_actions_to_latent_actions(
    dataset_path,
    conv_path,
    mlp_path,
    stats_path,
    data_modality='cropped_rgbd+color_mask',
    cameras=["frontview_image", "sideview_image"],
    video_length=2
):
    # Prepare new dataset path
    dirname = os.path.dirname(dataset_path)
    filename = os.path.basename(dataset_path)
    new_filename = filename.replace('.hdf5', f'_{data_modality}.hdf5')
    new_dataset_path = os.path.join(dirname, new_filename)

    # Copy the original dataset to a new location with a new name
    if not os.path.exists(new_dataset_path):
        shutil.copy2(dataset_path, new_dataset_path)
        print(f"Copied dataset to {new_dataset_path}")

    # Convert HDF5 to Zarr if not already done
    zarr_path = new_dataset_path.replace('.hdf5', '.zarr')
    if not os.path.exists(zarr_path):
        print(f"Converting {new_dataset_path} to Zarr format...")
        hdf5_to_zarr_parallel(new_dataset_path, max_workers=8)

    # Open the Zarr dataset for reading observations
    root = zarr.open(zarr_path, mode='a')

    # Open the HDF5 dataset for updating actions
    hdf5_file = h5py.File(new_dataset_path, 'a')

    # Preprocess data if necessary
    all_cameras = ['frontview_image', 'sideview_image', 'agentview_image', 'sideagentview_image']
    for camera_name_full in all_cameras:
        camera_name = camera_name_full.split('_')[0]
        camera_maskdepth_path = f'data/demo_0/obs/{camera_name}_maskdepth'

        if camera_maskdepth_path not in root:
            preprocess_data_parallel(root, camera_name, frontview_R)

    # Adjust camera names based on data_modality
    adjusted_cameras = []
    for camera in cameras:
        if data_modality == 'color_mask_depth':
            adjusted_camera = camera.split('_')[0] + '_maskdepth'
        elif 'cropped_rgbd' in data_modality:
            adjusted_camera = camera.split('_')[0] + '_rgbdcrop'
        else:
            adjusted_camera = camera
        adjusted_cameras.append(adjusted_camera)
    cameras = adjusted_cameras

    # Initialize the ActionIdentifier model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    action_identifier = load_action_identifier(
        conv_path=conv_path,
        mlp_path=mlp_path,
        resnet_version='resnet18',
        video_length=video_length,
        in_channels=len(cameras) * 6,  # Adjusted for multiple cameras
        action_length=1,
        num_classes=4,
        num_mlp_layers=3,
        stats_path=stats_path,
        coordinate_system='global',
        camera_name=cameras[0].split('_')[0]  # Use the first camera for initialization
    ).to(device)
    action_identifier.eval()

    # Process each demo
    demos = list(root["data"].keys())
    for demo in tqdm(demos, desc="Processing demos"):
        num_samples = root["data"][demo]["obs"][cameras[0]].shape[0]

        # Infer actions using the model
        inferred_actions = []
        for i in range(num_samples - video_length + 1):
            # Preprocess and concatenate observations from all cameras
            obs_seq = []
            for j in range(video_length):
                frames = []
                for camera in cameras:
                    obs = root['data'][demo]['obs'][camera][i + j] / 255.0
                    mask_depth_camera = '_'.join([camera.split('_')[0], "maskdepth"])
                    mask_depth = root['data'][demo]['obs'][mask_depth_camera][i + j] / 255.0

                    if data_modality == 'cropped_rgbd+color_mask':
                        mask_depth = mask_depth[:, :, :2]

                    obs = np.concatenate((obs, mask_depth), axis=2)
                    frames.append(obs)

                obs_seq.append(np.concatenate(frames, axis=2))

            # Convert observations to tensors
            obs_seq = [torch.from_numpy(rearrange(obs, "h w c -> c h w")).float() for obs in obs_seq]
            obs_tensor = torch.cat(obs_seq, dim=0).unsqueeze(0).to(device)  # Shape: [1, C*video_length, H, W]

            # Infer action
            with torch.no_grad():
                action = action_identifier.forward_conv(obs_tensor)
            inferred_actions.append(action.cpu().numpy().squeeze())

        inferred_actions.append(inferred_actions[-1]) # Repeat the last action to match the number of samples
        inferred_actions = np.array(inferred_actions)  # Shape: [num_samples - video_length + 1, action_dim]

        # Normalize inferred actions if needed
        # Here you can add your normalization code if required

        # Save inferred actions to the HDF5 dataset
        actions_group = hdf5_file['data'][demo]

        # Delete the existing 'actions' dataset
        if 'actions' in actions_group:
            del actions_group['actions']

        # Create a new 'actions' dataset with the inferred actions
        actions_group.create_dataset('actions', data=inferred_actions)

        print(f"Updated actions for {demo}")

    # Close HDF5 datasets
    hdf5_file.close()

    print("Processing complete.")

if __name__ == "__main__":
    import argparse
    from action_extractor.utils.utils import load_model

    parser = argparse.ArgumentParser(description="Process dataset actions to latent actions")
    parser.add_argument('--dataset_path', type=str, 
                        default='/home/yilong/Documents/policy_data/lift/obs_policy/lift_panda1000_policy_obs.hdf5',
                        help='Path to the HDF5 dataset')
    parser.add_argument('--encoder_model_path', type=str,
                        default='/home/yilong/Documents/action_extractor/results/iiwa16168,lift1000-cropped_rgbd+color_mask-delta_action_norot-frontside-bs1632_resnet-46.pth',
                        help='Path to the encoder model')
    parser.add_argument('--data_modality', type=str,
                        default='cropped_rgbd+color_mask', 
                        choices=['cropped_rgbd+color_mask', 'cropped_rgbd+color_mask_depth'],
                        help='Data modality to process')
    parser.add_argument('--stats_path', type=str, 
                        default='/home/yilong/Documents/ae_data/random_processing/iiwa16168/action_statistics_delta_action_norot.npz', 
                        help='Path to the statistics file of the training set of the encoder_model')
    parser.add_argument('--action_type', type=str, default='delta_action_norot', help='Type of action to process')
    parser.add_argument('--video_length', type=int, default=2, help='Length of video sequence')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for processing')
    parser.add_argument('--cameras', type=str, nargs='+', default=['frontview_image', 'sideview_image'], help='Camera views to process')

    args = parser.parse_args()

    process_dataset_actions_to_latent_actions(
        dataset_path=args.dataset_path,
        conv_path=args.encoder_model_path,
        mlp_path=None,
        stats_path=args.stats_path
    )