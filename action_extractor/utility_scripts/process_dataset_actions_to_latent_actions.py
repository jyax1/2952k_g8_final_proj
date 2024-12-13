import os
import h5py
import numpy as np
import torch
from tqdm import tqdm
import zarr
from glob import glob
from einops import rearrange

from action_extractor.action_identifier import load_action_identifier
from action_extractor.utils.dataset_utils import hdf5_to_zarr_parallel, preprocess_data_parallel
from action_extractor.utils.dataset_utils import pose_inv, frontview_K, frontview_R, sideview_K, sideview_R

def process_dataset_actions_to_latent_actions(
    dataset_path,
    conv_path,
    mlp_path,
    stats_path='/home/yilong/Documents/ae_data/random_processing/iiwa16168/action_statistics_delta_action_norot.npz',
    data_modality='cropped_rgbd+color_mask',
    cameras=["frontview_image", "sideview_image"]
):
    # Create output directory if it doesn't exist
    dirname = os.path.dirname(dataset_path)
    filename = os.path.basename(dataset_path)
    new_filename = filename.replace('.hdf5', f'_{data_modality}.hdf5')
    new_dataset_path = os.path.join(dirname, new_filename)

    # Copy dataset if needed
    if not os.path.exists(new_dataset_path):
        import shutil
        shutil.copy2(dataset_path, new_dataset_path)
        print(f"Copied dataset to {new_dataset_path}")

    # Preprocess dataset
    sequence_dirs = glob(f"{dirname}/**/*.hdf5", recursive=True)
    for seq_dir in sequence_dirs:
        zarr_path = seq_dir.replace('.hdf5', '.zarr')
        if not os.path.exists(zarr_path):
            hdf5_to_zarr_parallel(seq_dir, max_workers=8)

        # Check and preprocess camera data
        root = zarr.open(zarr_path, mode='a')
        all_cameras = ['frontview_image', 'sideview_image', 'agentview_image', 'sideagentview_image']
        
        for camera_name_full in all_cameras:
            camera_name = camera_name_full.split('_')[0]
            camera_maskdepth_path = f'data/demo_0/obs/{camera_name}_maskdepth'
            if camera_maskdepth_path not in root:
                preprocess_data_parallel(root, camera_name, frontview_R)

        # Adjust camera names based on data_modality
        for i in range(len(cameras)):
            if data_modality == 'color_mask_depth':
                cameras[i] = cameras[i].split('_')[0] + '_maskdepth'
            elif 'cropped_rgbd' in data_modality:
                cameras[i] = cameras[i].split('_')[0] + '_rgbdcrop'

    # Initialize the ActionIdentifier model
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

    # Process data
    zarr_files = glob(f"{dirname}/**/*.zarr", recursive=True)
    hdf5_file = h5py.File(new_dataset_path, 'a')

    for zarr_file in zarr_files:
        root = zarr.open(zarr_file, mode='r')
        demos = list(root['data'].keys())

        for demo in tqdm(demos, desc=f"Processing demos in {zarr_file}"):
            obs_group = root['data'][demo]['obs']
            num_samples = obs_group[cameras[0]].shape[0]

            # Process each frame pair
            latent_actions = []
            for i in range(num_samples - 1):
                obs_seq = []
                for j in range(2):  # video_length=2
                    frames = []
                    for camera in cameras:
                        obs = obs_group[camera][i+j] / 255.0
                        mask_depth_camera = '_'.join([camera.split('_')[0], "maskdepth"])
                        mask_depth = obs_group[mask_depth_camera][i+j] / 255.0
                        
                        if data_modality == 'cropped_rgbd+color_mask':
                            mask_depth = mask_depth[:, :, :2]
                        
                        obs = np.concatenate((obs, mask_depth), axis=2)
                        frames.append(obs)
                    
                    obs_seq.append(np.concatenate(frames, axis=2))

                # Convert to tensor
                obs_seq = [torch.from_numpy(rearrange(obs, "h w c -> c h w")).float() for obs in obs_seq]
                obs_tensor = torch.cat(obs_seq, dim=0).unsqueeze(0).to(device)

                # Get latent action
                with torch.no_grad():
                    latent = action_identifier.encode(obs_tensor)
                latent_actions.append(latent.cpu().numpy().squeeze())

            # Add final latent action (repeat last one)
            latent_actions.append(latent_actions[-1])
            latent_actions = np.array(latent_actions)

            # Save to HDF5
            actions_group = hdf5_file['data'][demo]
            if 'actions' in actions_group:
                del actions_group['actions']
            actions_group.create_dataset('actions', data=latent_actions)

    hdf5_file.close()
    print("Processing complete.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, 
                        default='/home/yilong/Documents/ae_data/random_processing/iiwa16168_test/lift_200.hdf5')
    parser.add_argument('--encoder_model_path', type=str,
                        default='/home/yilong/Documents/action_extractor/results/iiwa16168,lift1000-cropped_rgbd+color_mask-delta_position+gripper-frontside-cosine+mse-bs1632_resnet-53.pth')
    parser.add_argument('--stats_path', type=str, 
                        default='/home/yilong/Documents/ae_data/random_processing/iiwa16168/action_statistics_delta_action_norot.npz',)
    parser.add_argument('--data_modality', type=str, default='cropped_rgbd+color_mask')
    parser.add_argument('--cameras', type=str, nargs='+', default=['frontview_image', 'sideview_image'])

    args = parser.parse_args()

    process_dataset_actions_to_latent_actions(
        dataset_path=args.dataset_path,
        conv_path=args.encoder_model_path,
        mlp_path=None,
        stats_path=args.stats_path,
        data_modality=args.data_modality,
        cameras=args.cameras
    )