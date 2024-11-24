import os
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from ..datasets import BaseDataset
from einops import rearrange
from tqdm import tqdm
import shutil

class DatasetForLatentActions(BaseDataset):
    def __init__(self, path, video_length, data_modality, cameras):
        super().__init__(
            path=path,
            video_length=video_length,
            cameras=cameras,
            data_modality=data_modality,
            load_actions=False,  # Don't load actions
            compute_stats=False  # Don't compute action statistics
        )
    
    def __getitem__(self, idx):
        root, demo, index, task, camera = self.sequence_paths[idx]
        obs_seq = self.get_samples(root, demo, index)
        
        # Process observations same way as DatasetVideo2Action
        if self.data_modality != 'voxel':
            obs_seq = [torch.from_numpy(rearrange(obs, "h w c -> c h w")).float() for obs in obs_seq]
        else:
            obs_seq = [torch.from_numpy(obs).float() for obs in obs_seq]
            
        video = torch.cat(obs_seq, dim=0)
        return video, (demo, index)

def process_dataset_actions_to_latent_actions(dataset_path, encoder, data_modality, action_type, video_length, cameras, batch_size=32):
    # Split path into components and modify
    dirname = os.path.dirname(dataset_path)
    filename = os.path.basename(dataset_path)
    parent_dir = os.path.basename(dirname)
    grandparent_dir = os.path.dirname(dirname)

    # Create new directory and file names
    new_parent_dir = f"{parent_dir}_{data_modality}"
    new_filename = filename.replace('.hdf5', f'_{data_modality}.hdf5')

    # Combine into new full path
    new_dataset_path = os.path.join(grandparent_dir, new_parent_dir, new_filename)

    # Create new directory if it doesn't exist
    os.makedirs(os.path.dirname(new_dataset_path), exist_ok=True)
    
    # Copy original dataset to new location
    shutil.copy2(dataset_path, new_dataset_path)
    dataset_dir = os.path.dirname(dataset_path)
    
    # Create dataset for processing
    dataset = DatasetForLatentActions(
        path=dataset_dir,
        video_length=video_length,
        data_modality=data_modality,
        cameras=cameras
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Get device from model parameters
    device = next(encoder.parameters()).device
    
    # Get latent dimension from encoder
    with torch.no_grad():
        dummy_input = torch.randn(1, dataset[0][0].shape[0], 128, 128).to(device)
        latent_dim = encoder(dummy_input).shape[1]
    
    # Process sequences and write to new dataset
    with h5py.File(new_dataset_path, 'r+') as f:
        current_demo = None
        current_actions = []
        current_indices = []
        
        # Process all sequences
        encoder.eval()
        with torch.no_grad():
            for videos, (demos, indices) in tqdm(dataloader):
                videos = videos.to(device)
                latent_actions = encoder(videos).squeeze(-1).squeeze(-1).cpu().numpy()
                
                for i, (demo, idx) in enumerate(zip(demos, indices)):
                    if current_demo is None:
                        current_demo = demo
                        # Calculate correct dataset size
                        sequence_length = len(f['data'][demo]['obs']['frontview_image'])
                        action_length = sequence_length - (video_length - 1)
                        # Resize actions dataset for first demo
                        if demo in f['data']:
                            del f['data'][demo]['actions']
                        f['data'][demo].create_dataset('actions', shape=(action_length, latent_dim), dtype=np.float32)
                    
                    if demo != current_demo:
                        # Write accumulated actions
                        actions_dset = f['data'][current_demo]['actions']
                        for action, idx in zip(current_actions, current_indices):
                            if idx < len(actions_dset):  # Verify index is in bounds
                                actions_dset[idx] = action
                            else:
                                print(f"Warning: Skipping out of bounds index {idx} for demo {current_demo}")
                        
                        # Reset for new demo
                        current_demo = demo
                        current_actions = []
                        current_indices = []
                        # Calculate correct dataset size for new demo
                        sequence_length = len(f['data'][demo]['obs']['frontview_image'])
                        action_length = sequence_length - (video_length - 1)
                        # Resize actions dataset
                        if demo in f['data']:
                            del f['data'][demo]['actions']
                        f['data'][demo].create_dataset('actions', shape=(action_length, latent_dim), dtype=np.float32)
                    
                    current_actions.append(latent_actions[i])
                    current_indices.append(idx)
            
            # Write final demo's actions
            if current_demo is not None:
                actions_dset = f['data'][current_demo]['actions']
                for action, idx in zip(current_actions, current_indices):
                    if idx < len(actions_dset):  # Verify index is in bounds
                        actions_dset[idx] = action
                    else:
                        print(f"Warning: Skipping out of bounds index {idx} for demo {current_demo}")
    
    print(f"Created new dataset with latent actions at: {new_dataset_path}")

if __name__ == "__main__":
    import argparse
    from ..utils.utils import load_model

    parser = argparse.ArgumentParser(description="Process dataset actions to latent actions")
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the HDF5 dataset')
    parser.add_argument('--encoder_model_path', type=str, required=True, help='Path to the encoder model')
    parser.add_argument('--data_modality', type=str, required=True, 
                      choices=['cropped_rgbd+color_mask', 'cropped_rgbd+color_mask_depth'],
                      help='Data modality to process')
    parser.add_argument('--action_type', type=str, required=True, help='Type of action to process')
    parser.add_argument('--video_length', type=int, required=True, help='Length of video sequence')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for processing')
    parser.add_argument('--cameras', type=str, nargs='+', default=['frontview_image', 'sideview_image'], help='Camera views to process')

    args = parser.parse_args()

    # Load model using load_model function
    model = load_model(
        architecture='direct_resnet_mlp',
        horizon=args.video_length,
        cameras=args.cameras,
        action_type=args.action_type,
        data_modality=args.data_modality,
        resnet_layers_num=18
    )
    
    # Load encoder weights and extract encoder
    model.conv.load_state_dict(torch.load(args.encoder_model_path))
    encoder = model.conv
    encoder.eval()
    encoder.to('cuda' if torch.cuda.is_available() else 'cpu')

    process_dataset_actions_to_latent_actions(
        args.dataset_path,
        encoder,
        args.data_modality,
        args.action_type,
        args.video_length,
        args.cameras,
        args.batch_size
    )