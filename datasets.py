from torch.utils.data import Dataset
import os
from glob import glob
import torch
from einops import rearrange
import zarr
from torchvideotransforms import video_transforms, volume_transforms
from utils.dataset_utils import *
import numpy as np
from tqdm import tqdm

frontview_matrices = np.load('utils/frontview_matrices.npz')
frontview_K = frontview_matrices['K'] # Intrinsics
frontview_R = pose_inv(frontview_matrices['R']) # Extrinsics

sideview_matrices = np.load('utils/sideview_matrices.npz')
sideview_K = sideview_matrices['K'] # Intrinsics
sideview_R = pose_inv(sideview_matrices['R']) # Extrinsics

class BaseDataset(Dataset):
    def __init__(self, path='../datasets/', 
                 video_length=2, 
                 semantic_map=False, 
                 frame_skip=0, 
                 demo_percentage=1.0, 
                 num_demo_train=5000,
                 cameras=['frontview_image'], 
                 data_modality='rgb',
                 action_type='delta_action',
                 validation=False, 
                 random_crop=False, 
                 load_actions=False, 
                 compute_stats=True,
                 action_mean=None,  # Add precomputed action mean
                 action_std=None,
                 coordinate_system='disentangled'):  # Add precomputed action std
        self.path = path
        self.frame_skip = frame_skip
        self.semantic_map = semantic_map
        self.video_length = video_length
        self.load_actions = load_actions
        self.random_crop = random_crop
        self.sequence_paths = []
        self.compute_stats = compute_stats
        self.action_mean = action_mean  # Assign precomputed mean
        self.action_std = action_std    # Assign precomputed std
        self.sum_actions = None
        self.sum_square_actions = None
        self.n_samples = 0
        self.data_modality = data_modality
        self.action_type = action_type
        self.coordinate_system = coordinate_system
        self.cameras = cameras

        # Load dataset and compute stats if needed (only when stats are not provided)
        self._load_datasets(path, demo_percentage, num_demo_train, validation, cameras, max_workers=1)
        if self.compute_stats and (self.action_mean is None or self.action_std is None):
            self._compute_action_statistics()
            
        print(f"Label mean: {self.action_mean}")
        print(f"Label std: {self.action_std}")

        # Define transformation
        self.transform = video_transforms.Compose([volume_transforms.ClipToTensor()])
    
    def _load_datasets(self, path, demo_percentage, num_demo_train, validation, cameras, max_workers=8):
        # Find all HDF5 files and convert to Zarr if necessary
        sequence_dirs = glob(f"{path}/**/*.hdf5", recursive=True)
        for seq_dir in sequence_dirs:
            zarr_path = seq_dir.replace('.hdf5', '.zarr')
            if not os.path.exists(zarr_path):
                # Convert HDF5 to Zarr if it doesn't exist
                hdf5_to_zarr_parallel(seq_dir, max_workers=8)

            # Check for the '{camera}_maskdepth' subdirectory in the Zarr dataset
            root = zarr.open(zarr_path, mode='a')  # Open in append mode to modify if needed
            
            for i in range(len(cameras)):
                camera_name = cameras[i].split('_')[0]
                camera_maskdepth_path = f'data/demo_0/obs/{camera_name}_maskdepth'
                eef_pos_path = f'data/demo_0/obs/robot0_eef_pos_{camera_name}'
                eef_pos_disentangled_path = f'data/demo_0/obs/robot0_eef_pos_{camera_name}_disentangled'
                
                if self.data_modality == 'color_mask_depth':
                    cameras[i] = cameras[i].split('_')[0] + '_maskdepth'
                elif 'cropped_rgbd' in self.data_modality:
                    cameras[i] = cameras[i].split('_')[0] + '_rgbdcrop'

                # If any of the required data paths are missing, preprocess them
                if camera_maskdepth_path not in root:
                    # Call the preprocessing function if any data is missing
                    preprocess_data_parallel(root, camera_name, frontview_R)
        
        # Collect all Zarr files
        self.zarr_files = glob(f"{path}/**/*.zarr", recursive=True)
        self.stores = [zarr.DirectoryStore(zarr_file) for zarr_file in self.zarr_files]
        self.roots = [zarr.open(store, mode='r') for store in self.stores]

        # Process each demo within each Zarr file
        def process_demo(demo, data, task, camera=None):
            obs_frames = len(data['obs'][camera]) if camera else len(data['obs']['voxels'])
            for i in range(obs_frames - self.video_length * (self.frame_skip + 1)):
                self.sequence_paths.append((root, demo, i, task, camera))
                if self.compute_stats and self.load_actions:
                    camera_name = camera.split('_')[0]
                    
                    if self.coordinate_system == 'global':
                        position = 'robot0_eef_pos'
                    elif self.coordinate_system == 'camera':
                        position = f'robot0_eef_pos_{camera_name}'
                    elif self.coordinate_system == 'disentangled':
                        position = f'robot0_eef_pos_{camera_name}_disentangled'
                        
                    if self.action_type == 'position':
                        action = data['obs'][position][i]
                        
                    elif self.action_type == 'delta_position':
                        pos = data['obs'][position][i]
                        pos_next = data['obs'][position][i+1]
                        action = pos_next - pos
                        
                    elif self.action_type == 'position+gripper':
                        eef_pos = data['obs'][position][i]
                        gripper_qpos = data['obs']['robot0_gripper_qpos'][i]
                        action = np.concatenate([eef_pos, gripper_qpos])
                        
                    elif self.action_type == 'delta_position+gripper':
                        eef_pos = data['obs'][position][i]
                        gripper_qpos = data['obs']['robot0_gripper_qpos'][i]
                        
                        eef_pos_next = data['obs'][position][i+1]
                        gripper_qpos_next = data['obs']['robot0_gripper_qpos'][i+1]
                        action = np.concatenate([eef_pos_next - eef_pos, gripper_qpos_next - gripper_qpos])

                    elif self.action_type == 'pose' or self.action_type == 'delta_pose':
                        eef_pos = data['obs']['robot0_eef_pos'][i]    # Shape: (3,)
                        eef_quat = data['obs']['robot0_eef_quat'][i]  # Shape: (4,)
                        gripper_qpos = data['obs']['robot0_gripper_qpos'][i]  # Shape: (2,)
                        action = np.concatenate([eef_pos, eef_quat, gripper_qpos])

                        if self.action_type == 'delta_pose':
                            eef_pos_next = data['obs']['robot0_eef_pos'][i+1]    # Shape: (3,)
                            eef_quat_next = data['obs']['robot0_eef_quat'][i+1]  # Shape: (4,)
                            gripper_qpos_next = data['obs']['robot0_gripper_qpos'][i+1]  # Shape: (2,)
                            
                            pos_diff = eef_pos_next - eef_pos  # Shape: (3,)

                            quat_diff = quaternion_difference(eef_quat, eef_quat_next)  # Shape: (4,)

                            gripper_diff = gripper_qpos_next - gripper_qpos  # Shape: (2,)
                            action = np.concatenate([pos_diff, quat_diff, gripper_diff])
                            
                    else:
                        action = data['actions'][i]
                    
                    if self.sum_actions is None:
                        self.sum_actions = np.zeros(action.shape[-1])
                        self.sum_square_actions = np.zeros(action.shape[-1])

                    self.sum_actions += action
                    self.sum_square_actions += action ** 2
                    self.n_samples += 1

        # Process each Zarr file in parallel
        for zarr_file, root in zip(self.zarr_files, self.roots):
            if validation:
                print(f"Loading {zarr_file} for validation")
            else:
                print(f"Loading {zarr_file} for training")

            task = zarr_file.split("/")[-2].replace('_', ' ')
            demos = list(root['data'].keys())
            if demo_percentage is not None:
                if validation:
                    if demo_percentage == 0.0:
                        start_index = 0
                    else:
                        start_index = int(len(demos) // (1 / demo_percentage))
                    demos = demos[start_index:]
                else:
                    demos = demos[:int(len(demos) // (1 / demo_percentage))]
            else:
                demos = demos[:num_demo_train]

            # Use ThreadPoolExecutor to parallelize demo processing
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for demo in demos:
                    data = root['data'][demo]
                    if self.data_modality == 'voxel':
                        futures.append(executor.submit(process_demo, demo, data, task, 'voxel'))
                    else:
                        for camera in cameras:
                            if self.data_modality == 'color_mask_depth':
                                camera = camera.split('_')[0] + '_maskdepth'
                            elif 'cropped_rgbd' in self.data_modality:
                                camera = camera.split('_')[0] + '_rgbdcrop'
                            if camera in data['obs'].keys():
                                futures.append(executor.submit(process_demo, demo, data, task, camera))
                            else:
                                print(f'Camera {camera} not found in demo {demo}, file {zarr_file}')
                # Wait for all futures to complete
                for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                    pass


    def _compute_action_statistics(self):
        # Compute mean and std from the accumulated sums
        self.action_mean = self.sum_actions / self.n_samples
        variance = (self.sum_square_actions / self.n_samples) - (self.action_mean ** 2)
        self.action_std = np.sqrt(variance)
    
    def get_samples(self, root, demo, index):
        obs_seq = []
        actions_seq = []

        for i in range(self.video_length):
            frames = []
            
            for camera in self.cameras:
                if self.data_modality == 'voxel':
                    obs = root['data'][demo]['obs']['voxels'][index + i * (self.frame_skip + 1)] / 255.0
                    frames.append(obs)
                
                elif self.data_modality == 'rgbd':
                    obs = root['data'][demo]['obs'][camera][index + i * (self.frame_skip + 1)] / 255.0
                    depth_camera = '_'.join([camera.split('_')[0], "depth"])
                    depth = root['data'][demo]['obs'][depth_camera][index + i * (self.frame_skip + 1)] / 255.0
                    obs = np.concatenate((obs, depth), axis=2)
                    frames.append(obs)
                
                elif self.data_modality == 'rgb':
                    obs = root['data'][demo]['obs'][camera][index + i * (self.frame_skip + 1)] / 255.0
                    if self.semantic_map:
                        obs_semantic = root['data'][demo]['obs'][f"{camera}_semantic"][index + i * (self.frame_skip + 1)] / 255.0
                        obs = np.concatenate((obs, obs_semantic), axis=2)
                    frames.append(obs)
                
                elif self.data_modality == 'color_mask_depth':
                    obs = root['data'][demo]['obs'][camera][index + i * (self.frame_skip + 1)] / 255.0
                    frames.append(obs)
                    
                elif self.data_modality == 'cropped_rgbd':
                    obs = root['data'][demo]['obs'][camera][index + i * (self.frame_skip + 1)] / 255.0
                    frames.append(obs)
                    
                elif self.data_modality == 'cropped_rgbd+color_mask_depth':
                    obs = root['data'][demo]['obs'][camera][index + i * (self.frame_skip + 1)] / 255.0
                    mask_depth_camera = '_'.join([camera.split('_')[0], "maskdepth"])
                    mask_depth = root['data'][demo]['obs'][mask_depth_camera][index + i * (self.frame_skip + 1)] / 255.0
                    obs = np.concatenate((obs, mask_depth), axis=2)
                    frames.append(obs)
            
            # Concatenate frames from all cameras along the channel dimension
            obs_seq.append(np.concatenate(frames, axis=2))

            if self.load_actions:
                action = root['data'][demo]['actions'][index + i * (self.frame_skip + 1)]
                if i != self.video_length - 1:
                    actions_seq.append(action)

        if self.load_actions:
            return obs_seq, actions_seq

        return obs_seq

    def __len__(self):
        return len(self.sequence_paths)
    
class DatasetVideo(BaseDataset):
    def __init__(self, path='../datasets/', x_pattern=[0], y_pattern=[1], **kwargs):
        self.x_pattern = x_pattern
        self.y_pattern = y_pattern
        super().__init__(path=path, video_length=max(x_pattern + y_pattern) + 1, **kwargs)
    
    def __getitem__(self, idx):
        root, demo, index, task, camera = self.sequence_paths[idx]
        obs_seq = self.get_samples(root, demo, index)
        obs_seq = [torch.from_numpy(rearrange(obs, "h w c -> c h w")).float() for obs in obs_seq]
        x = torch.cat([obs_seq[i] for i in self.x_pattern], dim=0)
        y = torch.cat([obs_seq[i] for i in self.y_pattern], dim=0)
        return x, y

class DatasetVideo2Action(BaseDataset):
    def __init__(self, path='../datasets/', motion=False, image_plus_motion=False, action_type='delta_action', **kwargs):
        self.motion = motion
        self.image_plus_motion = image_plus_motion
        self.action_type = action_type
        assert not (self.motion and self.image_plus_motion), "Choose either only motion or only image_plus_motion"
        super().__init__(path=path, load_actions=True, action_type=action_type, **kwargs)

    def __getitem__(self, idx):
        root, demo, index, task, camera = self.sequence_paths[idx]
        obs_seq, actions_seq = self.get_samples(root, demo, index)
        
        camera_name = camera.split('_')[0]
        if self.coordinate_system == 'global':
            position = 'robot0_eef_pos'
        elif self.coordinate_system == 'camera':
            position = f'robot0_eef_pos_{camera_name}'
        elif self.coordinate_system == 'disentangled':
            position = f'robot0_eef_pos_{camera_name}_disentangled'

        # Handle action_type logic
        if self.action_type == 'delta_action':
            actions = np.concatenate(actions_seq)  # Current logic for delta_action
            
        elif self.action_type == 'absolute_action':
            # One-to-one mapping of actions to each frame (no need to skip the last frame)
            actions_seq = [root['data'][demo]['actions'][index + i * (self.frame_skip + 1)] for i in range(self.video_length)]
            actions = np.array(actions_seq)
            
        elif self.action_type == 'position':
            actions_seq = [root['data'][demo]['obs'][position][index + i * (self.frame_skip + 1)] for i in range(self.video_length)]
            actions = np.array(actions_seq)
            
        elif self.action_type == 'position+gripper':
            actions_seq = [np.concatenate([root['data'][demo]['obs'][position][index + i * (self.frame_skip + 1)], 
                                          root['data'][demo]['obs']['robot0_gripper_qpos'][index + i * (self.frame_skip + 1)]]) for i in range(self.video_length)]
            actions = np.array(actions_seq)
            
        elif self.action_type == 'delta_position':
            actions_seq = [root['data'][demo]['obs'][position][index + i * (self.frame_skip + 1)] for i in range(self.video_length-1)]
            actions_seq_next = [root['data'][demo]['obs'][position][index + (i+1) * (self.frame_skip + 1)] for i in range(self.video_length-1)]
            actions_diff = [actions_seq_next[i] - actions_seq[i] for i in range(len(actions_seq))]
            actions = np.array(actions_diff)
            
        elif self.action_type == 'delta_position+gripper':
            gripper_actions = [actions_seq[i][-1] for i in range(len(actions_seq))]
            actions_seq = [root['data'][demo]['obs'][position][index + i * (self.frame_skip + 1)] for i in range(self.video_length-1)]
            actions_seq_next = [root['data'][demo]['obs'][position][index + (i+1) * (self.frame_skip + 1)] for i in range(self.video_length-1)]
            actions_diff = [actions_seq_next[i] - actions_seq[i] for i in range(len(actions_seq))]
            actions = np.array([np.append(actions_diff[i], gripper_actions[i]) for i in range(len(actions_diff))])

        elif self.action_type == 'pose':
            for i in range(self.video_length):
                eef_pos = root['data'][demo]['obs']['robot0_eef_pos'][index + i * (self.frame_skip + 1)]    # Shape: (3,)
                eef_quat = root['data'][demo]['obs']['robot0_eef_quat'][index + i * (self.frame_skip + 1)]  # Shape: (4,)
                gripper_qpos = root['data'][demo]['obs']['robot0_gripper_qpos'][index + i * (self.frame_skip + 1)]  # Shape: (2,)
                action = np.concatenate([eef_pos, eef_quat, gripper_qpos])
                actions_seq.append(action)
            actions = np.array(actions_seq)
            
        elif self.action_type == 'delta_pose':
            actions_seq = []
            for i in range(self.video_length - 1):
                eef_pos = root['data'][demo]['obs']['robot0_eef_pos'][index + i * (self.frame_skip + 1)]    # Shape: (3,)
                eef_quat = root['data'][demo]['obs']['robot0_eef_quat'][index + i * (self.frame_skip + 1)]  # Shape: (4,)
                gripper_qpos = root['data'][demo]['obs']['robot0_gripper_qpos'][index + i * (self.frame_skip + 1)]  # Shape: (2,)
                
                eef_pos_next = root['data'][demo]['obs']['robot0_eef_pos'][index + (i+1) * (self.frame_skip + 1)]    # Shape: (3,)
                eef_quat_next = root['data'][demo]['obs']['robot0_eef_quat'][index + (i+1) * (self.frame_skip + 1)]  # Shape: (4,)
                gripper_qpos_next = root['data'][demo]['obs']['robot0_gripper_qpos'][index + (i+1) * (self.frame_skip + 1)]  # Shape: (2,)
                
                pos_diff = eef_pos_next - eef_pos  # Shape: (3,)

                quat_diff = quaternion_difference(eef_quat, eef_quat_next)  # Shape: (4,)

                gripper_diff = gripper_qpos_next - gripper_qpos  # Shape: (2,)
                actions_diff = np.concatenate([pos_diff, quat_diff, gripper_diff])
                actions_seq.append(actions_diff)
                
            actions = np.array(actions_seq)

        # If video_length == 1, return a flat action vector
        if self.video_length == 1 or ('delta' in self.action_type and self.video_length == 2):
            actions = actions.squeeze(0)  # Remove the first dimension to make it (7)

        # Standardize actions if mean and std are computed
        if self.action_mean is not None and self.action_std is not None:
            actions = (actions - self.action_mean) / self.action_std
        
        if self.action_type == 'delta_pose':
            actions[-2:] *= 100

        if self.data_modality != 'voxel':
            obs_seq = [torch.from_numpy(rearrange(obs, "h w c -> c h w")).float() for obs in obs_seq]
        else:
            obs_seq = [torch.from_numpy(obs).float() for obs in obs_seq]

        if self.motion or self.image_plus_motion:
            motion_seq = [(obs_seq[t] - obs_seq[t + 1]) for t in range(len(obs_seq) - 1)]
            motion_seq = [(motion - torch.min(motion)) / (torch.max(motion) - torch.min(motion)) for motion in motion_seq]

            if self.motion:
                video = torch.cat(motion_seq, dim=0)
            else:
                video = torch.cat(obs_seq + motion_seq, dim=0)
        else:
            video = torch.cat(obs_seq, dim=0)

        return video, torch.from_numpy(actions).float()


class DatasetVideo2VideoAndAction(BaseDataset):
    def __init__(self, path='../datasets/', x_pattern=[0], y_pattern=[1], **kwargs):
        self.x_pattern = x_pattern
        self.y_pattern = y_pattern
        super().__init__(path=path, video_length=max(x_pattern + y_pattern) + 1, load_actions=True, **kwargs)

    def __getitem__(self, idx):
        root, demo, index, task, camera = self.sequence_paths[idx]
        obs_seq, actions_seq = self.get_samples(root, demo, index, camera)
        actions = torch.from_numpy(np.concatenate(actions_seq))
        obs_seq = [torch.from_numpy(rearrange(obs, "h w c -> c h w")).float() for obs in obs_seq]

        x = torch.cat([obs_seq[i] for i in self.x_pattern], dim=0)
        y = torch.cat([obs_seq[i] for i in self.y_pattern], dim=0)

        actions = actions.view(actions.shape[0], 1, 1).expand(-1, 128, 128)
        output = torch.cat((y, actions.float()), dim=0)

        return x, output


if __name__ == "__main__":
    train_set = DatasetVideo2Action(
        path='/home/yilong/Documents/ae_data/datasets/mimicgen_core/coffee_rel',
        video_length=1,
        semantic_map=False,
        frame_skip=0,
        random_crop=True,
        demo_percentage=0.9,
        cameras=['frontview_image'],
        motion=True
    )

    print(train_set[0][0])
    print('break')