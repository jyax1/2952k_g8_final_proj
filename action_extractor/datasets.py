import os
import numpy as np
import torch
from action_extractor.utils.dataset_utils import pose_inv, camera_extrinsics, frontview_R
from action_extractor.utils.dataset_utils import *

from torch.utils.data import Dataset
from glob import glob
from einops import rearrange
import zarr
from zarr import DirectoryStore
from zarr import ZipStore
import shutil
from torchvideotransforms import video_transforms, volume_transforms
import numpy as np
from tqdm import tqdm

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
                 compute_stats=False,
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
        self.all_cameras = ['frontview_image', 'sideview_image', 'agentview_image', 'sideagentview_image']
        # self.all_cameras = ['frontview_image', 'sideview_image', 'agentview_image']

        # Load dataset and compute stats if needed (only when stats are not provided)
        self._load_datasets(path, demo_percentage, num_demo_train, validation, cameras, self.all_cameras, max_workers=1)
        if self.compute_stats and (self.action_mean is None or self.action_std is None):
            self._compute_action_statistics()
            
        print(f"Label mean: {self.action_mean}")
        print(f"Label std: {self.action_std}")

        # Define transformation
        self.transform = video_transforms.Compose([volume_transforms.ClipToTensor()])
    
    def _load_datasets(self, path, demo_percentage, num_demo_train, validation, cameras, all_cameras, max_workers=8):
        # Find all HDF5 files and convert to Zarr if necessary
        sequence_dirs = glob(f"{path}/**/*.hdf5", recursive=True)
        self.hdf5_files = sequence_dirs
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
                if self.data_modality == 'color_mask_depth':
                    cameras[i] = cameras[i].split('_')[0] + '_maskdepth'
                elif 'cropped_rgbd' in self.data_modality:
                    cameras[i] = cameras[i].split('_')[0] + '_rgbdcrop'
            
        # Collect all Zarr files
        self.zarr_files = glob(f"{path}/**/*.zarr.zip", recursive=True)
        self.stores = [ZipStore(zarr_file, mode='r') for zarr_file in self.zarr_files]
        self.roots = [zarr.group(store) for store in self.stores]

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
                        eef_pos_next = data['obs'][position][i+1]
                        gripper_action = data['actions'][i][-1]
                        
                        action = np.append(eef_pos_next - eef_pos, gripper_action)

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
                    
                    elif self.action_type == 'delta_action_norot':
                        action = data['actions'][i]
                        action = np.delete(action, [3, 4, 5])
                    
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
                        camera = cameras[0]
                        if self.data_modality == 'color_mask_depth':
                            camera = cameras[0].split('_')[0] + '_maskdepth'
                        elif 'cropped_rgbd' in self.data_modality:
                            camera = cameras[0].split('_')[0] + '_rgbdcrop'
                            
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
        
        # Generate the file name based on self.action_type
        file_name = f'action_statistics_{self.action_type}.npz'
        save_path = os.path.join(self.path, file_name)
        
        # Save the computed mean and std to an .npz file
        np.savez(save_path, action_mean=self.action_mean, action_std=self.action_std)
        print(f"Saved action statistics to {save_path}")
    
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
                    
                elif self.data_modality == 'cropped_rgbd+color_mask_depth' or self.data_modality == 'cropped_rgbd+color_mask':
                    obs = root['data'][demo]['obs'][camera][index + i * (self.frame_skip + 1)] / 255.0
                    mask_depth_camera = '_'.join([camera.split('_')[0], "maskdepth"])
                    mask_depth = root['data'][demo]['obs'][mask_depth_camera][index + i * (self.frame_skip + 1)] / 255.0
                    
                    if self.data_modality == 'cropped_rgbd+color_mask':
                        mask_depth = mask_depth[:, :, :2]
                    
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
            
        if self.action_type == 'delta_action_norot':
            actions_seq = [np.delete(action, [3, 4, 5]) for action in actions_seq]
            actions = np.concatenate(actions_seq)
                
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
        if self.video_length == 1 or ('delta' in self.action_type and self.video_length == 2) and 'delta_action' not in self.action_type:
            actions = actions.squeeze(0)  # Remove the first dimension to make it (7)

        # Standardize actions if mean and std are computed
        if self.action_mean is not None and self.action_std is not None:
            self.action_mean[-1] = 0. ###### Gripper action is always either -1 or 1, no need to standardize
            self.action_std[-1] = 1.  ###### Gripper action is always either -1 or 1
            
            actions = (actions - self.action_mean) / self.action_std
        
        # if self.action_type == 'delta_pose':
        #     actions[-2:] *= 100

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
    from action_extractor.utility_scripts.validation_visualization import visualize_visible_points
    DEPTH_MINMAX = {'birdview_depth': [1.180, 2.480],
                'agentview_depth': [0.1, 1.1],
                'sideagentview_depth': [0.1, 1.1],
                'sideview_depth': [1.0, 2.0],
                'robot0_eye_in_hand_depth': [0., 1.0],
                'sideview2_depth': [0.8, 2.2],
                'backview_depth': [0.6, 1.6],
                'frontview_depth': [1.2, 2.2],
                'spaceview_depth': [0.45, 1.45],
                'farspaceview_depth': [0.58, 1.58],
                }
    
    frontview_x_range, frontview_y_range, frontview_z_range = get_visible_xyz_range(frontview_R, frontview_K, z_range=(1.2, 2.2))
    sideview_x_range, sideview_y_range, sideview_z_range = get_visible_xyz_range(sideview_R, sideview_K, z_range=(1.0, 2.0))
    agentview_x_range, agentview_y_range, agentview_z_range = get_visible_xyz_range(agentview_R, agentview_K, z_range=(0.1, 1.1))
    sideagentview_x_range, sideagentview_y_range, sideagentview_z_range = get_visible_xyz_range(sideagentview_R, sideagentview_K, z_range=(0.1, 1.1))
    
    print(f"frontview: x: {frontview_x_range[0]:.2f}, {frontview_x_range[1]:.2f}, y: {frontview_y_range[0]:.2f}, {frontview_y_range[1]:.2f}, z: {frontview_z_range[0]:.2f}, {frontview_z_range[1]:.2f}")
    print(f"sideview: x: {sideview_x_range[0]:.2f}, {sideview_x_range[1]:.2f}, y: {sideview_y_range[0]:.2f}, {sideview_y_range[1]:.2f}, z: {sideview_z_range[0]:.2f}, {sideview_z_range[1]:.2f}")
    print(f"agentview: x: {agentview_x_range[0]:.2f}, {agentview_x_range[1]:.2f}, y: {agentview_y_range[0]:.2f}, {agentview_y_range[1]:.2f}, z: {agentview_z_range[0]:.2f}, {agentview_z_range[1]:.2f}")
    print(f"sideagentview: x: {sideagentview_x_range[0]:.2f}, {sideagentview_x_range[1]:.2f}, y: {sideagentview_y_range[0]:.2f}, {sideagentview_y_range[1]:.2f}, z: {sideagentview_z_range[0]:.2f}, {sideagentview_z_range[1]:.2f}") 
    
    visualize_visible_points(agentview_K, agentview_R, agentview_x_range, agentview_y_range, agentview_z_range)