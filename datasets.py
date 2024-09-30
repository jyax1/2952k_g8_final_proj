'''
Loads two main categories of datasets with sub-categories:
1. Consecutive video frames with corresponding actions for supervised learning
    a. Raw frames (Baseline)
    b. Preprocessed into "motion, objects, scene" triplets (Sun et al.: MOSO: Decomposing MOtion, Scene and Object for Video Prediction, 2023)
    c. Preprocessed into "motion" and raw image pairs (Inspired by Sun et al.)
    d. Preprocessed into "objects, scene" pairs (Inspired by Sun et al.)
    e. Preprocessed with segmentation (SAM 2)
    f. All the above combined with different action label types:
        i. Delta action
        ii. Absolute gripper pose in work space
        iii. Joint angles
        iv. (Need to look into it for more options)
2. Consecutive video frames without labels for self-supervised pretraining
    (Same sub-categories)

Questions:
1. Do the frames have to be consecutive? (Can there be skipped)
'''


from torch.utils.data import Dataset
import os
from glob import glob
import torch
from einops import rearrange
import zarr
from torchvideotransforms import video_transforms, volume_transforms
from utils.dataset_utils import hdf5_to_zarr
import numpy as np


class BaseDataset(Dataset):
    def __init__(self, path='../datasets/', 
                 video_length=2, 
                 semantic_map=False, 
                 frame_skip=0, 
                 demo_percentage=1.0, 
                 cameras=['frontview_image'], 
                 validation=False, 
                 random_crop=False, 
                 load_actions=False, 
                 compute_stats=True):
        self.path = path
        self.frame_skip = frame_skip
        self.semantic_map = semantic_map
        self.video_length = video_length
        self.load_actions = load_actions
        self.random_crop = random_crop
        self.sequence_paths = []
        self.compute_stats = compute_stats  # Flag to compute stats during dataset loading
        self.action_mean = None
        self.action_std = None
        self.sum_actions = None
        self.sum_square_actions = None
        self.n_samples = 0

        # Load dataset and compute stats if needed
        self._load_datasets(path, demo_percentage, validation, cameras)
        if self.compute_stats:
            self._compute_action_statistics()

        # Define transformation
        self.transform = video_transforms.Compose([volume_transforms.ClipToTensor()])

    def _load_datasets(self, path, demo_percentage, validation, cameras):
        # Find all HDF5 files and convert to Zarr if necessary
        sequence_dirs = glob(f"{path}/**/*.hdf5", recursive=True)
        for seq_dir in sequence_dirs:
            zarr_path = seq_dir.replace('.hdf5', '.zarr')
            if not os.path.exists(zarr_path):
                hdf5_to_zarr(seq_dir)

        # Collect all Zarr files
        self.zarr_files = glob(f"{path}/**/*.zarr", recursive=True)
        self.stores = [zarr.DirectoryStore(zarr_file) for zarr_file in self.zarr_files]
        self.roots = [zarr.open(store, mode='r') for store in self.stores]

        # Collect all observation data paths
        for zarr_file, root in zip(self.zarr_files, self.roots):
            print(f"Loading {zarr_file}")
            task = zarr_file.split("/")[-2].replace('_', ' ')
            demos = list(root['data'].keys())
            if validation:
                demos = demos[int(len(demos) // (1 / demo_percentage)):]
            else:
                demos = demos[:int(len(demos) // (1 / demo_percentage))]

            for demo in demos:
                data = root['data'][demo]
                for camera in cameras:
                    if camera in data['obs'].keys():
                        obs_frames = len(data['obs'][camera])
                        for i in range(obs_frames - self.video_length * (self.frame_skip + 1)):
                            self.sequence_paths.append((root, demo, i, task, camera))
                            if self.compute_stats and self.load_actions:
                                # Accumulate statistics for each action during loading
                                actions_seq = data['actions'][i]
                                if self.sum_actions is None:
                                    self.sum_actions = np.zeros(actions_seq.shape[-1])
                                    self.sum_square_actions = np.zeros(actions_seq.shape[-1])

                                self.sum_actions += actions_seq
                                self.sum_square_actions += actions_seq ** 2
                                self.n_samples += 1
                    else:
                        print(f'Camera {camera} not found in demo {demo}, file {zarr_file}')

    def _compute_action_statistics(self):
        # Compute mean and std from the accumulated sums
        self.action_mean = self.sum_actions / self.n_samples
        variance = (self.sum_square_actions / self.n_samples) - (self.action_mean ** 2)
        self.action_std = np.sqrt(variance)

    def get_samples(self, root, demo, index, camera):
        obs_seq = []
        actions_seq = []
        for i in range(self.video_length):
            obs = root['data'][demo]['obs'][camera][index + i * (self.frame_skip + 1)] / 255.0
            if self.semantic_map:
                obs_semantic = root['data'][demo]['obs'][f"{camera}_semantic"][index + i * (self.frame_skip + 1)] / 255.0
                obs = np.concatenate((obs, obs_semantic), axis=2)
            obs_seq.append(obs)

            if self.load_actions:
                action = root['data'][demo]['actions'][index + i * (self.frame_skip + 1)]
                if i != self.video_length - 1:
                    actions_seq.append(action)

        if self.load_actions:
            return obs_seq, actions_seq
        return obs_seq

    def __len__(self):
        return len(self.sequence_paths)

class DatasetVideo2Action(BaseDataset):
    def __init__(self, path='../datasets/', motion=False, image_plus_motion=False, action_type='delta_pose', **kwargs):
        self.motion = motion
        self.image_plus_motion = image_plus_motion
        self.action_type = action_type  # New argument to specify action type
        assert not (self.motion and self.image_plus_motion), "Choose either only motion or only image_plus_motion"
        super().__init__(path=path, load_actions=True, **kwargs)

    def __getitem__(self, idx):
        root, demo, index, task, camera = self.sequence_paths[idx]
        obs_seq, actions_seq = self.get_samples(root, demo, index, camera)

        # Handle action_type logic
        if self.action_type == 'delta_pose':
            actions = np.concatenate(actions_seq)  # Current logic for delta_pose
        elif self.action_type == 'absolute_pose':
            # One-to-one mapping of actions to each frame (no need to skip the last frame)
            actions_seq = [root['data'][demo]['actions'][index + i * (self.frame_skip + 1)] for i in range(self.video_length)]
            actions = np.array(actions_seq)

        # If video_length == 1, return a flat action vector
        if self.video_length == 1:
            actions = actions.squeeze(0)  # Remove the first dimension to make it (7)

        # Standardize actions if mean and std are computed
        if self.action_mean is not None and self.action_std is not None:
            actions = (actions - self.action_mean) / self.action_std

        obs_seq = [torch.from_numpy(rearrange(obs, "h w c -> c h w")).float() for obs in obs_seq]

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
        path="/users/ysong135/scratch/datasets",
        video_length=2,
        semantic_map=False,
        frame_skip=0,
        random_crop=True,
        demo_percentage=0.9,
        cameras=['frontview_image'],
        motion=True
    )

    print(train_set[0][0])
    print('break')