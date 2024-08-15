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
import glob
import os
from glob import glob
import torch
from tqdm import tqdm
from PIL import Image
import numpy as np
import json
import torchvision.transforms as T
import random
from torchvideotransforms import video_transforms, volume_transforms
from einops import rearrange

import h5py
random.seed(0)
from matplotlib import pyplot as plt
import scipy.ndimage

#from raft import RAFT
from torchvision.models.optical_flow import raft_large

import sys

import zarr
from utils.dataset_utils import hdf5_to_zarr

class DatasetVideo(Dataset):
    def __init__(self, path='../datasets/', sample_per_seq=2, condition_length=1, semantic_map=False, frame_skip=3, demo_percentage=1.0, cameras=['frontview_image'], validation=False, random_crop=False):
        if semantic_map:
            print("Preparing image data from zarr dataset with semantic channel (RGB/RGBD + semantic) ...")
        else:
            print("Preparing image data from zarr dataset ...")
        
        self.frame_skip = frame_skip
        self.semantic_map = semantic_map
        self.sample_per_seq = sample_per_seq
        self.condition_length = condition_length
        self.sequence_paths = []

        # Find all HDF5 files in the directory
        sequence_dirs = glob(f"{path}/**/*.hdf5", recursive=True)

        for seq_dir in sequence_dirs:
            # Check if the corresponding Zarr file exists
            zarr_path = seq_dir.replace('.hdf5', '.zarr')
            if not os.path.exists(zarr_path):
                # Convert HDF5 to Zarr if Zarr file does not exist
                hdf5_to_zarr(seq_dir)
        
        # Collect all Zarr files
        self.zarr_files = glob(f"{path}/**/*.zarr", recursive=True)
        
        # Open each Zarr file and store the root groups
        self.stores = [zarr.DirectoryStore(zarr_file) for zarr_file in self.zarr_files]
        self.roots = [zarr.open(store, mode='r') for store in self.stores]
        
        self.tasks = []
        
        # Collect all observation data paths
        for zarr_file, root in zip(self.zarr_files, self.roots):
            task = zarr_file.split("/")[-2].replace('_', ' ')
            if validation:
                demos = list(root['data'].keys())[int(len(root['data'].keys())//(1/demo_percentage)):]
            else:
                demos = list(root['data'].keys())[:int(len(root['data'].keys())//(1/demo_percentage))]
            for demo in demos:
                data = root['data'][demo]
                for camera in cameras:
                    obs_frames = len(data['obs'][camera])
                    for i in range(obs_frames - self.sample_per_seq * (self.frame_skip + 1)):
                        self.sequence_paths.append((root, demo, i, task, camera))

        self.transform = video_transforms.Compose([
                volume_transforms.ClipToTensor()
        ])
        print('Done')

    def get_samples(self, root, demo, index, camera):
        obs_seq = []
        for i in range(self.sample_per_seq):
            obs = root['data'][demo]['obs'][camera][index + i*(self.frame_skip + 1)] / 255.0

            if self.semantic_map:
                obs_semantic = root['data'][demo]['obs'][f"{camera}_semantic"][index + i*(self.frame_skip + 1)] / 255.0
                obs = np.concatenate((obs, obs_semantic), axis=2)

            obs_seq.append(obs)

        return obs_seq

    def __len__(self):
        return len(self.sequence_paths)

    def __getitem__(self, idx):
        root, demo, index, task, camera = self.sequence_paths[idx]
        obs_seq = self.get_samples(root, demo, index, camera)
        
        obs_seq = [torch.from_numpy(rearrange(obs, "h w c -> c h w")).float() for obs in obs_seq]
        x = torch.cat(obs_seq[self.condition_length:], dim=0)
        x_cond = torch.cat(obs_seq[:self.condition_length], dim=0)
        return x, task, x_cond
    

class DatasetVideo2DeltaAction(Dataset):
    def __init__(self, path='../datasets/', video_length=2, semantic_map=False, frame_skip=0, demo_percentage=1.0, 
                 cameras=['frontview_image'], validation=False, random_crop=False, motion=False, image_plus_motion=False):
        if semantic_map:
            print("Preparing image data from zarr dataset with semantic channel (RGB/RGBD + semantic) ...")
        else:
            print("Preparing image data from zarr dataset ...")
        
        self.frame_skip = frame_skip
        self.semantic_map = semantic_map
        self.video_length = video_length
        self.motion = motion
        self.image_plus_motion = image_plus_motion
        assert not (self.motion and self.image_plus_motion), "Choose either only motion or only image_plus_motion"

        self.sequence_paths = []

        # Find all HDF5 files in the directory
        sequence_dirs = glob(f"{path}/**/*.hdf5", recursive=True)

        for seq_dir in sequence_dirs:
            # Check if the corresponding Zarr file exists
            zarr_path = seq_dir.replace('.hdf5', '.zarr')
            if not os.path.exists(zarr_path):
                # Convert HDF5 to Zarr if Zarr file does not exist
                hdf5_to_zarr(seq_dir)
        
        # Collect all Zarr files
        self.zarr_files = glob(f"{path}/**/*.zarr", recursive=True)
        
        # Open each Zarr file and store the root groups
        self.stores = [zarr.DirectoryStore(zarr_file) for zarr_file in self.zarr_files]
        self.roots = [zarr.open(store, mode='r') for store in self.stores]
        
        self.tasks = []
        
        # Collect all observation data paths
        for zarr_file, root in zip(self.zarr_files, self.roots):
            task = zarr_file.split("/")[-2].replace('_', ' ')
            if validation:
                demos = list(root['data'].keys())[int(len(root['data'].keys())//(1/demo_percentage)):]
            else:
                demos = list(root['data'].keys())[:int(len(root['data'].keys())//(1/demo_percentage))]
            for demo in demos:
                data = root['data'][demo]
                for camera in cameras:
                    obs_frames = len(data['obs'][camera])
                    #for i in range(2):
                    for i in range(obs_frames - self.video_length * (self.frame_skip + 1)):
                        self.sequence_paths.append((root, demo, i, task, camera))

        self.transform = video_transforms.Compose([
                volume_transforms.ClipToTensor()
        ])
        print('Done')

    def get_samples(self, root, demo, index, camera):
        obs_seq = []
        actions_seq = []
        for i in range(self.video_length):
            obs = root['data'][demo]['obs'][camera][index + i*(self.frame_skip + 1)] / 255.0
            action = root['data'][demo]['actions'][index + i*(self.frame_skip + 1)]

            if self.semantic_map:
                obs_semantic = root['data'][demo]['obs'][f"{camera}_semantic"][index + i*(self.frame_skip + 1)] / 255.0
                obs = np.concatenate((obs, obs_semantic), axis=2)

            if i != self.video_length - 1:
                actions_seq.append(action)

            obs_seq.append(obs)

        return obs_seq, actions_seq

    def __len__(self):
        return len(self.sequence_paths)

    def __getitem__(self, idx):
        root, demo, index, task, camera = self.sequence_paths[idx]
        obs_seq, actions_seq = self.get_samples(root, demo, index, camera)
        
        actions = torch.from_numpy(np.concatenate(actions_seq))
        obs_seq = [torch.from_numpy(rearrange(obs, "h w c -> c h w")).float() for obs in obs_seq]

        if self.motion or self.image_plus_motion:
            motion_seq = []

            for t in range(len(obs_seq)-1):
                motion = obs_seq[t] - obs_seq[t+1]
                motion = (motion - torch.min(motion)) / (torch.max(motion) - torch.min(motion))
                motion_seq.append(motion)
            
            if self.motion:
                video = torch.cat(motion_seq, dim=0)
            else:
                video = torch.cat(obs_seq + motion_seq, dim=0)

        else:
            video = torch.cat(obs_seq, dim=0)

        return video, actions.float() # Will this cause performance issue?

    

if __name__ == "__main__":
    train_set = DatasetVideo2DeltaAction(
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