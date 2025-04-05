#!/usr/bin/env python3
import h5py

def count_demonstrations(hdf5_file):
    # Open the file in read-only mode
    with h5py.File(hdf5_file, 'r') as f:
        # Access the 'data' group
        data_group = f['data']
        # Get all keys in the 'data' group
        demo_keys = list(data_group.keys())
        # Print the number of demonstrations
        print("Number of demonstrations:", len(demo_keys))
        # Optionally, print the keys themselves

if __name__ == '__main__':
    hdf5_file = "/home/yilong/Documents/mimicgen/datasets/core/square_d0.hdf5"
    hdf5_file = "/home/yilong/Documents/mimicgen/datasets/robot/square_d0_ur5e.hdf5"
    hdf5_file = "/home/yilong/Documents/mimicgen/datasets/robot/square_d0_iiwa.hdf5"
    hdf5_file = "/home/yilong/Documents/robo/robomimic_3d/datasets/lift/mh/low_dim_v141.hdf5"
    hdf5_file = "/home/yilong/Documents/datasets/source/coffee_d0_abs_obs.hdf5"
    count_demonstrations(hdf5_file)