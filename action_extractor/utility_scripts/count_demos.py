#!/usr/bin/env python3
import h5py
import argparse

def count_demonstrations(hdf5_file):
    # Open the file in read-only mode
    with h5py.File(hdf5_file, 'r') as f:
        data_group = f['data']
        demo_keys = list(data_group.keys())
        print("Number of demonstrations:", len(demo_keys))

def main():
    parser = argparse.ArgumentParser(description='Count demonstrations in an HDF5 file')
    parser.add_argument('hdf5_file', type=str, help='Path to the HDF5 file')
    args = parser.parse_args()
    
    count_demonstrations(args.hdf5_file)

if __name__ == '__main__':
    main()