#!/usr/bin/env python3
"""
Script that copies the first n demonstrations from a source HDF5 file to a specified target HDF5 file.
Usage:
    python copy_n_demos.py /path/to/source.hdf5 /path/to/target.hdf5 100
"""

import os
import h5py
import numpy as np
from tqdm import tqdm
import shutil

def copy_n_demos(source_path, target_path, n):
    """
    Copies the first n demos from 'source_path' HDF5 file into a new HDF5 file at 'target_path'.
    Any demos with index >= n are deleted from the copy.
    """
    print(f"Copying from:\n  {source_path}\nto:\n  {target_path}")
    
    # Copy entire file first
    shutil.copy2(source_path, target_path)
    
    # Open target file and delete excess demos
    with h5py.File(target_path, 'r+') as target:
        all_demos = list(target['data'].keys())
        demos_to_delete = []
        
        # Find demos with index >= n
        for demo in all_demos:
            try:
                # typically "demo_0", "demo_1", etc.
                demo_idx = int(demo.split('_')[1])
                if demo_idx >= n:
                    demos_to_delete.append(demo)
            except (IndexError, ValueError):
                print(f"Warning: Skipping malformed demo name: {demo}")
        
        if demos_to_delete:
            print(f"Deleting {len(demos_to_delete)} demos with index >= {n}")
            
            for demo in tqdm(demos_to_delete, desc="Deleting excess demos"):
                del target['data'][demo]
    
    print(f"Successfully created dataset with demos [0..{n-1}] at:\n  {target_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Copy first N demos from source HDF5 to a new HDF5 file.")
    parser.add_argument('--source_path', type=str, help='Path to the source HDF5 file')
    parser.add_argument('--target_path', type=str, help='Path to the target HDF5 file')
    parser.add_argument('-n', type=int, help='Number of demos to keep')
    
    args = parser.parse_args()
    
    copy_n_demos(args.source_path, args.target_path, args.n)
