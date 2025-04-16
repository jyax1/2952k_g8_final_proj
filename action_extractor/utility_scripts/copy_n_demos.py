#!/usr/bin/env python3
"""
Script that copies the first N demonstrations from a source HDF5 file to a new HDF5 file.
Instead of copying demos one-by-one, it copies the entire 'data' group first and then deletes
the demos that should not be present.
All top-level items (besides the 'data' group) are copied in full.
Usage:
    python copy_n_demos_modified.py --source_path /path/to/source.hdf5 --target_path /path/to/target.hdf5 --n 100
"""

import os
import h5py
import numpy as np
from tqdm import tqdm

def copy_n_demos_only(source_path, target_path, n):
    """
    Copies all data from 'source_path' into a new HDF5 file at 'target_path', then deletes
    all demos from the 'data' group that are not among the first n demos.
    Also copies all other groups/datasets at top-level (besides the 'data' group) in full.
    """
    print(f"Copying entire dataset from:\n  {source_path}\nto:\n  {target_path}")
    
    if os.path.exists(target_path):
        raise FileExistsError(f"Target file already exists: {target_path}")
    
    with h5py.File(source_path, 'r') as fsrc, h5py.File(target_path, 'w') as fdst:
        # 1) Copy all top-level items (except 'data' group) directly
        for key in fsrc.keys():
            if key != 'data':
                fsrc.copy(key, fdst)
                print(f"Copied top-level item '{key}' (non-'data').")
        
        # 2) If 'data' group does not exist, just return
        if 'data' not in fsrc:
            print("No 'data' group found in source; nothing else to copy.")
            return
        
        # 3) Copy the entire 'data' group from source to destination
        fsrc.copy('data', fdst)
        print("Copied entire 'data' group.")
        
        # 4) Gather all demos in the target's 'data' group. Typically named "demo_0", "demo_1", ...
        all_demos = list(fdst['data'].keys())
        
        # Sort demos by their integer index, e.g. "demo_0" -> 0, "demo_10" -> 10, etc.
        def get_demo_index(demo_name):
            # Parse the integer from something like 'demo_123'
            try:
                return int(demo_name.split('_')[1])
            except (IndexError, ValueError):
                return -1  # fallback if the name is malformed
        
        all_demos_sorted = sorted(all_demos, key=get_demo_index)
        
        # 5) Identify demos to keep and demos to delete
        demos_to_keep = all_demos_sorted[:n]
        demos_to_delete = all_demos_sorted[n:]
        
        print(f"Found {len(all_demos_sorted)} total demos. Keeping {len(demos_to_keep)} demos and deleting {len(demos_to_delete)} demos...")
        
        # 6) Delete the demos that are not in the first n
        for demo_name in tqdm(demos_to_delete, desc="Deleting extra demos"):
            del fdst['data'][demo_name]
        
        total_samples = 0  # Initialize total counter here
        
        # 7) Compute the total samples from the kept demos
        for demo_name in demos_to_keep:
            demo_grp = fdst['data'][demo_name]
            if "actions" in demo_grp:
                actions = demo_grp["actions"][()]  # shape (T, action_dim)
                total_samples += actions.shape[0]
            else:
                print(f"Warning: Demo '{demo_name}' does not have an 'actions' dataset; skipping count.")
        
        print(f"Total samples from kept demos: {total_samples}")

        # 8) Set the global attribute 'total' in the 'data' group of the target file.
        fdst['data'].attrs["total"] = total_samples
        print(f"Set global attribute 'total' to {total_samples} in 'data' group.")
        
        print(f"Done copying demos into {target_path}.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Copy first N demos from source HDF5 to a new HDF5 file by copying entire dataset then deleting extras.")
    parser.add_argument('--source_path', type=str, required=True, help='Path to the source HDF5 file')
    parser.add_argument('--target_path', type=str, required=True, help='Path to the target HDF5 file')
    parser.add_argument('--n', type=int, required=True, help='Number of demos to keep')
    
    args = parser.parse_args()
    
    copy_n_demos_only(args.source_path, args.target_path, args.n)
