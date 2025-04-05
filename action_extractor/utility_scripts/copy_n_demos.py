#!/usr/bin/env python3
"""
Script that copies the first N demonstrations from a source HDF5 file to a new HDF5 file.
It does NOT copy all demos and then delete the unwanted ones; it only copies the first N to begin with.
All top-level items (besides the 'data' group) are copied in full.
Usage:
    python copy_n_demos_minimal.py --source_path /path/to/source.hdf5 --target_path /path/to/target.hdf5 --n 100
"""

import os
import h5py
import numpy as np
from tqdm import tqdm

def copy_n_demos_only(source_path, target_path, n):
    """
    Copies the first n demos from 'source_path' into a new HDF5 file at 'target_path'.
    Also copies all other groups/datasets at top-level (besides the 'data' group) in full.
    """
    print(f"Copying up to {n} demos from:\n  {source_path}\nto:\n  {target_path}")
    
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
        
        # 3) Create 'data' group in the target
        fdst.create_group('data')
        
        # 4) Gather all demos in the source's 'data' group. Typically named "demo_0", "demo_1", ...
        all_demos = list(fsrc['data'].keys())
        
        # Sort demos by their integer index, e.g. "demo_0" -> 0, "demo_10" -> 10, etc.
        # (In case they are not in lexical order or we want to handle re-labeled demos.)
        def get_demo_index(demo_name):
            # Parse the integer from something like 'demo_123'
            try:
                return int(demo_name.split('_')[1])
            except (IndexError, ValueError):
                return -1  # or some fallback if the name is malformed
        
        all_demos_sorted = sorted(all_demos, key=get_demo_index)
        
        # 5) Copy only the first n demos
        demos_to_copy = all_demos_sorted[:n]
        
        print(f"Found {len(all_demos_sorted)} total demos. Copying {len(demos_to_copy)} demos...")
        
        total_samples = 0  # <-- Initialize total counter here
        
        for demo_name in tqdm(demos_to_copy, desc="Copying demos"):
            # Copy the entire group for that demo
            fsrc.copy(fsrc['data'][demo_name], fdst['data'], name=demo_name)
            
            # Read the demo group from the source file to get the transition count
            demo_grp_in = fsrc['data'][demo_name]
            if "actions" in demo_grp_in:
                actions = demo_grp_in["actions"][()]  # shape (T, action_dim)
                total_samples += actions.shape[0]       # T is the number of transitions
            else:
                print(f"Warning: Demo '{demo_name}' does not have an 'actions' dataset; skipping count.")
        
        print(f"Total samples from copied demos: {total_samples}")

        # Set the global attribute 'total' in the 'data' group of the target file.
        fdst['data'].attrs["total"] = total_samples
        print(f"Set global attribute 'total' to {total_samples} in 'data' group.")
        
        print(f"Done copying {len(demos_to_copy)} demos into {target_path}.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Copy first N demos from source HDF5 to a new HDF5 file (without copying everything).")
    parser.add_argument('--source_path', type=str, required=True, help='Path to the source HDF5 file')
    parser.add_argument('--target_path', type=str, required=True, help='Path to the target HDF5 file')
    parser.add_argument('--n', type=int, required=True, help='Number of demos to keep')
    
    args = parser.parse_args()
    
    copy_n_demos_only(args.source_path, args.target_path, args.n)