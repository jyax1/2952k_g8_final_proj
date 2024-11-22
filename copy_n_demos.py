'''
Script that copies the first n demonstrations from a source HDF5 file to a new HDF5 file.
'''

import os
import h5py
import numpy as np
from tqdm import tqdm
import shutil

def copy_n_demos(source_path, n):
    # Extract n from source path and create target path
    dirname = os.path.dirname(source_path)
    filename = os.path.basename(source_path)
    base_name = filename.split('panda')[0]
    suffix = filename.split('panda')[1].split('_', 1)[1]
    target_path = os.path.join(dirname, f"{base_name}panda{n}_{suffix}")
    
    print(f"Copying {source_path} to {target_path}")
    
    # Copy entire file first
    shutil.copy2(source_path, target_path)
    
    # Open target file and delete excess demos
    with h5py.File(target_path, 'r+') as target:
        all_demos = list(target['data'].keys())
        demos_to_delete = []
        
        # Find demos with index >= n
        for demo in all_demos:
            try:
                demo_idx = int(demo.split('_')[1])
                if demo_idx >= n:
                    demos_to_delete.append(demo)
            except (IndexError, ValueError):
                print(f"Warning: Skipping malformed demo name: {demo}")
        
        if demos_to_delete:
            print(f"Deleting {len(demos_to_delete)} demos with index >= {n}")
            
            # Delete excess demos
            for demo in tqdm(demos_to_delete, desc="Deleting excess demos"):
                del target['data'][demo]
    
    print(f"Successfully created dataset with demos 0 to {n-1} at {target_path}")

if __name__ == "__main__":
    import argparse
    import shutil
    
    parser = argparse.ArgumentParser()
    parser.add_argument('source_path', type=str, help='Path to source HDF5 file')
    parser.add_argument('n', type=int, help='Number of demos to keep')
    
    args = parser.parse_args()
    
    copy_n_demos(args.source_path, args.n)