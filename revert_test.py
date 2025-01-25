#!/usr/bin/env python3

"""
revert_append_last_action.py

Given an existing HDF5 file where each demo had 1 extra action row appended,
this script removes that last row, restoring the original (N-1, D) shape
for each demo.

Implementation details:
  1) For each demo in /data, read the current "actions" dataset as a NumPy array.
  2) If there's at least one row, drop the last row.
  3) Delete the old dataset and re-create it without that row.

Usage:
  python revert_append_last_action.py --hdf5-path /path/to/file.hdf5
"""

import os
import h5py
import numpy as np
import argparse

def revert_last_action_inplace(hdf5_path: str):
    """
    For each demo's 'actions' in hdf5, remove the last row if it exists.
    """

    with h5py.File(hdf5_path, "r+") as f:
        demos = list(f["data"].keys())
        print(f"[revert_last_action_inplace] Found {len(demos)} demos in {hdf5_path}. Removing last action row for each...")

        for demo in demos:
            actions_ds = f["data"][demo]["actions"]
            old_actions = actions_ds[:]  # shape (M, D)

            # If no data or only 1 row, skipping might be safer:
            if old_actions.shape[0] <= 1:
                print(f"  Demo '{demo}' has {old_actions.shape[0]} rows, skipping removal.")
                continue

            # Drop the last row
            new_actions = old_actions[:-1, :]  # shape (M-1, D)

            # Remove old dataset
            del f["data"][demo]["actions"]

            # Create new dataset with shape (M-1, D)
            f["data"][demo].create_dataset(
                "actions",
                data=new_actions,
                compression="gzip",  # optional
            )

    print("[revert_last_action_inplace] Done removing last action row for each demo.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Revert last appended action row from HDF5 demos.")
    parser.add_argument("--hdf5-path", type=str, required=True,
                        help="Path to the HDF5 file to be edited in-place.")
    args = parser.parse_args()

    if not os.path.isfile(args.hdf5_path):
        raise FileNotFoundError(f"File not found: {args.hdf5_path}")

    revert_last_action_inplace(args.hdf5_path)