"""
append_last_action_inplace.py

Given an existing HDF5 file, for each demo:
 - read "actions"
 - append one more row that is a copy of the last action
 - re-save to "actions" with one extra row
Thus, if originally shape was (N-1, 7), it becomes (N, 7).
"""

import h5py
import numpy as np
import os

def append_last_action_inplace(hdf5_path: str):
    # Backup suggestion (optional):
    # import shutil
    # backup_path = hdf5_path + ".bak"
    # shutil.copy(hdf5_path, backup_path)

    with h5py.File(hdf5_path, "r+") as f:
        demos = list(f["data"].keys())
        print(f"Found {len(demos)} demos in {hdf5_path}. Appending last action for each.")

        for demo in demos:
            actions_ds = f["data"][demo]["actions"]
            old_actions = actions_ds[:]  # shape (M, D)
            if old_actions.shape[0] == 0:
                # Edge case: no actions => skip
                print(f"Demo {demo} has 0 actions, skipping.")
                continue

            last_action = old_actions[-1].copy()
            new_actions = np.concatenate(
                [old_actions, last_action[None, :]],
                axis=0
            )  # shape (M+1, D)

            # Remove old dataset
            del f["data"][demo]["actions"]

            # Create new dataset
            f["data"][demo].create_dataset(
                "actions",
                data=new_actions,
                compression="gzip",  # optional
            )

    print("Done appending last actions in place.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5-path", type=str, required=True,
                        help="Path to the HDF5 file to be edited in-place.")
    args = parser.parse_args()

    if not os.path.isfile(args.hdf5_path):
        raise FileNotFoundError(f"File {args.hdf5_path} not found!")

    append_last_action_inplace(args.hdf5_path)