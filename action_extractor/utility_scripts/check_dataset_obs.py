#!/usr/bin/env python

import h5py
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Open an .hdf5 file and list the keys in f['data']['demo_0']['obs']"
    )
    parser.add_argument("hdf5_path", type=str, help="Path to the .hdf5 file")
    args = parser.parse_args()

    with h5py.File(args.hdf5_path, "r") as f:
        # Navigate to the group of interest
        obs_group = f["data"]["demo_0"]["obs"]
        
        # Print each key
        print("Keys in f['data']['demo_0']['obs']:")
        for key in obs_group.keys():
            print("  ", key)

if __name__ == "__main__":
    main()
