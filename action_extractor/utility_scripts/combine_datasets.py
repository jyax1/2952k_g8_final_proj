#!/usr/bin/env python3
import os
import argparse
import h5py
import numpy as np

def combine_datasets(dataset1_path, dataset2_path, p, output_path, seed):
    # Open the two input files
    f1 = h5py.File(dataset1_path, 'r')
    f2 = h5py.File(dataset2_path, 'r')
    
    # Check that both files have a "data" group
    if "data" not in f1 or "data" not in f2:
        raise ValueError("Both datasets must have a 'data' group.")
    
    data1 = f1["data"]
    data2 = f2["data"]
    
    # Get sorted demo keys (assumes groups are named "demo_0", "demo_1", etc.)
    demos1 = sorted(list(data1.keys()), key=lambda x: int(x.replace("demo_", "")))
    demos2 = sorted(list(data2.keys()), key=lambda x: int(x.replace("demo_", "")))
    
    if len(demos1) != len(demos2):
        raise ValueError("The two datasets must have the same number of demos.")
    
    n_demos = len(demos1)
    n_replace = int((p / 100.0) * n_demos)
    
    # Set the random seed for reproducibility and randomly choose indices to replace
    np.random.seed(seed)
    indices_to_replace = set(np.random.choice(n_demos, size=n_replace, replace=False))
    
    print(f"Total demos: {n_demos}, replacing {n_replace} demos with demos from dataset2.")
    
    # Create output file by copying the entire "data" group from dataset1
    f_out = h5py.File(output_path, "w")
    f1.copy("data", f_out, name="data")
    data_out = f_out["data"]
    
    # For each demo to replace, delete the group in f_out and copy from dataset2
    for i, demo_key in enumerate(demos1):
        if i in indices_to_replace:
            print(f"Replacing {demo_key} with version from dataset2.")
            del data_out[demo_key]
            # Copy the demo group from dataset2 into the output file's "data" group
            f2.copy(f"data/{demo_key}", data_out, name=demo_key)
    
    # Close all files
    f1.close()
    f2.close()
    f_out.close()
    print(f"Combined dataset saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Combine two HDF5 datasets by replacing p%% of demos from dataset1 with demos from dataset2."
    )
    parser.add_argument("--dataset1", type=str, required=True, help="Path to the first (base) dataset (HDF5 file)")
    parser.add_argument("--dataset2", type=str, required=True, help="Path to the second dataset (HDF5 file)")
    parser.add_argument("--p", type=int, required=True, choices=[10,20,30,40,50,60,70,80,90],
                        help="Percentage of demos to replace (must be one of 10,20,...,90)")
    parser.add_argument("--output", type=str, required=True, help="Path for the output combined dataset (HDF5 file)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    combine_datasets(args.dataset1, args.dataset2, args.p, args.output, args.seed)

if __name__ == "__main__":
    main()
