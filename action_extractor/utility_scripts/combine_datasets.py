#!/usr/bin/env python3
import argparse
import h5py
import random

def combine_datasets(dataset1_path, dataset2_path, num_demos_to_add, output_path):
    with h5py.File(dataset1_path, 'r') as f1, \
         h5py.File(dataset2_path, 'r') as f2, \
         h5py.File(output_path, 'w') as f_out:
    
        # Check that both files have a "data" group
        if "data" not in f1 or "data" not in f2:
            raise ValueError("Both datasets must have a 'data' group.")

        # Get sorted demo keys (assumes groups are named "demo_0", "demo_1", etc.)
        demos2 = list(f2["data"].keys())
        # demos_success = json.loads(f2["data"].attrs["demos_success"])
        demo_lengths = []
        for demo in demos2:
            actions = f2["data"][demo]["actions"]
            traj_length = len(actions)
            # success =demos_success[demo]
            demo_lengths.append((demo, traj_length))
            
        if len(demo_lengths) < num_demos_to_add:
            raise ValueError("Not enough demos in dataset2 to sample from.")

        selected_demos = random.sample(demo_lengths, num_demos_to_add)
        
        # Create output file by copying the entire "data" group from dataset1
        f1.copy("data", f_out, name="data")
        data_out = f_out["data"]
        
        existing_demos = list(data_out.keys())
        starting_index = len(existing_demos)

        for i, (demo_key, traj_length) in enumerate(selected_demos):
            new_demo_name = f"demo_{starting_index + i}"
            print(f"Adding {demo_key} as {new_demo_name} with trajectory length: {traj_length}")
            f2.copy(f"data/{demo_key}", data_out, name=new_demo_name)
            
        total_samples = 0
        for demo in data_out.keys():
            total_samples += data_out[demo].attrs["num_samples"]
            
        data_out.attrs["total"] = total_samples
        print(f"New total num_samples: {total_samples}; new number of demos: {len(data_out.keys())}")
    
    print(f"Combined dataset saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Combine two HDF5 datasets by adding demos from dataset2 to a copy of dataset1."
    )
    parser.add_argument("--dataset1", type=str, required=True, help="Path to the first (base) dataset (HDF5 file)")
    parser.add_argument("--dataset2", type=str, required=True, help="Path to the second dataset (HDF5 file)")
    parser.add_argument("--num_demos_to_add", type=int, required=True,
                    help="Number of demos to add from dataset2")
    parser.add_argument("--output", type=str, required=True, help="Path for the output combined dataset (HDF5 file)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (optional)")

    
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
    combine_datasets(args.dataset1, args.dataset2, args.num_demos_to_add, args.output)

if __name__ == "__main__":
    main()
