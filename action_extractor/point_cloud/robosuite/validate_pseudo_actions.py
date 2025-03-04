#!/usr/bin/env python

"""
validate_pseudo_actions.py

Usage:
    python validate_pseudo_actions.py --file1 /path/to/original.hdf5 --file2 /path/to/pseudo_labeled.hdf5

This script compares the 'actions' dataset in two HDF5 files and measures:
 - Position differences (first 3 dims)
 - Orientation differences (next 3 dims, axis-angle form)
We ignore the last dimension (often a gripper).
We also normalize each difference by the average magnitude of that quantity
in file1, so we can gauge how large the difference is relative to typical scale.

We produce per-demo metrics and a global summary.
"""

import argparse
import h5py
import numpy as np

def compute_position_orientation_differences(actions1, actions2):
    """
    Compare position (first 3 dims) and orientation (next 3 dims)
    between two actions arrays of shape (T, >=6).

    Returns a dict with:
        pos_mean_l2, pos_max_l2, pos_ratio (pos_mean_l2 / average pos magnitude in file1)
        ori_mean_l2, ori_max_l2, ori_ratio (ori_mean_l2 / average ori magnitude in file1)
        ...
    We ignore the final dimension if it exists.
    """

    # 1) Ensure same # of steps
    T1 = actions1.shape[0]
    T2 = actions2.shape[0]
    T_common = min(T1, T2)

    # Slice if lengths differ
    a1 = actions1[:T_common]
    a2 = actions2[:T_common]

    # 2) Extract relevant dims:
    #    pos = first 3 dims, ori = next 3 dims
    #    ignore the rest (e.g., dimension 7 for gripper)
    pos1, pos2 = a1[:, :3], a2[:, :3]
    ori1, ori2 = a1[:, 3:6], a2[:, 3:6]

    # 3) Compute L2 differences
    # Position
    pos_diff = pos1 - pos2  # (T_common, 3)
    pos_l2_per_step = np.linalg.norm(pos_diff, axis=-1)
    pos_mean_l2 = float(np.mean(pos_l2_per_step))
    pos_max_l2 = float(np.max(pos_l2_per_step))

    # Orientation (axis-angle)
    ori_diff = ori1 - ori2  # (T_common, 3)
    ori_l2_per_step = np.linalg.norm(ori_diff, axis=-1)
    ori_mean_l2 = float(np.mean(ori_l2_per_step))
    ori_max_l2 = float(np.max(ori_l2_per_step))

    # 4) Compute average magnitude of pos, ori in file1 for scale
    pos_mag_1 = np.linalg.norm(pos1, axis=-1)  # shape (T_common,)
    ori_mag_1 = np.linalg.norm(ori1, axis=-1)

    # If the magnitudes are near zero, watch out for divide by zero.
    # We'll do a small epsilon clamp if needed.
    pos_scale = float(np.mean(pos_mag_1))
    ori_scale = float(np.mean(ori_mag_1))
    epsilon = 1e-9
    if pos_scale < epsilon:
        pos_scale = epsilon
    if ori_scale < epsilon:
        ori_scale = epsilon

    pos_ratio = pos_mean_l2 / pos_scale
    ori_ratio = ori_mean_l2 / ori_scale

    return {
        "pos_mean_l2": pos_mean_l2,
        "pos_max_l2":  pos_max_l2,
        "pos_ratio":   pos_ratio,    # ratio of mean L2 to average magnitude
        "ori_mean_l2": ori_mean_l2,
        "ori_max_l2":  ori_max_l2,
        "ori_ratio":   ori_ratio,
        "len_file1":   T1,
        "len_file2":   T2,
        "len_cmp":     T_common
    }

def main():
    parser = argparse.ArgumentParser(description="Compare 'actions' between two HDF5 datasets, focusing on position & orientation.")
    parser.add_argument("--file1", type=str, required=True,
                        help="Path to the first HDF5 file (reference/original).")
    parser.add_argument("--file2", type=str, required=True,
                        help="Path to the second HDF5 file (pseudo-labeled).")
    args = parser.parse_args()

    # Open files
    f1 = h5py.File(args.file1, "r")
    f2 = h5py.File(args.file2, "r")

    demos1 = set(f1["data"].keys())
    demos2 = set(f2["data"].keys())

    # Find demos in common
    common_demos = sorted(demos1.intersection(demos2), key=lambda d: int(d.split("_")[-1]))
    if not common_demos:
        print("No demos in common between the two files!")
        f1.close()
        f2.close()
        return

    # We'll aggregate global stats for position and orientation across all demos
    global_pos_mean_l2 = 0.0
    global_pos_max_l2  = 0.0
    global_pos_ratio   = 0.0

    global_ori_mean_l2 = 0.0
    global_ori_max_l2  = 0.0
    global_ori_ratio   = 0.0

    total_steps_compared = 0

    print(f"Comparing the following demos: {common_demos}")
    for demo in common_demos:
        actions1 = f1["data"][demo]["actions"][()]
        actions2 = f2["data"][demo]["actions"][()]

        # Skip if these are too short or not 6D+ actions
        if actions1.shape[1] < 6 or actions2.shape[1] < 6:
            print(f"Warning: {demo} doesn't have at least 6 action dims; skipping.")
            continue

        metrics = compute_position_orientation_differences(actions1, actions2)

        # Print per-demo
        print(f"\nDemo {demo}:")
        print(f"  File1 length:       {metrics['len_file1']} steps")
        print(f"  File2 length:       {metrics['len_file2']} steps")
        print(f"  Compared length:    {metrics['len_cmp']} steps")

        print(f"  Pos mean L2:        {metrics['pos_mean_l2']:.6f}")
        print(f"  Pos max L2:         {metrics['pos_max_l2']:.6f}")
        print(f"  Pos ratio:          {metrics['pos_ratio']:.6f}  (mean L2 / avg pos magnitude in file1)")

        print(f"  Ori mean L2:        {metrics['ori_mean_l2']:.6f}")
        print(f"  Ori max L2:         {metrics['ori_max_l2']:.6f}")
        print(f"  Ori ratio:          {metrics['ori_ratio']:.6f}  (mean L2 / avg ori magnitude in file1)")

        # We'll do a weighted sum of each mean-l2 metric * #steps, so we can get an overall average
        step_count = metrics['len_cmp']
        total_steps_compared += step_count

        # Weighted accumulation for "mean" metrics
        global_pos_mean_l2 += metrics["pos_mean_l2"] * step_count
        global_pos_ratio   += metrics["pos_ratio"]   * step_count

        global_ori_mean_l2 += metrics["ori_mean_l2"] * step_count
        global_ori_ratio   += metrics["ori_ratio"]   * step_count

        # For max, we just track the max across all demos
        if metrics["pos_max_l2"] > global_pos_max_l2:
            global_pos_max_l2 = metrics["pos_max_l2"]
        if metrics["ori_max_l2"] > global_ori_max_l2:
            global_ori_max_l2 = metrics["ori_max_l2"]

    # Final aggregated summary
    if total_steps_compared > 0:
        agg_pos_mean_l2 = global_pos_mean_l2 / total_steps_compared
        agg_pos_ratio   = global_pos_ratio   / total_steps_compared

        agg_ori_mean_l2 = global_ori_mean_l2 / total_steps_compared
        agg_ori_ratio   = global_ori_ratio   / total_steps_compared
    else:
        # no demos or no steps
        agg_pos_mean_l2 = 0.0
        agg_pos_ratio   = 0.0
        agg_ori_mean_l2 = 0.0
        agg_ori_ratio   = 0.0

    print("\n=== GLOBAL SUMMARY (across all common demos) ===")
    print(f"Compared a total of {total_steps_compared} steps (summed across demos).")
    print(f"Pos mean L2 (avg):        {agg_pos_mean_l2:.6f}")
    print(f"Pos max L2 (max across):  {global_pos_max_l2:.6f}")
    print(f"Pos ratio (avg):          {agg_pos_ratio:.6f}")

    print(f"Ori mean L2 (avg):        {agg_ori_mean_l2:.6f}")
    print(f"Ori max L2 (max across):  {global_ori_max_l2:.6f}")
    print(f"Ori ratio (avg):          {agg_ori_ratio:.6f}")

    f1.close()
    f2.close()

if __name__ == "__main__":
    main()
