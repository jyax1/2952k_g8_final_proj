#!/usr/bin/env python

import argparse
import os

import h5py
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D projection

os.makedirs("debug", exist_ok=True)

# ----------------------------------------------------------------
# Helper: Convert quaternion [w, x, y, z] or [x, y, z, w] to (axis * angle)
# We'll assume your data is [qx, qy, qz, qw], so reorder if needed.
# ----------------------------------------------------------------
def quat2axisangle(q):
    """
    Convert quaternion [qw, qx, qy, qz] => 3D axis * angle in R^3
    If your dataset is [qx, qy, qz, qw], reorder accordingly.
    """
    x, y, z, w = q
    eps = 1e-12
    norm_q = np.linalg.norm(q)
    if norm_q < eps:
        return np.zeros(3, dtype=np.float32)
    q_normalized = q / norm_q
    x, y, z, w = q_normalized
    angle = 2.0 * np.arccos(w)
    s = np.sqrt(1.0 - w*w)
    if s < eps:
        # angle near zero => axis can be anything
        return np.zeros(3, dtype=np.float32)
    axis = np.array([x, y, z], dtype=np.float32) / s
    return axis * angle

# ----------------------------------------------------------------
# Load data from an HDF5 file
# ----------------------------------------------------------------
def load_demo_data_from_hdf5(hdf5_path):
    """
    Loads from an HDF5 file with the assumed structure:
      root['data'][demo][...]
    where each demo has:
      - 'actions': shape (T,7) => [dx, dy, dz, rx, ry, rz, gripper]
      - 'obs/robot0_eef_pos': shape (T,3)
      - 'obs/robot0_eef_quat': shape (T,4), presumed [qx, qy, qz, qw]

    Returns:
      abs_positions       : Nx3
      abs_orientations_3d : Nx3   (quaternion -> axis*angle)
      rel_positions       : Nx3
      rel_orientations    : Nx3
      rel_gripper         : Nx1
      hdf5_colors         : Nx3   (colors for each point)
    """
    abs_positions       = []
    abs_orientations_3d = []
    rel_positions       = []
    rel_orientations    = []
    rel_gripper         = []
    hdf5_colors         = []

    # Just one file => define a color palette for each demo or a single color
    # Let's define a different color for each demo to see them distinctly
    with h5py.File(hdf5_path, 'r') as f:
        if 'data' not in f:
            raise KeyError(f"Expected 'data' group in {hdf5_path}, not found!")

        demos = list(f['data'].keys())
        num_demos = len(demos)
        color_map = sns.color_palette("husl", num_demos)

        for i_demo, demo_name in enumerate(demos):
            data_group = f['data'][demo_name]

            # Get actions
            if 'actions' not in data_group:
                print(f"Demo {demo_name} missing 'actions'. Skipping.")
                continue

            actions = data_group['actions'][:]  # shape (T,7)

            # Observations
            obs_group = data_group.get('obs', None)
            if obs_group is None:
                print(f"Demo {demo_name} missing 'obs' group. Skipping.")
                continue

            pos = obs_group['robot0_eef_pos'][:]     # shape (T,3)
            quat = obs_group['robot0_eef_quat'][:]   # shape (T,4), [qx,qy,qz,qw]

            T = actions.shape[0]

            if T != pos.shape[0] or T != quat.shape[0]:
                print(f"Demo {demo_name} has inconsistent shapes. Skipping.")
                continue

            # ----------------------------------------------------------------
            # Absolute data:
            # ----------------------------------------------------------------
            abs_positions.append(pos)

            # Convert each quaternion row to 3D axis-angle
            abs_ori_3d = np.zeros((T,3), dtype=np.float32)
            for i_frame in range(T):
                qxyzw = quat[i_frame]
                # reorder if needed => [x,y,z,w]
                abs_ori_3d[i_frame] = quat2axisangle(qxyzw)
            abs_orientations_3d.append(abs_ori_3d)

            # ----------------------------------------------------------------
            # Relative/delta data:
            # ----------------------------------------------------------------
            # actions shape (T,7) => [dx, dy, dz, rx, ry, rz, gripper]
            delta_pos = actions[:, 0:3]
            delta_ori = actions[:, 3:6]
            delta_grip= actions[:, 6]

            rel_positions.append(delta_pos)
            rel_orientations.append(delta_ori)
            rel_gripper.append(delta_grip.reshape(-1,1))

            # ----------------------------------------------------------------
            # Colors
            # ----------------------------------------------------------------
            # Color for this entire demo
            file_color = color_map[i_demo]
            hdf5_colors.extend([file_color]*T)

    # Concatenate all
    abs_positions       = np.concatenate(abs_positions, axis=0) if abs_positions else np.zeros((0,3))
    abs_orientations_3d = np.concatenate(abs_orientations_3d, axis=0) if abs_orientations_3d else np.zeros((0,3))
    rel_positions       = np.concatenate(rel_positions, axis=0) if rel_positions else np.zeros((0,3))
    rel_orientations    = np.concatenate(rel_orientations, axis=0) if rel_orientations else np.zeros((0,3))
    rel_gripper         = np.concatenate(rel_gripper, axis=0) if rel_gripper else np.zeros((0,1))
    hdf5_colors         = np.array(hdf5_colors)

    # Print some stats
    print("\n--- Stats: Absolute Positions ---")
    if len(abs_positions) > 0:
        print(f" X in [{abs_positions[:,0].min()}, {abs_positions[:,0].max()}]")
        print(f" Y in [{abs_positions[:,1].min()}, {abs_positions[:,1].max()}]")
        print(f" Z in [{abs_positions[:,2].min()}, {abs_positions[:,2].max()}]")

    print("\n--- Stats: Relative Positions (delta) ---")
    if len(rel_positions) > 0:
        print(f" dX in [{rel_positions[:,0].min()}, {rel_positions[:,0].max()}]")
        print(f" dY in [{rel_positions[:,1].min()}, {rel_positions[:,1].max()}]")
        print(f" dZ in [{rel_positions[:,2].min()}, {rel_positions[:,2].max()}]")

    return (abs_positions, abs_orientations_3d,
            rel_positions, rel_orientations, rel_gripper,
            hdf5_colors)

# ----------------------------------------------------------------
# Visualization
# ----------------------------------------------------------------
def visualize_data_distributions(
    abs_positions, abs_orientations_3d,
    rel_positions, rel_orientations, rel_gripper,
    hdf5_colors, 
    task_name="some_task",
    save_image=True
):
    """
    Create a 2x3 figure:
      Row1 => (Abs pos 3D scatter, Abs ori 3D scatter, Abs gripper hist)
      Row2 => (Rel pos 3D scatter, Rel ori 3D scatter, Rel gripper hist)
    """
    fig = plt.figure(figsize=(16,10))

    # For consistent axis limits, let's get range 
    # We'll do separate for abs pos, rel pos, abs ori, rel ori
    pos_limit_abs = np.max(np.abs(abs_positions)) * 1.05 if len(abs_positions) > 0 else 1.0
    ori_limit_abs = np.max(np.abs(abs_orientations_3d)) * 1.05 if len(abs_orientations_3d) > 0 else 1.0
    pos_limit_rel = np.max(np.abs(rel_positions)) * 1.05 if len(rel_positions) > 0 else 1.0
    ori_limit_rel = np.max(np.abs(rel_orientations)) * 1.05 if len(rel_orientations) > 0 else 1.0

    # Row1 col1 => 3D scatter of abs pos
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    if len(abs_positions)>0:
        ax1.scatter(abs_positions[:,0], abs_positions[:,1], abs_positions[:,2],
                    c=hdf5_colors, s=5)
        ax1.set_title("Absolute EEF Positions")
        ax1.set_xlim([-pos_limit_abs, pos_limit_abs])
        ax1.set_ylim([-pos_limit_abs, pos_limit_abs])
        ax1.set_zlim([-pos_limit_abs, pos_limit_abs])
    else:
        ax1.text(0.5,0.5,0.5, "No abs pos data", horizontalalignment='center')
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")

    # Row1 col2 => 3D scatter of abs ori
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    if len(abs_orientations_3d)>0:
        ax2.scatter(abs_orientations_3d[:,0], abs_orientations_3d[:,1], abs_orientations_3d[:,2],
                    c=hdf5_colors, s=5)
        ax2.set_title("Absolute Orientation (axis-angle)")
        ax2.set_xlim([-ori_limit_abs, ori_limit_abs])
        ax2.set_ylim([-ori_limit_abs, ori_limit_abs])
        ax2.set_zlim([-ori_limit_abs, ori_limit_abs])
    else:
        ax2.text(0.5,0.5,0.5, "No abs ori data", horizontalalignment='center')
    ax2.set_xlabel("Ori X")
    ax2.set_ylabel("Ori Y")
    ax2.set_zlabel("Ori Z")

    # Row1 col3 => we don't have absolute gripper
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.text(0.2,0.5, "No absolute gripper,\nonly relative in data", fontsize=12)
    ax3.set_title("Absolute Gripper? Not in 'obs'")
    ax3.set_axis_off()

    # Row2 col1 => 3D scatter rel pos
    ax4 = fig.add_subplot(2,3,4, projection='3d')
    if len(rel_positions)>0:
        ax4.scatter(rel_positions[:,0], rel_positions[:,1], rel_positions[:,2],
                    c=hdf5_colors, s=5)
        ax4.set_title("Relative Positions (delta)")
        ax4.set_xlim([-pos_limit_rel, pos_limit_rel])
        ax4.set_ylim([-pos_limit_rel, pos_limit_rel])
        ax4.set_zlim([-pos_limit_rel, pos_limit_rel])
    else:
        ax4.text(0.5,0.5,0.5, "No rel pos data", horizontalalignment='center')
    ax4.set_xlabel("dX")
    ax4.set_ylabel("dY")
    ax4.set_zlabel("dZ")

    # Row2 col2 => 3D scatter rel ori
    ax5 = fig.add_subplot(2,3,5, projection='3d')
    if len(rel_orientations)>0:
        ax5.scatter(rel_orientations[:,0], rel_orientations[:,1], rel_orientations[:,2],
                    c=hdf5_colors, s=5)
        ax5.set_title("Relative Orientation (axis-angle)")
        ax5.set_xlim([-ori_limit_rel, ori_limit_rel])
        ax5.set_ylim([-ori_limit_rel, ori_limit_rel])
        ax5.set_zlim([-ori_limit_rel, ori_limit_rel])
    else:
        ax5.text(0.5,0.5,0.5, "No rel ori data", horizontalalignment='center')
    ax5.set_xlabel("dOri X")
    ax5.set_ylabel("dOri Y")
    ax5.set_zlabel("dOri Z")

    # Row2 col3 => histogram rel gripper
    ax6 = fig.add_subplot(2,3,6)
    if len(rel_gripper)>0:
        ax6.hist(rel_gripper, bins=20, color='b', alpha=0.7)
        ax6.set_title("Relative Gripper Motion")
        ax6.set_xlabel("Delta Gripper (Open/Close)")
        ax6.set_ylabel("Frequency")
    else:
        ax6.text(0.2,0.5, "No rel gripper data", fontsize=12)
        ax6.set_axis_off()

    plt.tight_layout()

    if save_image:
        file_path = os.path.join("debug", f"{task_name}_action_distribution.png")
        plt.savefig(file_path)
        print(f"Saved distribution plot to {file_path}")
    # plt.show()  # Uncomment if you want an interactive window

# ----------------------------------------------------------------
# Main function
# ----------------------------------------------------------------
def main(hdf5_path):
    """
    High-level function to load data from a single .hdf5 file,
    visualize, and optionally save a figure.
    """
    # Derive a "task name" from the file name (just for labeling)
    task_name = os.path.splitext(os.path.basename(hdf5_path))[0]

    (abs_positions, abs_orientations_3d,
     rel_positions, rel_orientations, rel_gripper,
     hdf5_colors) = load_demo_data_from_hdf5(hdf5_path)

    visualize_data_distributions(
        abs_positions, abs_orientations_3d,
        rel_positions, rel_orientations, rel_gripper,
        hdf5_colors,
        task_name=task_name,
        save_image=True
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize data from a single .hdf5 file.")
    parser.add_argument("hdf5_path", type=str, help="Path to the .hdf5 file.")
    args = parser.parse_args()

    if not os.path.isfile(args.hdf5_path):
        raise FileNotFoundError(f"File not found: {args.hdf5_path}")

    main(args.hdf5_path)
