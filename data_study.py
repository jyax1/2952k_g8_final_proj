import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import seaborn as sns
import os


from datasets import DatasetVideo2DeltaAction

os.makedirs("debug", exist_ok=True)

def load_demo_data(dataset):
    """ Load all actions from the entire dataset demo by demo. """
    delta_positions = []
    delta_orientations = []
    delta_gripper = []
    hdf5_colors = []
    
    # Color palette for different files
    unique_files = list(set([zarr_file.split('/')[-1] for zarr_file in dataset.zarr_files]))
    color_map = sns.color_palette("husl", len(unique_files))  # Create a color palette

    # Iterate through each demo
    for zarr_file, root in zip(dataset.zarr_files, dataset.roots):
        print(f"Processing demo: {zarr_file}")
        task = zarr_file.split("/")[-2].replace('_', ' ')
        
        demos = list(root['data'].keys())  # All demos in the current zarr file
        for demo in demos:
            data = root['data'][demo]
            actions = data['actions'][:]  # Load all actions for this demo at once

            # Extract delta positions, orientations, and gripper motions
            delta_pos = actions[:, :3]  # First 3 dimensions
            delta_ori = actions[:, 3:6]  # Next 3 dimensions (orientation)
            delta_grip = actions[:, 6]   # Last dimension (gripper)

            delta_positions.append(delta_pos)
            delta_orientations.append(delta_ori)
            delta_gripper.append(delta_grip)

            # Color code each demo based on the corresponding zarr file
            color_idx = unique_files.index(zarr_file.split('/')[-1])
            demo_color = [color_map[color_idx]] * len(actions)
            hdf5_colors.extend(demo_color)

    # Convert lists to NumPy arrays
    delta_positions = np.concatenate(delta_positions, axis=0)
    delta_orientations = np.concatenate(delta_orientations, axis=0)
    delta_gripper = np.concatenate(delta_gripper, axis=0)
    hdf5_colors = np.array(hdf5_colors)
    
    return delta_positions, delta_orientations, delta_gripper, hdf5_colors

def visualize_action_distributions(delta_positions, delta_orientations, delta_gripper, hdf5_colors, task_name, save_image=False):
    fig = plt.figure(figsize=(16, 6))

    # 3D scatter plot for delta positions (x, y, z)
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(delta_positions[:, 0], delta_positions[:, 1], delta_positions[:, 2], c=hdf5_colors, s=20)
    ax1.set_title("Delta Positions (x, y, z)")
    ax1.set_xlabel("Delta X")
    ax1.set_ylabel("Delta Y")
    ax1.set_zlabel("Delta Z")

    # 3D scatter plot for delta orientations (axis-angle representation)
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(delta_orientations[:, 0], delta_orientations[:, 1], delta_orientations[:, 2], c=hdf5_colors, s=20)
    ax2.set_title("Delta Orientations (axis-angle)")
    ax2.set_xlabel("Orientation X")
    ax2.set_ylabel("Orientation Y")
    ax2.set_zlabel("Orientation Z")

    # 1D histogram for delta gripper motion (open/close)
    ax3 = fig.add_subplot(133)
    ax3.hist(delta_gripper, bins=20, color='b', alpha=0.7)
    ax3.set_title("Delta Gripper Motion")
    ax3.set_xlabel("Gripper Motion (Open/Close)")
    ax3.set_ylabel("Frequency")

    plt.tight_layout()

    # Optionally save the plot to the debug directory
    if save_image:
        file_path = os.path.join("debug", f"{task_name}_action_distribution.png")
        plt.savefig(file_path)
        print(f"Saved action distribution plot for task: {task_name} to {file_path}")


def process_subdirectory(subdir_path):
    dataset = DatasetVideo2DeltaAction(path=subdir_path)
    delta_positions, delta_orientations, delta_gripper, hdf5_colors = load_demo_data(dataset)
    
    # Extract the last part of the subdirectory path to use as a task name
    task_name = subdir_path.rsplit('/', 1)[-1]
    
    visualize_action_distributions(delta_positions, delta_orientations, delta_gripper, hdf5_colors, task_name, save_image=True)

if __name__ == '__main__':
    base_path = '/home/yilong/Documents/ae_data/datasets/mimicgen_core'
    
    # Iterate over every subdirectory in the base_path
    for subdir in os.listdir(base_path):
        subdir_path = os.path.join(base_path, subdir)
        
        # Check if the path is a directory
        if os.path.isdir(subdir_path):
            print(f"Processing subdirectory: {subdir_path}")
            process_subdirectory(subdir_path)