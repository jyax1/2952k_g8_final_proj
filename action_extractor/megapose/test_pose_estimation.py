import h5py
import numpy as np
from pathlib import Path
from action_extractor.utils.dataset_utils import camera_extrinsics, frontview_R
from megapose.datasets.scene_dataset import CameraData
from megapose.lib3d.transform import Transform

# Load data from HDF5
def load_data_sample(hdf5_path, demo_key, camera="frontview", idx=0):
    with h5py.File(hdf5_path, 'r') as root:
        # Get RGB and depth
        rgb = root['data'][demo_key]['obs'][f'{camera}_image'][idx]  # (128, 128, 3)
        depth = root['data'][demo_key]['obs'][f'{camera}_depth'][idx]  # (128, 128, 1)
        
        # Get gripper pose as ground truth
        eef_pos = root['data'][demo_key]['obs']['robot0_eef_pos'][idx]  # (3,)
        eef_quat = root['data'][demo_key]['obs']['robot0_eef_quat'][idx]  # (4,)
        
        return rgb, depth.squeeze(), eef_pos, eef_quat

# Camera intrinsics for 128x128 frontview camera 
frontview_K = np.array([
    [128.0, 0.0, 64.0],  # Focal length & principal point
    [0.0, 128.0, 64.0], 
    [0.0, 0.0, 1.0]
])

# Test the pose estimation
def test_pose_estimation(hdf5_path, mesh_path):
    # Load a sample
    rgb, depth, gt_pos, gt_quat = load_data_sample(hdf5_path, "demo_0")
    
    # Define a bounding box covering most of the image
    bbox = (20, 20, 108, 108)  # (xmin, ymin, xmax, ymax)
    
    # Get pose prediction
    predictions = get_pose_from_inputs(
        rgb=rgb,
        depth=depth,
        intrinsics=frontview_K,
        mesh_path=mesh_path,
        bounding_box=bbox
    )
    
    # Visualize results
    camera_data = CameraData(
        K=frontview_K,
        resolution=(128, 128),
        TWC=Transform(frontview_R)  # Using the frontview extrinsics
    )
    
    output_dir = Path("debug/pose_estimation")
    visualize_pose_estimation(
        rgb=rgb,
        camera_data=camera_data,
        predictions=predictions,
        mesh_path=mesh_path,
        output_dir=output_dir
    )
    
    # Print comparison
    print("Ground truth position:", gt_pos)
    print("Predicted position:", predictions.poses[0][:3, 3].cpu().numpy())

# Run test
mesh_path = "action_extractor/megapose/mesh/panda_hand.obj" 
hdf5_path = "path/to/your/data.hdf5"  # Replace with actual path
test_pose_estimation(hdf5_path, mesh_path)