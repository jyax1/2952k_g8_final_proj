from action_extractor.utils.angles_utils import *

def get_4x4_poses(pos_array, quat_array):
    """
    Given arrays of positions and quaternions, returns a list of 4x4 poses in SE(3).
    
    'pos_array' is expected to have shape (N, 3)
    'quat_array' is expected to have shape (N, 4) in the order [qx, qy, qz, qw].
    """
    
    # 3) Build 4x4 transforms
    all_poses = []
    for (px, py, pz), (qx, qy, qz, qw) in zip(pos_array, quat_array):
        # Convert quaternion to rotation
        R = quaternion_to_rotation_matrix(qx, qy, qz, qw)
        # Create 4x4 pose
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = [px, py, pz]
        all_poses.append(T)

    return all_poses

def load_ground_truth_poses(obs_group):
    """
    Given an HDF5 group with 'robot0_eef_pos' and 'robot0_eef_quat', 
    returns a list of 4x4 poses in SE(3).
    
    'robot0_eef_pos' is expected to have shape (N, 3)
    'robot0_eef_quat' is expected to have shape (N, 4) in the order [qx, qy, qz, qw].
    """
    
    # 1) Load arrays from the HDF5 group
    pos_array = obs_group["robot0_eef_pos"][:]    # shape (N, 3)
    quat_array = obs_group["robot0_eef_quat"][:]  # shape (N, 4) => [qx, qy, qz, qw]

    return get_4x4_poses(pos_array, quat_array)