import numpy as np
import open3d as o3d

from robosuite.utils.camera_utils import get_real_depth_map, get_camera_extrinsic_matrix, get_camera_intrinsic_matrix
from robomimic.utils.obs_utils import undiscretize_depth
from robomimic.envs.env_robosuite import depth2fgpcd, np2o3d

# -----------------------------------------------------------
# Main function to reconstruct point clouds from obs_group
# -----------------------------------------------------------

def reconstruct_pointclouds_from_obs_group(
    obs_group,
    env,
    camera_names,
    camera_height,
    camera_width,
    max_points=1000000,
    verbose=False
):
    """
    Reconstructs per-timestep point clouds from raw images+depth in 'obs_group'.
    Returns exactly the format you previously stored as:
       point_clouds_points = [points for points in obs_group["pointcloud_points"]]
       point_clouds_colors = [colors for colors in obs_group["pointcloud_colors"]]

    Args:
        obs_group (h5py.Group): e.g. root["data"][demo]["obs"], containing keys like
            "<camera>_image" and "<camera>_depth".
        env: The environment object used in get_observation. Must provide:
            - env.camera_names
            - env.camera_heights
            - env.camera_widths
            - env.sim
            - use_camera_obs (bool)
        max_points (int): Maximum number of points per cloud (padded or truncated).
        verbose (bool): Print debug info if True.

    Returns:
        point_clouds_points (list of (M, 3) float32):
            Each element is a 2D array of 3D points for one timestep.
        point_clouds_colors (list of (M, 3) uint8):
            Each element is a 2D array of colors for that timestep, matching the points.
    """

    # We replicate the workspace logic from get_observation
    center = np.array([0, 0, 0.7])
    ws_size = 0.8
    # workspace shape (3,2): min and max for x,y,z
    # but your code structured it differently, so let's keep consistent:
    workspace = np.array([
        [center[0] - ws_size/2, center[0] + ws_size/2],
        [center[1] - ws_size/2, center[1] + ws_size/2],
        [center[2],             center[2] + ws_size]
    ])

    # We determine how many timesteps are in the dataset from the first camera's image
    first_cam = camera_names[0]
    # e.g. if the key is "<camera>_image"
    num_samples = obs_group[f"{first_cam}_image"].shape[0]
    if verbose:
        print(f"Found {num_samples} timesteps in obs_group for camera: {first_cam}")

    def pad_or_truncate(arr, target_size):
        if len(arr) > target_size:
            return arr[:target_size]
        elif len(arr) < target_size:
            return np.pad(arr, ((0, target_size - len(arr)), (0, 0)), mode='constant')
        return arr

    point_clouds_points = []
    point_clouds_colors = []

    # For each timestep, combine all cameras into a single point cloud
    for i in range(num_samples):
        all_pcds = o3d.geometry.PointCloud()

        # For each camera, replicate the logic from get_observation
        for cam_idx, camera_name in enumerate(camera_names):
            # 1) Retrieve color & depth from obs_group
            color = obs_group[f"{camera_name}_image"][i]
            depth = obs_group[f"{camera_name}_depth"][i]  # shape (H,W) or (H,W,1)

            # Possibly compute real-depth map
            depth = undiscretize_depth(depth, f"{camera_name}_depth")
            # If depth is shape (H,W,1), flatten to (H,W)
            if depth.ndim == 3 and depth.shape[2] == 1:
                depth = depth[..., 0]

            # 2) Camera intrinsics
            int_mat = get_camera_intrinsic_matrix(env.sim, camera_name, camera_height, camera_width)
            fx, fy, cx, cy = int_mat[0,0], int_mat[1,1], int_mat[0,2], int_mat[1,2]

            # 3) Unproject to 3D in camera coords
            mask = np.ones_like(depth, dtype=bool)
            pcd_cam = depth2fgpcd(depth, mask, [fx, fy, cx, cy])  # shape (N,3)

            # 4) Transform to world
            ext_mat = get_camera_extrinsic_matrix(env.sim, camera_name)
            # same logic as your snippet
            ones = np.ones((pcd_cam.shape[0], 1), dtype=pcd_cam.dtype)
            pcd_cam_hom = np.concatenate([pcd_cam, ones], axis=1).T  # (4,N)
            trans_pcd = (ext_mat @ pcd_cam_hom)[:3].T  # shape (N,3)

            # 5) Filter by workspace
            mask_ws = (
                (trans_pcd[:,0] > workspace[0,0]) & (trans_pcd[:,0] < workspace[0,1]) &
                (trans_pcd[:,1] > workspace[1,0]) & (trans_pcd[:,1] < workspace[1,1]) &
                (trans_pcd[:,2] > workspace[2,0]) & (trans_pcd[:,2] < workspace[2,1])
            )
            # trans_pcd = trans_pcd[mask_ws]

            # Extract color for these points
            color_flat = color.reshape(-1, 3)[mask.flatten()]
            # color_flat = color_flat[mask_ws]
            # scale color to [0,1]
            color_float = color_flat.astype(np.float64) / 255.0

            # Build partial pcd
            pcd_o3d = np2o3d(trans_pcd, color_float)

            all_pcds += pcd_o3d

        # After all cameras, convert to np arrays
        points_np = np.asarray(all_pcds.points, dtype=np.float32)
        colors_np = (np.asarray(all_pcds.colors) * 255).astype(np.uint8)

        # Pad/truncate
        points_np = pad_or_truncate(points_np, max_points)
        colors_np = pad_or_truncate(colors_np, max_points)

        point_clouds_points.append(points_np)
        point_clouds_colors.append(colors_np)

    if verbose:
        print(f"Constructed {len(point_clouds_points)} point clouds, each up to {max_points} points.")

    return point_clouds_points, point_clouds_colors