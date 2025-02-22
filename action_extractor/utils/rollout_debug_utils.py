import numpy as np
import open3d as o3d
import os
import copy

def save_point_clouds_as_ply(
    point_clouds_points,
    point_clouds_colors,
    output_dir="output_dir/point_clouds"
):
    """
    Saves each point cloud (points + colors) as a separate .ply file
    in the specified output directory.

    Args:
        point_clouds_points (list of (N,3) arrays): 
            Each entry is an (N,3) float32 array of point coordinates.
        point_clouds_colors (list of (N,3) arrays):
            Each entry is an (N,3) uint8 array of point colors.
        output_dir (str): Directory to save .ply files.
    """
    # Ensure the output folder exists
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over each point cloud
    for i, (pts, cols) in enumerate(zip(point_clouds_points, point_clouds_colors)):
        # Create an Open3D point cloud object
        pcd = o3d.geometry.PointCloud()

        # Convert points to double precision if needed
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))

        # Convert colors from uint8 [0..255] to float [0..1]
        colors_float = cols.astype(np.float32) / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors_float)

        # Construct a filename for each timestep
        filename = os.path.join(output_dir, f"cloud_{i:04d}.ply")

        # Write the point cloud to disk
        o3d.io.write_point_cloud(filename, pcd)
        print(f"Saved {filename}")


def save_pointclouds_with_bbox_as_ply(point_clouds_points,
                                      point_clouds_colors,
                                      poses,
                                      box_dims = np.array([0.063045, 0.204516, 0.091946]),
                                      output_dir="debug/pointcloud_traj"):
    """
    Saves each original point cloud (points + colors), along with an overlaid bounding box
    (drawn as thick red lines formed by additional points), into a single .ply file per cloud.

    Args:
      point_clouds_points : list of np.ndarray of shape (N,3)
      point_clouds_colors : list of np.ndarray of shape (N,3) in [0,1] for R,G,B
      poses               : list of np.ndarray of shape (4,4), each an SE(3) transform
                            placing the bounding box in the global coordinate system
      box_dims            : (3,) array for bounding box (X, Y, Z) dimensions
      output_dir          : directory to save .ply files

    The .ply file will contain:
      - The original point cloud in its original colors
      - Additional points forming "thick lines" along the box edges, colored bright red.
    """

    os.makedirs(output_dir, exist_ok=True)
    half_dims = 0.5 * box_dims

    # We'll sample each of the 12 edges with a certain resolution to appear "thick"
    # in the final point set. Increase N_samples for a denser line.
    N_samples = 15

    # For convenience, define the 8 corners of the box in its local coordinate system:
    #    [±half_dims[0], ±half_dims[1], ±half_dims[2]]
    # Then define the 12 edges as pairs of corners.
    # corners = all 8 sign combinations
    corners_local = np.array([
        [ half_dims[0],  half_dims[1],  half_dims[2]],
        [ half_dims[0],  half_dims[1], -half_dims[2]],
        [ half_dims[0], -half_dims[1],  half_dims[2]],
        [ half_dims[0], -half_dims[1], -half_dims[2]],
        [-half_dims[0],  half_dims[1],  half_dims[2]],
        [-half_dims[0],  half_dims[1], -half_dims[2]],
        [-half_dims[0], -half_dims[1],  half_dims[2]],
        [-half_dims[0], -half_dims[1], -half_dims[2]],
    ])

    # Define edges as pairs of indices into the corners array
    # so we can sample line segments between them.
    edges = [
        (0, 1), (0, 2), (0, 4),  # edges from corner 0
        (7, 6), (7, 5), (7, 3),  # edges from corner 7
        (1, 3), (1, 5),         # edges that connect top face
        (2, 3), (2, 6),         # edges that connect top face
        (4, 5), (4, 6),         # edges that connect bottom face
    ]
    # (Note: there are multiple ways to define the 12 edges, this is one arrangement.)

    # We'll create a helper function to sample along an edge:
    def sample_edge_points(p0, p1, n_samples=10):
        """
        Returns n_samples points uniformly sampled on the line segment from p0 to p1.
        """
        t_vals = np.linspace(0.0, 1.0, n_samples)
        return (1 - t_vals)[:, None] * p0 + t_vals[:, None] * p1

    for i, (pts, cols) in enumerate(zip(point_clouds_points, point_clouds_colors)):
        pose = poses[i]
        # If the pose is identity (or the bounding box doesn't apply), we still show the identity box

        # Decompose pose
        R = pose[:3, :3]  # 3x3 rotation
        t = pose[:3, 3]   # 3x1 translation vector

        # Construct bounding-box line points in the global frame
        # We'll store them in a list, along with bright red color
        bbox_line_points = []
        for (idx0, idx1) in edges:
            corner0 = corners_local[idx0]
            corner1 = corners_local[idx1]
            # Sample along this edge
            local_line = sample_edge_points(corner0, corner1, N_samples)
            # Transform to global: p_global = R * p_local + t
            line_global = (local_line @ R.T) + t
            bbox_line_points.append(line_global)

        bbox_line_points = np.concatenate(bbox_line_points, axis=0)  # shape (~ 12*N_samples, 3)
        # All red color
        bbox_line_colors = np.tile([1.0, 0.0, 0.0], (bbox_line_points.shape[0], 1))

        # Combine with the original cloud
        combined_points = np.vstack((pts, bbox_line_points))
        combined_colors = np.vstack((cols, bbox_line_colors))

        # Save to PLY
        filename = os.path.join(output_dir, f"pointcloud_{i:04d}.ply")
        with open(filename, 'w') as f:
            # ASCII PLY header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(combined_points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")

            # Write out each point
            for (x, y, z), (r, g, b) in zip(combined_points, combined_colors):
                rr = int(255 * r)
                gg = int(255 * g)
                bb = int(255 * b)
                f.write(f"{x} {y} {z} {rr} {gg} {bb}\n")

        print(f"Saved {filename}")
        
def save_hand_poses(all_hand_poses, filename="all_hand_poses.npy"):
    """
    Saves the Nx4x4 array of poses to a .npy file.
    """
    np.save(filename, all_hand_poses)
    print(f"Saved poses to {filename}")
    
def render_model_on_pointclouds(point_clouds_points, point_clouds_colors, poses, model, 
                                output_dir="debug/rendered_frames", verbose=True):
    """
    Given lists of point clouds (points and colors) and pose estimations,
    renders the model (transformed by each pose) on top of the point cloud and saves
    the combined result into .ply files.

    Args:
        point_clouds_points (list of ndarray): Each element is an (N,3) array of points.
        point_clouds_colors (list of ndarray): Each element is an (N,3) array of colors.
        poses (list of ndarray): Each element is a 4x4 transformation matrix.
            (Assumed to be the transform that maps the model into the point cloud frame.)
        model (o3d.geometry.PointCloud): The model as an Open3D point cloud.
        output_dir (str): Directory to save the rendered .ply files.
        verbose (bool): If True, print debug messages.
    """
    os.makedirs(output_dir, exist_ok=True)

    for i, (pts, cols, pose) in enumerate(zip(point_clouds_points, point_clouds_colors, poses)):
        if verbose:
            print(f"Rendering frame {i}...")

        # Create the original point cloud from pts.
        orig_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        
        # Normalize colors: if max value > 1, assume colors are in 0-255 range.
        if np.max(cols) > 1.0:
            norm_cols = cols.astype(np.float32) / 255.0
        else:
            norm_cols = cols.astype(np.float32)
        norm_cols = np.clip(norm_cols, 0.0, 1.0)
        orig_pcd.colors = o3d.utility.Vector3dVector(norm_cols)
        
        # Make a copy of the model and apply the corresponding pose.
        model_copy = copy.deepcopy(model)
        model_copy.transform(pose)
        if len(model_copy.points) > 0:
            # Set the model's color to red.
            red_colors = np.tile([1.0, 0.0, 0.0], (len(model_copy.points), 1))
            model_copy.colors = o3d.utility.Vector3dVector(red_colors)
        
        # Combine the transformed model with the original point cloud.
        combined_pcd = model_copy + orig_pcd
        
        # Save the combined point cloud as a PLY file.
        out_path = os.path.join(output_dir, f"frame_{i:04d}.ply")
        o3d.io.write_point_cloud(out_path, combined_pcd)
        if verbose:
            print(f"Saved rendered frame to {out_path}")
            
def render_model_on_pointclouds_two_colors(
    point_clouds_points,
    point_clouds_colors,
    poses_red,
    poses_blue,
    model,
    output_dir="debug/rendered_frames",
    verbose=True
):
    """
    Given lists of point clouds (points and colors) and TWO sets of pose estimations,
    renders the model (transformed by poses_red in red, and by poses_blue in blue)
    on top of the point cloud and saves the combined result into .ply files.

    Args:
        point_clouds_points (list of ndarray): Each element is an (N,3) array of points.
        point_clouds_colors (list of ndarray): Each element is an (N,3) array of colors.
        poses_red (list of ndarray): Each element is a 4x4 transformation matrix
            for the red model. Must be the same length as point_clouds_points.
        poses_blue (list of ndarray): Each element is a 4x4 transformation matrix
            for the blue model. Must be the same length as point_clouds_points.
        model (o3d.geometry.PointCloud): The model as an Open3D point cloud.
        output_dir (str): Directory to save the rendered .ply files.
        verbose (bool): If True, print debug messages.
    """
    if len(poses_red) != len(point_clouds_points) or len(poses_blue) != len(point_clouds_points):
        raise ValueError("All input lists (point clouds, poses_red, poses_blue) must have the same length.")

    os.makedirs(output_dir, exist_ok=True)

    for i, (pts, cols, pose_r, pose_b) in enumerate(zip(point_clouds_points, point_clouds_colors, poses_red, poses_blue)):
        if verbose:
            print(f"Rendering frame {i}...")

        # Create the original point cloud from pts.
        orig_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))

        # Normalize colors: if max value > 1, assume colors are in 0-255 range.
        if np.max(cols) > 1.0:
            norm_cols = cols.astype(np.float32) / 255.0
        else:
            norm_cols = cols.astype(np.float32)
        norm_cols = np.clip(norm_cols, 0.0, 1.0)
        orig_pcd.colors = o3d.utility.Vector3dVector(norm_cols)

        # Copy the model for the red pose.
        model_copy_red = copy.deepcopy(model)
        model_copy_red.transform(pose_r)
        if len(model_copy_red.points) > 0:
            red_colors = np.tile([1.0, 0.0, 0.0], (len(model_copy_red.points), 1))
            model_copy_red.colors = o3d.utility.Vector3dVector(red_colors)

        # Copy the model for the blue pose.
        model_copy_blue = copy.deepcopy(model)
        model_copy_blue.transform(pose_b)
        if len(model_copy_blue.points) > 0:
            blue_colors = np.tile([0.0, 0.0, 1.0], (len(model_copy_blue.points), 1))
            model_copy_blue.colors = o3d.utility.Vector3dVector(blue_colors)

        # Combine everything: original cloud + red model + blue model
        combined_pcd = orig_pcd + model_copy_red + model_copy_blue

        # Save the combined point cloud as a PLY file.
        out_path = os.path.join(output_dir, f"frame_{i:04d}.ply")
        o3d.io.write_point_cloud(out_path, combined_pcd)
        if verbose:
            print(f"Saved rendered frame to {out_path}")
            
def render_positions_on_pointclouds_two_colors(
    point_clouds_points,
    point_clouds_colors,
    poses_red,
    poses_blue,
    output_dir="debug/rendered_frames",
    verbose=True
):
    """
    Given lists of point clouds (points and colors) and TWO sets of pose estimations,
    instead of rendering the model, we render two small clusters of points:
    one in red (transformed by poses_red) and one in blue (transformed by poses_blue).
    This helps debug the correctness of the pose coordinates.

    Args:
        point_clouds_points (list of ndarray): Each element is an (N,3) array of scene points.
        point_clouds_colors (list of ndarray): Each element is an (N,3) array of colors.
        poses_red (list of ndarray): Each element is a 4x4 transform for the "red" cluster.
        poses_blue (list of ndarray): Each element is a 4x4 transform for the "blue" cluster.
        output_dir (str): Directory to save the rendered .ply files.
        verbose (bool): If True, prints debug messages.
    """
    if len(poses_red) != len(point_clouds_points) or len(poses_blue) != len(point_clouds_points):
        raise ValueError("All input lists (point clouds, poses_red, poses_blue) must have the same length.")

    os.makedirs(output_dir, exist_ok=True)

    # Define a small local cluster around the origin to visualize each pose.
    # For instance, a little "cross" of 7 points with radius = 0.01.
    radius = 0.01
    cluster_local = np.array([
        [0.0, 0.0, 0.0],
        [ radius, 0.0,   0.0],
        [-radius, 0.0,   0.0],
        [0.0,  radius, 0.0],
        [0.0, -radius, 0.0],
        [0.0, 0.0,  radius],
        [0.0, 0.0, -radius],
    ], dtype=np.float32)

    for i, (pts, cols, pose_r, pose_b) in enumerate(zip(point_clouds_points, point_clouds_colors, poses_red, poses_blue)):
        if verbose:
            print(f"Rendering frame {i}...")

        # 1) Create the original scene point cloud from pts.
        orig_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))

        # Normalize colors: if max value > 1, assume 0-255 range.
        if np.max(cols) > 1.0:
            norm_cols = cols.astype(np.float32) / 255.0
        else:
            norm_cols = cols.astype(np.float32)
        norm_cols = np.clip(norm_cols, 0.0, 1.0)
        orig_pcd.colors = o3d.utility.Vector3dVector(norm_cols)

        # 2) Transform the local cluster by poses_red and poses_blue.
        #    This shows us exactly where each pose is in the scene.
        cluster_red_world = (pose_r @ np.hstack([cluster_local, np.ones((len(cluster_local), 1))]).T).T
        cluster_blue_world = (pose_b @ np.hstack([cluster_local, np.ones((len(cluster_local), 1))]).T).T

        # Convert to Nx3 (dropping homogeneous component).
        cluster_red_world_xyz = cluster_red_world[:, :3]
        cluster_blue_world_xyz = cluster_blue_world[:, :3]

        # 3) Create small point clouds for the red cluster and the blue cluster.
        cluster_red_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cluster_red_world_xyz))
        cluster_blue_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cluster_blue_world_xyz))

        if len(cluster_red_pcd.points) > 0:
            red_colors = np.tile([1.0, 0.0, 0.0], (len(cluster_red_pcd.points), 1))
            cluster_red_pcd.colors = o3d.utility.Vector3dVector(red_colors)
        if len(cluster_blue_pcd.points) > 0:
            blue_colors = np.tile([0.0, 0.0, 1.0], (len(cluster_blue_pcd.points), 1))
            cluster_blue_pcd.colors = o3d.utility.Vector3dVector(blue_colors)

        # 4) Combine everything: original scene + red cluster + blue cluster.
        combined_pcd = orig_pcd + cluster_red_pcd + cluster_blue_pcd

        # 5) Save to .ply
        out_path = os.path.join(output_dir, f"frame_{i:04d}.ply")
        o3d.io.write_point_cloud(out_path, combined_pcd)

        if verbose:
            print(f"Saved rendered frame {i} with red/blue clusters to {out_path}")