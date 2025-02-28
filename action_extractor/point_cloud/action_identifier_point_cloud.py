import numpy as np
import open3d as o3d
import random
import copy
from scipy.spatial.transform import Rotation as R
import os

def load_model_as_pointcloud(model_path, num_points=30000, model_in_mm=True):
    """
    Loads a mesh, optionally scaled from mm->m, then samples into a point cloud.
    Then swaps the x,y axes.
    """
    mesh = o3d.io.read_triangle_mesh(model_path)
    if mesh.is_empty():
        raise ValueError(f"Could not load mesh from: {model_path}")

    # Convert from mm to m if desired
    if model_in_mm:
        mesh.scale(0.001, center=(0,0,0))

    # =========================
    # Swap x,y axes: (x,y,z) -> (y,-x,z)
    # =========================
    swap_xy = np.array([
        [0.0, -1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    mesh.transform(swap_xy)

    # Now sample
    pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    return pcd

def load_model_as_mesh(model_path, model_in_mm=True):
    """
    Loads a mesh from the given path, optionally converting from mm to m,
    then swaps the x,y axes.
    """
    mesh = o3d.io.read_triangle_mesh(model_path)
    if mesh.is_empty():
        raise ValueError(f"Could not load mesh from: {model_path}")

    # Convert from mm to m if desired
    if model_in_mm:
        mesh.scale(0.001, center=(0, 0, 0))

    # =========================
    # Swap x,y axes: (x,y,z) -> (y,-x,z)
    # =========================
    swap_xy = np.array([
        [0.0, -1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    mesh.transform(swap_xy)

    return mesh


def cluster_and_keep_largest(pcd_o3d, eps=0.02, min_points=20):
    if len(pcd_o3d.points) == 0:
        return pcd_o3d

    labels = np.array(pcd_o3d.cluster_dbscan(eps=eps, min_points=min_points))
    if len(labels) == 0 or np.all(labels < 0):
        return o3d.geometry.PointCloud()

    unique_labels, counts = np.unique(labels, return_counts=True)
    valid_mask = (unique_labels >= 0)
    if not np.any(valid_mask):
        return o3d.geometry.PointCloud()
    valid_labels = unique_labels[valid_mask]
    valid_counts = counts[valid_mask]
    largest_label = valid_labels[np.argmax(valid_counts)]
    indices_largest = np.where(labels == largest_label)[0]
    return pcd_o3d.select_by_index(indices_largest)

# Helper to convert a 3x3 rotation matrix to a quaternion (x,y,z,w)
def matrix_to_quat(R_mat):
    R_mat_copy = np.array(R_mat, copy=True, dtype=np.float64)
    rot = R.from_matrix(R_mat_copy)
    q = rot.as_quat()  # [x, y, z, w]
    return q

# Compute the orientation difference (in radians) between a rotation matrix R_mat
# and a base quaternion q_base (x,y,z,w).
def orientation_angle_diff(R_mat, q_base):
    q_curr = matrix_to_quat(R_mat)
    # print(f'R_est: {q_curr}') # For providing correct base_orientation
    # Dot product of unit quaternions => cos(half the angle)
    dot = np.abs(np.dot(q_curr, q_base))
    dot = np.clip(dot, -1.0, 1.0)
    angle = 2.0 * np.arccos(dot)
    return angle

def merge_meshes(mesh1, mesh2):
    """
    Merge two triangle meshes (mesh1 + mesh2) into a single mesh.
    This is useful when mesh2 is a small "marker" shape placed at a specific coordinate.
    
    The resulting mesh will contain all of mesh1's vertices / faces plus all of mesh2's.
    Vertex colors will also be combined if present; if mesh1 or mesh2 doesn't have colors,
    we assign defaults.
    """
    merged = o3d.geometry.TriangleMesh()

    # 1. Convert geometry to numpy arrays
    verts1 = np.asarray(mesh1.vertices)
    tris1  = np.asarray(mesh1.triangles)

    verts2 = np.asarray(mesh2.vertices)
    tris2  = np.asarray(mesh2.triangles)

    # 2. Offset mesh2's triangle indices so they come after mesh1
    tris2_offset = tris2 + len(verts1)

    # 3. Combine vertices and triangles
    merged_verts = np.vstack([verts1, verts2])
    merged_tris  = np.vstack([tris1, tris2_offset])

    merged.vertices  = o3d.utility.Vector3dVector(merged_verts)
    merged.triangles = o3d.utility.Vector3iVector(merged_tris)

    # 4. Combine colors. If the original meshes have no vertex colors,
    #    we assign default greys for mesh1 and keep the assigned color for mesh2.
    if mesh1.has_vertex_colors():
        colors1 = np.asarray(mesh1.vertex_colors)
    else:
        # default e.g. gray for each vertex
        colors1 = np.tile([0.6, 0.6, 0.6], (len(verts1), 1))
    
    if mesh2.has_vertex_colors():
        colors2 = np.asarray(mesh2.vertex_colors)
    else:
        # default e.g. bright red for each vertex
        colors2 = np.tile([1.0, 0.0, 0.0], (len(verts2), 1))

    merged_colors = np.vstack([colors1, colors2])
    merged.vertex_colors = o3d.utility.Vector3dVector(merged_colors)

    # Optional: If you care about vertex normals, do the same for normals
    # or let Open3D recompute them later.

    return merged

def create_marker_sphere(radius=0.005, color=(1.0, 0.0, 0.0)):
    """
    Creates a small sphere TriangleMesh of the given radius,
    paints it uniformly 'color' (r,g,b) in [0,1], 
    and returns it as a TriangleMesh.
    """
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=12)
    sphere.paint_uniform_color(color)
    return sphere

def visualize_mesh_with_bottom_marker(mesh_o3d, bottom_center_local, marker_radius=0.005, save_path="local_mesh_with_marker.ply"):
    """
    Merges the given mesh with a small 'marker' sphere placed at bottom_center_local (x,y,z)
    in the mesh's local coordinate frame. Writes to save_path as a single .ply TriangleMesh.

    Args:
        mesh_o3d (o3d.geometry.TriangleMesh): your main model
        bottom_center_local (array-like): [x, y, z, 1] homogeneous or [x, y, z], local coords
        marker_radius (float): how big of a sphere to place
        save_path (str): where to write the merged .ply
    """
    # 1) Create the small sphere marker in local coordinates (at origin).
    marker = create_marker_sphere(radius=marker_radius, color=(1.0, 0.0, 0.0))

    # 2) Build a 4x4 transform that puts the marker at bottom_center_local
    if len(bottom_center_local) == 4:
        bottom_center_local = bottom_center_local[:3]  # discard the homogeneous '1'
    marker_transform = np.eye(4)
    marker_transform[0:3, 3] = bottom_center_local  # place sphere center at [x, y, z]

    # 3) Transform the sphere
    marker.transform(marker_transform)

    # 4) Merge the original mesh + marker into a single TriangleMesh
    combined_mesh = merge_meshes(mesh_o3d, marker)

    # 5) Save as .ply
    o3d.io.write_triangle_mesh(save_path, combined_mesh)
    print(f"Saved local mesh + marker sphere as: {save_path}")

def get_poses_from_pointclouds(
    point_clouds_points,
    point_clouds_colors,
    model_path,
    green_threshold=0.9,
    non_green_max=0.7,
    voxel_size=0.002,
    mesh_num_points=20000,
    debug_dir="debug/pointclouds_with_model",
    model_in_mm=True,
    dbscan_eps=0.02,
    dbscan_min_points=20,
    base_orientation_quat=np.array([ 0.61854268,  0.78458513,  0.03966061, -0.01606732]),
    max_orientation_angle=np.pi / 8,
    verbose=True,
    icp_method="updown",
    offset=[-0.002, 0, 0.078]
):
    """
    Estimates poses for a Franka Panda gripper in a sequence of green-colored point clouds.
    For the first frame, it does RANSAC (with orientation filter) + ICP.
    For subsequent frames, it does RANSAC (no orientation filter) + ICP, or just ICP.

    The returned pose for each frame has its origin placed at the "bottom center" of the
    mesh (plus an adjustable offset). That way, when orientation changes, the distance
    between the true jaw center and the returned position remains accurate.

    Args:
        point_clouds_points, point_clouds_colors: Lists of (N,3) arrays of points and colors
                                                  for each frame.
        model_path (str): Path to the gripper mesh file.
        green_threshold, non_green_max, voxel_size, ...
            (Various thresholds and parameters for point cloud processing.)
        mesh_num_points (int): Number of points to sample if you are also using a model
                               point cloud. (Used by load_model_as_pointcloud, though we
                               also load the full mesh for accuracy.)
        bottom_offset (float): How far below the mesh's lowest plane (in local Z) we want
                               the final reported pose origin, in meters.
        ...
    Returns:
        poses (list of ndarray): Each element is a 4x4 transform with the orientation from ICP
                                 but the translation set to the "bottom center" minus bottom_offset.
    """

    if verbose:
        os.makedirs(debug_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1) Load the model as a mesh (for accurate geometry)
    # -------------------------------------------------------------------------
    # (We assume you have a helper function that loads the mesh in consistent units, e.g. meters.)
    model_mesh_o3d = load_model_as_mesh(model_path, model_in_mm=model_in_mm)
    # Also load a point cloud representation for ICP (like before).
    object_model_o3d = load_model_as_pointcloud(model_path,
                                                num_points=mesh_num_points,
                                                model_in_mm=model_in_mm)

    # -------------------------------------------------------------------------
    # 1a) Find the "bottom center" of the mesh in local coordinates
    # -------------------------------------------------------------------------
    def get_center_of_bounding_box_in_local(mesh_o3d, offset):
        """
        Computes the axis-aligned bounding box (AABB) of the mesh, 
        and returns the coordinate of its center in local space. 
        Optionally shifts the Z coordinate by 'offset' meters (e.g., move center "down" or "up").

        Returns:
            np.array of shape (4,): [cx, cy, cz, 1.0]
        """

        # 1) If mesh is empty, fallback

        # 2) Obtain the axis-aligned bounding box
        aabb = mesh_o3d.get_axis_aligned_bounding_box()
        # 3) Get the center of that bounding box
        center = aabb.get_center()  # 3D coords (x, y, z)

        # 4) Optionally apply an offset to z
        center += offset

        # 5) Return as homogeneous [cx, cy, cz, 1]
        return np.array([center[0], center[1], center[2], 1.0], dtype=np.float32)


    # Precompute the local bottom center (4x1 homogeneous)
    bottom_center_local = get_center_of_bounding_box_in_local(
        mesh_o3d=model_mesh_o3d,
        offset=offset
    )
    
    if verbose:
        from copy import deepcopy
        # 1) Assign a uniform color to the mesh
        #    We can do this if your mesh lacks built-in per-vertex colors
        model_mesh_copy = deepcopy(model_mesh_o3d)  # don't overwrite original
        model_mesh_copy.paint_uniform_color([0.2, 0.7, 0.2])  # pale green

        # 2) Create a small "marker" sphere in local coords at origin,
        #    then transform it to the bottom_center_local
        marker_radius = 0.005
        marker = create_marker_sphere(radius=marker_radius, color=(1.0, 0.0, 0.0))

        # We assume bottom_center_local is a 4D [x, y, z, 1]. Extract the first 3 dims:
        base_pt = bottom_center_local[:3]

        # Build a transform that moves the marker to [x, y, z]
        marker_transform = np.eye(4)
        marker_transform[0:3, 3] = base_pt

        # Apply the transform to the marker
        marker.transform(marker_transform)

        # 3) Merge the painted mesh + the marker sphere into one mesh
        combined_local_debug = merge_meshes(model_mesh_copy, marker)

        # 4) Save exactly one .ply in local coords
        local_debug_path = os.path.join(debug_dir, "local_model_with_bottom_cluster.ply")
        o3d.io.write_triangle_mesh(local_debug_path, combined_local_debug)
        print(f"Saved local debug mesh with bottom sphere at:\n  {local_debug_path}")

    # -------------------------------------------------------------------------
    # 2) Helper function for clustering / voxel downsampling
    # -------------------------------------------------------------------------
    def preprocess_pcd(pcd, voxel):
        if voxel > 0:
            pcd_down = pcd.voxel_down_sample(voxel)
        else:
            pcd_down = pcd
        pcd_down.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=2.0 * voxel if voxel > 0 else 0.01,
                max_nn=30
            )
        )
        return pcd_down

    # ... ICP functions remain unchanged ...
    # (icp_threshold_updown_force_one, icp_multiscale_standard, compute_fpfh, global_registration, etc.)

    def icp_threshold_updown_force_one(
        source,
        target,
        init_transform,
        total_iterations=2_000_000,
        verbose=True
    ):
        """
        Custom "up-down" ICP procedure:
          - Dynamically adjusts the threshold up/down based on fitness.
          - Has a total iteration budget.
        """
        times_visited = {}
        total_iterations_used = 0

        def get_iteration_count(threshold):
            t_key = float(threshold)
            vcount = times_visited.get(t_key, 0)
            # Example formula
            it_count = 10000 * (voxel_size / threshold) * 8 * (1 + 0.1 * vcount)
            return min(109875, max(1, int(it_count))) # <-- capping at 100k, so that icp call doesn't get stuck

        def run_icp(threshold_val, init_pose):
            nonlocal total_iterations_used

            t_key = float(threshold_val)
            old_visits = times_visited.get(t_key, 0)
            times_visited[t_key] = old_visits + 1

            it_count = get_iteration_count(threshold_val)

            if total_iterations_used + it_count > total_iterations:
                it_count = total_iterations - total_iterations_used

            if it_count <= 0:
                return 0.0, 999999.0, init_pose, 0

            crit = o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=it_count,
                relative_fitness=1e-6,
                relative_rmse=1e-6
            )
            result_icp = o3d.pipelines.registration.registration_icp(
                source,
                target,
                threshold_val,
                init_pose,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                criteria=crit
            )

            total_iterations_used += it_count
            return (result_icp.fitness,
                    result_icp.inlier_rmse,
                    result_icp.transformation,
                    it_count)

        current_transform = init_transform.copy()
        best_transform = current_transform.copy()
        best_fitness = 0.0
        threshold = 8.0 * voxel_size
        best_threshold = threshold
        best_rmse = 1.0

        # Loop until we exhaust total_iterations
        while True:
            fitness, rmse, new_transform, used_iters = run_icp(threshold, current_transform)

            if verbose:
                print(f"[UpDownICP] thr={threshold:.4f}, used_iters={used_iters}, "
                      f"fitness={fitness:.4f}, rmse={rmse:.5f}, total_used={total_iterations_used}/{total_iterations}")

            if used_iters <= 0:
                if verbose:
                    print("No iteration budget left or 0 used => break.")
                break

            if fitness >= 0.99:
                threshold *= 0.5
                current_transform = new_transform
                if threshold < best_threshold:
                    best_transform = current_transform.copy()
                    best_threshold = threshold
                    best_fitness = fitness
                    best_rmse = rmse
            else:
                threshold *= 1.5
                # current_transform = new_transform

            if total_iterations_used >= total_iterations:
                break

        if verbose:
            print(f"[UpDownICP] total_iterations_used={total_iterations_used}, "
                  f"best_threshold={best_threshold:.4f}, fitness={best_fitness:.4f}, rmse={best_rmse:.5f}")

        return best_transform
    
    def icp_multiscale_standard(
        source,
        target,
        init_transform,
        verbose=True,
    ):
        """
        A classic multi-scale ICP pipeline:
          1. We define multiple voxel sizes (coarse to fine).
          2. Downsample source & target at each scale.
          3. Run ICP with some max iteration at each scale.
          4. Pass the result as initialization to the next scale.
        """

        # Example multi-scale parameters (tune as needed).
        voxel_sizes = [5.0 * voxel_size, 2.5 * voxel_size, 1.0 * voxel_size]
        max_iters = [50, 30, 14]  # number of iterations per scale

        current_transform = init_transform.copy()

        for scale_idx, vsize in enumerate(voxel_sizes):
            # Downsample
            source_down = source.voxel_down_sample(vsize)
            target_down = target.voxel_down_sample(vsize)

            # Re-estimate normals for each downsample
            source_down.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=vsize * 2.0, max_nn=30
                )
            )
            target_down.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=vsize * 2.0, max_nn=30
                )
            )

            # ICP threshold can be e.g. ~1-2x the voxel size
            icp_dist_thresh = 1.5 * vsize

            if verbose:
                print(f"[Multi-Scale ICP] scale={scale_idx}, voxel={vsize}, max_iter={max_iters[scale_idx]}")

            # Setup criteria
            crit = o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=max_iters[scale_idx],
                relative_fitness=1e-6,
                relative_rmse=1e-6
            )

            # Run point-to-plane ICP
            result_icp = o3d.pipelines.registration.registration_icp(
                source_down,
                target_down,
                icp_dist_thresh,
                current_transform,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                criteria=crit
            )

            current_transform = result_icp.transformation

            if verbose:
                print(f"   => fitness={result_icp.fitness:.4f}, rmse={result_icp.inlier_rmse:.5f}")

        return current_transform
    
    def compute_fpfh(pcd, voxel):
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=2.0 * voxel if voxel > 0 else 0.01,
                max_nn=30
            )
        )
        radius_feature = 5.0 * voxel if voxel > 0 else 0.05
        return o3d.pipelines.registration.compute_fpfh_feature(
            pcd,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
        )

    def global_registration(source, target, source_fpfh, target_fpfh, seed):
        o3d.utility.random.seed(seed)
        dist_thresh = max(voxel_size * 1.5, 0.04)
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source,
            target,
            source_fpfh,
            target_fpfh,
            mutual_filter=False,
            max_correspondence_distance=dist_thresh,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            ransac_n=4,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist_thresh)
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
                max_iteration=100000, confidence=0.99
            )
        )
        return result
    
    # -------------------------------------------------------------------------
    # 3) Registration for first frame
    # -------------------------------------------------------------------------
    def register_green_points_to_model_first_frame(green_pts_np, model_pcd_o3d):
        # (Unchanged, except for referencing your existing code.)
        if len(green_pts_np) < 10:
            if verbose:
                print("    Not enough green points (<10). Identity.")
            return np.eye(4)

        green_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(green_pts_np))
        largest_cluster = cluster_and_keep_largest(green_pcd, eps=dbscan_eps, min_points=dbscan_min_points)
        if len(largest_cluster.points) < 10:
            if verbose:
                print("    Largest cluster <10 points, identity.")
            return np.eye(4)

        green_o3d_clean = preprocess_pcd(largest_cluster, voxel_size)
        model_o3d_clean = preprocess_pcd(model_pcd_o3d, voxel_size)
        if len(green_o3d_clean.points) < 10 or len(model_o3d_clean.points) < 10:
            if verbose:
                print("    After cleaning, too few points. Identity.")
            return np.eye(4)

        src_fpfh = compute_fpfh(green_o3d_clean, voxel_size)
        tgt_fpfh = compute_fpfh(model_o3d_clean, voxel_size)

        found_good_ransac = False
        best_transform = None
        attempt = 0

        while True:
            seed = random.randint(0, 2**31 - 1)
            result_ransac = global_registration(green_o3d_clean, model_o3d_clean, src_fpfh, tgt_fpfh, seed=seed)
            attempt += 1

            if verbose:
                print(f"[Try #{attempt}] RANSAC seed={seed} -> fitness={result_ransac.fitness:.4f}, "
                      f"rmse={result_ransac.inlier_rmse:.5f}")

            R_est = result_ransac.transformation[:3, :3]
            angle_diff = orientation_angle_diff(R_est, base_orientation_quat)

            if angle_diff <= max_orientation_angle:
                # Found a transform with orientation near base.
                if result_ransac.fitness >= 1.0 - 1e-12:
                    found_good_ransac = True
                    best_transform = result_ransac.transformation
                    if verbose:
                        print("Found a valid transform with ~1.0 fitness. Done.")
                    break
                else:
                    if verbose:
                        print("Orientation valid but fitness < 1.0. Trying another seed.")
            else:
                if verbose:
                    print("Orientation not valid, trying another seed.")

            # Possibly cap attempts if needed
            # if attempt > 200: break

        if found_good_ransac:
            # Refine with ICP
            if icp_method == "updown":
                final_transform = icp_threshold_updown_force_one(
                    green_o3d_clean, model_o3d_clean, best_transform, verbose=verbose
                )
            else:  # "multiscale"
                final_transform = icp_multiscale_standard(
                    green_o3d_clean, model_o3d_clean, best_transform, verbose=verbose
                )
            return final_transform

        if verbose:
            print("    All attempts failed to find orientation near base. Fallback identity.")
        return np.eye(4)

    # -------------------------------------------------------------------------
    # 4) Registration for subsequent frames
    # -------------------------------------------------------------------------
    def register_green_points_to_model_subsequent_frames(green_pts_np, model_pcd_o3d, prev_transform):
        if len(green_pts_np) < 10:
            return np.eye(4)

        green_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(green_pts_np))
        largest_cluster = cluster_and_keep_largest(green_pcd, eps=dbscan_eps, min_points=dbscan_min_points)
        if len(largest_cluster.points) < 10:
            return np.eye(4)

        green_o3d_clean = preprocess_pcd(largest_cluster, voxel_size)
        model_o3d_clean = preprocess_pcd(model_pcd_o3d, voxel_size)
        if len(green_o3d_clean.points) < 10 or len(model_o3d_clean.points) < 10:
            return np.eye(4)

        if icp_method == "updown":
            final_transform = icp_threshold_updown_force_one(
                green_o3d_clean, model_o3d_clean, init_transform=prev_transform, verbose=verbose
            )
        else:  # "multiscale"
            final_transform = icp_multiscale_standard(
                green_o3d_clean, model_pcd_o3d, init_transform=prev_transform, verbose=verbose
            )
        return final_transform

    # -------------------------------------------------------------------------
    # 5) Main Loop
    # -------------------------------------------------------------------------
    poses = []
    prev_transform_green_to_model = None

    for i, (pts, cols) in enumerate(zip(point_clouds_points, point_clouds_colors)):
        if verbose:
            print(f"\n=== Frame {i} ===")

        # Identify green points
        mask = (
            (cols[:,1] >= green_threshold) &
            (cols[:,0] <= non_green_max) &
            (cols[:,2] <= non_green_max)
        )
        green_pts = pts[mask]
        if verbose:
            print(f"  #points={len(pts)}, #green={len(green_pts)}")

        # Debug: Save first frame's green points
        if i == 0 and verbose:
            debug_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(green_pts))
            debug_pcd.colors = o3d.utility.Vector3dVector(np.tile([0,1,0], (len(green_pts),1)))
            debug_out_path = os.path.join(debug_dir, "first_frame_green_points.ply")
            o3d.io.write_point_cloud(debug_out_path, debug_pcd)
            print(f"  Saved first frame green points to {debug_out_path}")

        # If no green points, fallback
        if len(green_pts) < 10:
            if verbose:
                print("  No green points, identity.")
            poses.append(np.eye(4))
            continue

        # Register
        if i == 0:
            transform_green_to_model = register_green_points_to_model_first_frame(
                green_pts, object_model_o3d
            )
        else:
            transform_green_to_model = register_green_points_to_model_subsequent_frames(
                green_pts, object_model_o3d, prev_transform=prev_transform_green_to_model
            )

        # Invert to get T_model_in_cloud
        T_model_in_cloud = np.linalg.inv(transform_green_to_model)

        # ---------------------------------------------------------------------
        # 6) Adjust the final transform to place the "lowered bottom center"
        #    at the transform's origin.
        # ---------------------------------------------------------------------
        # local point in model coords
        local_pt_4 = bottom_center_local  # [x, y, z, 1]
        world_pt_4 = T_model_in_cloud @ local_pt_4  # shape (4,)

        # Now set T_model_in_cloud's translation to that point:
        T_model_in_cloud_no_offset = T_model_in_cloud.copy()
        T_model_in_cloud[0:3, 3] = world_pt_4[0:3]

        # That's our final pose for this frame
        poses.append(T_model_in_cloud)
        prev_transform_green_to_model = transform_green_to_model.copy()

        # Debug: save combined PLY
        if verbose:
            orig_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
            # Normalize colors
            if np.max(cols) > 1.0:
                norm_cols = cols.astype(np.float32) / 255.0
            else:
                norm_cols = cols.astype(np.float32)
            norm_cols = np.clip(norm_cols, 0.0, 1.0)
            orig_pcd.colors = o3d.utility.Vector3dVector(norm_cols)

            # Make a copy of the model pcd and transform
            model_copy = copy.deepcopy(object_model_o3d)
            model_copy.transform(T_model_in_cloud_no_offset)
            if len(model_copy.points) > 0:
                red_colors = np.tile([1.0, 0.0, 0.0], (len(model_copy.points), 1))
                model_copy.colors = o3d.utility.Vector3dVector(red_colors)

            out_pcd = model_copy + orig_pcd
            out_path = os.path.join(debug_dir, f"frame_{i:04d}.ply")
            o3d.io.write_point_cloud(out_path, out_pcd)
            print(f"  Saved debug PLY: {out_path}")

    # Return the final poses list
    return poses



def get_fingers_distances_from_pointclouds(
    point_clouds_points,
    point_clouds_colors,
    cyan_threshold=0.9,
    magenta_threshold=0.9,
    non_cyan_max=0.3,
    non_magenta_max=0.3,
    dbscan_eps=0.02,
    dbscan_min_points=20,
    debug_dir="debug/fingers",
    verbose=True
):
    """
    Computes a simple estimate of the distance between two gripper fingers:
      - One finger is assumed to be colored 'cyan' (high G and B, low R).
      - The other finger is assumed to be colored 'magenta' (high R and B, low G).
    For each frame:
      1) We extract points belonging to each color via a mask.
      2) Cluster them, keep the largest cluster.
      3) Compute cluster centroids.
      4) Compute the distance between the two centroids.

    Args:
        point_clouds_points (list of np.ndarray): Each element is (N_i x 3) array of XYZ points.
        point_clouds_colors (list of np.ndarray): Each element is (N_i x 3) array of RGB in [0,1].
        cyan_threshold (float): Minimum G and B for a point to be considered "very cyan".
        magenta_threshold (float): Minimum R and B for a point to be considered "very magenta".
        non_cyan_max (float): Maximum R for a point to be considered "very cyan".
        non_magenta_max (float): Maximum G for a point to be considered "very magenta".
        dbscan_eps (float): DBSCAN eps parameter for clustering.
        dbscan_min_points (int): Minimum cluster size for DBSCAN.
        debug_dir (str): Directory path to save debug PLY files.
        verbose (bool): If True, prints out status messages and saves debug files.

    Returns:
        distances (list of float): Distance between the two centroids for each frame.
                                  If either cluster is empty in a frame, the distance is np.nan.
    """
    os.makedirs(debug_dir, exist_ok=True)

    # We'll store the distances here (one distance per frame)
    distances = []

    # Loop over each frame in the sequence
    for i, (pts, cols) in enumerate(zip(point_clouds_points, point_clouds_colors)):
        if verbose:
            print(f"\n=== Frame {i} ===")

        # 1) Identify mask for "cyan" points:
        #    - R <= non_cyan_max
        #    - G >= cyan_threshold
        #    - B >= cyan_threshold
        mask_cyan = (
            (cols[:, 0] <= non_cyan_max) & 
            (cols[:, 1] >= cyan_threshold) &
            (cols[:, 2] >= cyan_threshold)
        )

        # 2) Identify mask for "magenta" points:
        #    - R >= magenta_threshold
        #    - G <= non_magenta_max
        #    - B >= magenta_threshold
        mask_magenta = (
            (cols[:, 0] >= magenta_threshold) &
            (cols[:, 1] <= non_magenta_max) &
            (cols[:, 2] >= magenta_threshold)
        )

        cyan_pts = pts[mask_cyan]
        magenta_pts = pts[mask_magenta]

        if verbose:
            print(f"  #points total={len(pts)}")
            print(f"    #cyan   ={len(cyan_pts)}")
            print(f"    #magenta={len(magenta_pts)}")

        # 3) Make Open3D point clouds for each color
        cyan_pcd_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cyan_pts))
        magenta_pcd_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(magenta_pts))

        # 4) Cluster and keep the largest cluster for each color
        largest_cyan_pcd    = cluster_and_keep_largest(cyan_pcd_o3d, eps=dbscan_eps, min_points=dbscan_min_points)
        largest_magenta_pcd = cluster_and_keep_largest(magenta_pcd_o3d, eps=dbscan_eps, min_points=dbscan_min_points)

        # 5) Compute centroids. If empty, we'll store np.nan
        if len(largest_cyan_pcd.points) < 1 or len(largest_magenta_pcd.points) < 1:
            if verbose:
                print("  One or both finger clusters are empty. Distance = NaN")
            distance = np.nan
        else:
            cyan_centroid = np.mean(np.asarray(largest_cyan_pcd.points), axis=0)
            magenta_centroid = np.mean(np.asarray(largest_magenta_pcd.points), axis=0)
            distance = np.linalg.norm(cyan_centroid - magenta_centroid)
            if verbose:
                print(f"  Distance = {distance:.4f}")

        distances.append(distance)

        # 6) Optionally save debug PLY if verbose
        if verbose:
            # Paint the largest cluster for visualization
            if len(largest_cyan_pcd.points) > 0:
                largest_cyan_pcd.paint_uniform_color([0.0, 1.0, 1.0])  # cyan
            if len(largest_magenta_pcd.points) > 0:
                largest_magenta_pcd.paint_uniform_color([1.0, 0.0, 1.0])  # magenta

            # Combine them for a single debug cloud
            out_pcd = largest_cyan_pcd + largest_magenta_pcd
            out_path = os.path.join(debug_dir, f"fingers_frame_{i:04d}.ply")
            o3d.io.write_point_cloud(out_path, out_pcd)
            print(f"  Saved debug PLY: {out_path}")

    return distances