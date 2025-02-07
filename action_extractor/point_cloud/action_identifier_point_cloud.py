import numpy as np
import open3d as o3d
import random
import copy
from scipy.spatial.transform import Rotation as R
import os

def load_model_as_pointcloud(model_path, num_points=30000, model_in_mm=True):
    mesh = o3d.io.read_triangle_mesh(model_path)
    if mesh.is_empty():
        raise ValueError(f"Could not load mesh from: {model_path}")

    if model_in_mm:
        mesh.scale(0.001, center=(0,0,0))

    pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    return pcd

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
    # Dot product of unit quaternions => cos(half the angle)
    dot = np.abs(np.dot(q_curr, q_base))
    dot = np.clip(dot, -1.0, 1.0)
    angle = 2.0 * np.arccos(dot)
    return angle

def get_poses_from_pointclouds(
    point_clouds_points,
    point_clouds_colors,
    model_path,
    green_threshold=0.9,
    non_green_max=0.7,
    voxel_size=0.001,
    mesh_num_points=50000,
    debug_dir="debug/pointclouds_with_model",
    model_in_mm=True,
    dbscan_eps=0.02,
    dbscan_min_points=20,
    base_orientation_quat=np.array([0.7253942, 0.6844675, 0.05062998, 0.05238412]),
    max_orientation_angle=np.pi/2,
    verbose=True
):
    """
    Modified version that only prints when `verbose=True`.

    The function estimates poses for a model in a sequence of green-colored
    point clouds. For the first frame, it does RANSAC with an orientation filter
    plus ICP. For subsequent frames, it does RANSAC (without orientation filter)
    plus ICP, storing the best transform and using it optionally in the next frame.
    """

    # -------------------------------
    # 1) Load model once
    # -------------------------------
    os.makedirs(debug_dir, exist_ok=True)
    object_model_o3d = load_model_as_pointcloud(model_path,
                                                num_points=mesh_num_points,
                                                model_in_mm=model_in_mm)

    # -------------------------------------------------------------------------
    # Helper function: cluster, keep largest
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

    # -------------------------------------------------------------------------
    # Custom Up/Down ICP (modified to use verbose)
    # -------------------------------------------------------------------------
    def icp_threshold_updown_force_one(
        source,
        target,
        init_transform,
        max_total_iterations=2_000_000
    ):
        """
        Runs a custom "up-down" ICP procedure. 
        All print statements are controlled by the outer `verbose` flag.
        """
        times_visited = {}
        total_iterations_used = 0

        def get_iteration_count(threshold):
            t_key = float(threshold)
            vcount = times_visited.get(t_key, 0)
            it_count = 10000 * (voxel_size / threshold) * 8 * (1 + 0.1 * vcount)
            return max(1, int(it_count))

        def run_icp(threshold, init_pose):
            nonlocal total_iterations_used

            t_key = float(threshold)
            old_visits = times_visited.get(t_key, 0)
            times_visited[t_key] = old_visits + 1

            it_count = get_iteration_count(threshold)

            if total_iterations_used + it_count > max_total_iterations:
                it_count = max_total_iterations - total_iterations_used
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
                threshold,
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

        while threshold >= 1 * voxel_size:
            fitness, rmse, new_transform, used_iters = run_icp(threshold, current_transform)
            # if verbose:
            #     print(f"[UpDownICP] threshold={threshold:.4f}, used_iters={used_iters}, "
            #           f"fitness={fitness:.4f}, rmse={rmse:.5f}")

            if used_iters <= 0:
                if verbose:
                    print("No iteration budget left, break up/down loop.")
                break

            if fitness >= 0.98:
                threshold *= 0.5
                current_transform = new_transform
                if threshold < best_threshold:
                    best_transform = current_transform.copy()
                    best_threshold = threshold
                    best_fitness = fitness
                    best_rmse = rmse
            else:
                threshold *= 1.5
                current_transform = new_transform

        # final pass
        if threshold >= 1 * voxel_size:
            if verbose:
                print("Threshold never shrank below 1 * voxel_size.")
                print(f"[UpDownICP] Best threshold={best_threshold:.4f}, "
                      f"used_iters={max_total_iterations}, fitness={best_fitness:.4f}, rmse={best_rmse:.5f}")
            return best_transform
        else:
            if verbose:
                print(f"Threshold < 1 * voxel_size => final pass at threshold={threshold:.4f} ignoring fitness...")

        f_final, rmse_final, final_transform, used_iters_final = run_icp(threshold, current_transform)
        if verbose:
            print(f"[UpDownICP] Final pass threshold={threshold:.4f}, used_iters={used_iters_final}, "
                  f"fitness={f_final:.4f}, rmse={rmse_final:.5f}")
            print("Ignoring final pass fitness. Returning transform anyway.")

        return final_transform

    # -------------------------------------------------------------------------
    # FPFH
    # -------------------------------------------------------------------------
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
    # First-frame registration (with orientation filter)
    # -------------------------------------------------------------------------
    def register_green_points_to_model_first_frame(green_pts_np, model_pcd_o3d):
        if len(green_pts_np) < 10:
            if verbose:
                print("    Not enough green points (<10). Identity.")
            return np.eye(4)

        green_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(green_pts_np))
        largest_cluster = cluster_and_keep_largest(
            green_pcd, eps=dbscan_eps, min_points=dbscan_min_points
        )
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

        found_1_point_0 = False
        best_transform = None
        attempt = 0

        while True:
            seed = random.randint(0, 2**31 - 1)
            result_ransac = global_registration(
                green_o3d_clean, model_o3d_clean, src_fpfh, tgt_fpfh, seed=seed
            )
            attempt += 1

            if verbose:
                print(f"[Try #{attempt}] RANSAC seed={seed} -> fitness={result_ransac.fitness:.4f}, "
                      f"rmse={result_ransac.inlier_rmse:.5f}")

            R_est = result_ransac.transformation[:3, :3]
            angle_diff = orientation_angle_diff(R_est, base_orientation_quat)
            if angle_diff <= max_orientation_angle:
                if result_ransac.fitness >= 1.0 - 1e-12:
                    found_1_point_0 = True
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

        if found_1_point_0:
            final_transform = icp_threshold_updown_force_one(
                green_o3d_clean, model_o3d_clean, best_transform
            )
            return final_transform

        if verbose:
            print("    All attempts failed to find orientation near base. Fallback identity.")
        return np.eye(4)

    # -------------------------------------------------------------------------
    # Subsequent-frame registration (no orientation filter)
    # -------------------------------------------------------------------------
    def register_green_points_to_model_subsequent_frames(green_pts_np, model_pcd_o3d, prev_transform):
        """
        If you want to skip RANSAC and just do ICP from prev_transform,
        you can do that easily by calling `icp_threshold_updown_force_one` 
        directly with `prev_transform`.
        In the current implementation, we do exactly that.
        """
        if len(green_pts_np) < 10:
            return np.eye(4)

        green_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(green_pts_np))
        largest_cluster = cluster_and_keep_largest(
            green_pcd, eps=dbscan_eps, min_points=dbscan_min_points
        )
        if len(largest_cluster.points) < 10:
            return np.eye(4)

        green_o3d_clean = preprocess_pcd(largest_cluster, voxel_size)
        model_o3d_clean = preprocess_pcd(model_pcd_o3d, voxel_size)
        if len(green_o3d_clean.points) < 10 or len(model_o3d_clean.points) < 10:
            return np.eye(4)

        # For now, we just do ICP from prev_transform
        final_transform = icp_threshold_updown_force_one(
            green_o3d_clean, model_o3d_clean, init_transform=prev_transform
        )
        return final_transform

    # -------------------------------------------------------------------------
    #  Main Loop
    # -------------------------------------------------------------------------
    poses = []
    prev_transform_green_to_model = None

    for i, (pts, cols) in enumerate(zip(point_clouds_points, point_clouds_colors)):
        if verbose:
            print(f"\n=== Frame {i} ===")
        mask = (
            (cols[:,1] >= green_threshold) &
            (cols[:,0] <= non_green_max) &
            (cols[:,2] <= non_green_max)
        )
        green_pts = pts[mask]
        if verbose:
            print(f"  #points={len(pts)}, #green={len(green_pts)}")

        if len(green_pts) < 10:
            if verbose:
                print("  No green points, identity.")
            poses.append(np.eye(4))
            continue

        if i == 0:
            # First frame => orientation filter + ICP
            transform_green_to_model = register_green_points_to_model_first_frame(
                green_pts, object_model_o3d
            )
        else:
            # Subsequent frames => ICP from prev_transform
            transform_green_to_model = register_green_points_to_model_subsequent_frames(
                green_pts,
                object_model_o3d,
                prev_transform=prev_transform_green_to_model
            )

        T_model_in_cloud = np.linalg.inv(transform_green_to_model)
        poses.append(T_model_in_cloud)

        # Store for next iteration
        prev_transform_green_to_model = transform_green_to_model.copy()

        # Save debug .ply if desired
        if verbose:
            green_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(green_pts))
            green_pcd.colors = o3d.utility.Vector3dVector(
                np.tile([0,1,0], (len(green_pcd.points),1))
            )
            model_copy = copy.deepcopy(object_model_o3d)
            model_copy.transform(T_model_in_cloud)
            if len(model_copy.points) > 0:
                model_copy.colors = o3d.utility.Vector3dVector(
                    np.tile([1,0,0], (len(model_copy.points),1))
                )
            out_pcd = model_copy + green_pcd
            out_path = os.path.join(debug_dir, f"frame_{i:04d}.ply")
            o3d.io.write_point_cloud(out_path, out_pcd)
            print(f"  Saved debug PLY: {out_path}")

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