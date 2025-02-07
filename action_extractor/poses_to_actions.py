import numpy as np
from action_extractor.utils.angles import *

def smooth_positions(
    poses: list[np.ndarray],
    window_size: int = 2,
    dist_threshold: float = 0.15
) -> np.ndarray:
    """
    Smooths a sequence of SE(3) 4x4 poses by removing sudden 'outlier' 3D positions.
    Returns an (N,3) NumPy array of cleaned translations.

    Steps:
      1) Extract positions from each 4x4 pose => p_i in R^3.
      2) Outlier detection pass:
         - For each frame i, gather its neighbors in [i - window_size .. i + window_size],
           skipping i itself and any indices out of range.
         - Compute the median of those neighbor positions (robust local estimate).
         - If dist(p_i, median) > dist_threshold => mark p_i as outlier.
      3) Outlier replacement pass:
         - For each outlier, find the nearest inlier to the left, and the nearest inlier to the right.
         - If both exist => linearly interpolate.
         - If only one side exists => clamp to that inlier's position.
      4) Return the final array of size (N, 3).

    :param poses: List of N SE(3) 4x4 matrices. poses[i][:3,:3] is rotation, poses[i][:3,3] is translation.
    :param window_size: How many frames on each side to consider as neighbors for local median.
    :param dist_threshold: Distance above which a point is flagged as an outlier from its neighbors.
    :return: Nx3 float32 array of cleaned/smoothed positions.
    """
    N = len(poses)
    if N == 0:
        return np.zeros((0,3), dtype=np.float32)

    # 1) Extract the Nx3 positions
    positions = np.array([pose[:3, 3] for pose in poses], dtype=np.float32)

    # 2) First pass: detect outliers
    outlier_mask = np.zeros(N, dtype=bool)
    for i in range(N):
        # Gather neighbors in range [i-window_size.. i+window_size], excluding i
        left = max(0, i - window_size)
        right = min(N, i + window_size + 1)
        neighbor_indices = [idx for idx in range(left, right) if idx != i]

        if len(neighbor_indices) == 0:
            # No neighbors => can't decide => skip outlier check
            continue

        neighbor_pts = positions[neighbor_indices]  # shape (#neighbors, 3)

        # Median in x, y, z among neighbors
        local_median = np.median(neighbor_pts, axis=0)
        dist_i = np.linalg.norm(positions[i] - local_median)

        if dist_i > dist_threshold:
            outlier_mask[i] = True

    # 3) Second pass: replace outliers by interpolation
    cleaned_positions = positions.copy()
    for i in range(N):
        if not outlier_mask[i]:
            continue

        # Find nearest inlier to the left
        left_idx = i - 1
        while left_idx >= 0 and outlier_mask[left_idx]:
            left_idx -= 1

        # Find nearest inlier to the right
        right_idx = i + 1
        while right_idx < N and outlier_mask[right_idx]:
            right_idx += 1

        if left_idx < 0 and right_idx >= N:
            # Everything is outlier => fallback: keep original or zero
            # We'll just keep original, but you could do something else
            pass
        elif left_idx < 0:
            # No inlier to the left => clamp to right
            cleaned_positions[i] = cleaned_positions[right_idx]
        elif right_idx >= N:
            # No inlier to the right => clamp to left
            cleaned_positions[i] = cleaned_positions[left_idx]
        else:
            # Linear interpolate
            frac = (i - left_idx) / float(right_idx - left_idx)
            p_left = cleaned_positions[left_idx]
            p_right = cleaned_positions[right_idx]
            cleaned_positions[i] = p_left + frac * (p_right - p_left)

    return cleaned_positions

def poses_to_absolute_actions(
    poses, 
    gripper_actions,
    env,
    smooth=True
):
    """
    We smooth out the translation portion by looking ahead for a future
    index j where the jump is acceptable, and linearly interpolate intermediate
    positions. If no such j is found (end of trajectory), we do a linear 
    extrapolation from the past velocity or hold the last stable position.

    The orientation logic remains the same as before:
    - orientation is computed incrementally from poses[i]->poses[i+1].
    - we do not alter orientation if there's a large jump in position.

    Steps in detail:
    1) SMOOTH POSITIONS:
        - We'll create a 'smoothed_positions' array of shape (len(poses), 3).
        - Starting from i=0, if jump i->i+1 is small, keep it.
        If jump is big, search forward for j with a small jump from i.
        If found, linearly interpolate from i..j.
        If not found (end), extrapolate or hold constant.
    2) ORIENTATION:
        - For each i in [0 .. num_samples-2],
        compute delta quaternion from poses[i]-> poses[i+1].
        Accumulate into `current_orientation`.
        Convert to axis-angle (with sign unification).
    3) BUILD ACTIONS:
        - For each i in [0 .. num_samples-2], the final action 
        uses `smoothed_positions[i+1]` as [px,py,pz] 
        and the just-computed orientation as [rx,ry,rz].
        - Set gripper=1.0 or your logic.

    Returns:
    all_actions of shape (num_samples-1, 7).
    """

    num_samples = len(poses)
    if num_samples < 2:
        return np.zeros((0, 7), dtype=np.float32)

    # 1) Compute a smoothed set of positions
    if smooth:
        smoothed_positions = smooth_positions(poses, dist_threshold=0.15)
    else:
        smoothed_positions = np.array([pose[:3, 3] for pose in poses], dtype=np.float32)

    # 2) Start from environment's known initial eef quaternion 
    current_orientation = env.env.env._eef_xquat.astype(np.float32)
    current_orientation = quat_normalize(current_orientation)
    current_position = env.env.env._eef_xpos.astype(np.float32)

    # We'll have num_actions = num_samples - 1
    num_actions = num_samples - 1
    all_actions = np.zeros((num_actions + 10, 7), dtype=np.float32)

    prev_rvec = None
    z_offset = None

    for i in range(num_actions):
        # --- Orientation from poses[i] -> poses[i+1] (the "real" rotation) ---
        # --- Position from the precomputed 'smoothed_positions' ---
        if z_offset is None:
            z_offset = smoothed_positions[i][2] - current_position[2]
            
        px, py, pz = smoothed_positions[i+1]
        
        R_i  = poses[i][:3, :3]
        R_i1 = poses[i+1][:3, :3]

        q_i  = rotation_matrix_to_quaternion(R_i)
        q_i1 = rotation_matrix_to_quaternion(R_i1)

        q_i   = quat_normalize(q_i)
        q_i1  = quat_normalize(q_i1)
        q_inv = quat_inv(q_i)

        q_delta = quat_multiply(q_inv, q_i1)
        q_delta = quat_normalize(q_delta)

        # Accumulate orientation
        current_orientation = quat_multiply(current_orientation, q_delta)
        current_orientation = quat_normalize(current_orientation)

        # Convert to axis-angle, unify sign
        rvec = quat2axisangle(current_orientation)
        if prev_rvec is not None and np.dot(rvec, prev_rvec) < 0:
            rvec = -rvec
        prev_rvec = rvec
        
        pz -= z_offset

        # Build the 7D action
        current_position = np.array([px, py, pz], dtype=np.float32)
        all_actions[i, :3]  = [px, py, pz]
        all_actions[i, 3:6] = rvec
        all_actions[i][-1] = gripper_actions[i]

    # Add 10 buffer absolute actions that are copies of the last action
    for i in range(10):
        all_actions[num_actions+i] = all_actions[num_actions-1]
        
    return all_actions