import numpy as np
from action_extractor.utils.angles_utils import *

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
    smooth=True,
    control_freq=30,
    policy_freq=10
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
         compute delta quaternion from poses[i]->poses[i+1].
         Accumulate into `current_orientation`.
         Convert to axis-angle (with sign unification).
    3) BUILD ACTIONS:
       - For each i in [0 .. num_samples-2], the final action 
         uses `smoothed_positions[i+1]` as [px,py,pz] 
         and the just-computed orientation as [rx,ry,rz].
       - Set gripper=1.0 or your logic.
    4) REPEAT:
       - Each computed action is repeated steps_per_policy times so that if
         steps_per_policy = 2, a₀, a₁, a₂, … becomes a₀, a₀, a₁, a₁, a₂, a₂, ….
       
    Returns:
       all_actions of shape ((num_samples-1)*steps_per_policy, 7).
    """
    import numpy as np

    num_samples = len(poses)
    if num_samples < 2:
        return np.zeros((0, 7), dtype=np.float32)

    # 1) Compute a smoothed set of positions.
    if smooth:
        smoothed_positions = smooth_positions(poses, dist_threshold=0.15)
    else:
        smoothed_positions = np.array([pose[:3, 3] for pose in poses], dtype=np.float32)

    # Compute steps_per_policy and preallocate the final action array.
    steps_per_policy = control_freq // policy_freq
    num_actions = num_samples - 1  # one action per pose transition.
    total_control_steps = num_actions * steps_per_policy
    all_actions = np.zeros((total_control_steps, 7), dtype=np.float32)

    # Compute a position offset (if needed) so that the first pose aligns with the initial eef position.
    prev_rvec = None

    # Loop over each policy step and compute one absolute action.
    for i in range(num_actions):
        # --- Compute target position from the precomputed smoothed positions ---
        px, py, pz = smoothed_positions[i+1]
        position = np.array([px, py, pz])

        # --- Compute orientation: poses[i] -> poses[i+1] ---
        R_i1 = poses[i+1][:3, :3]
        
        rvec = rotation_matrix_to_angle_axis(R_i1)

        if prev_rvec is not None and np.dot(rvec, prev_rvec) < 0:
            rvec = -rvec
        prev_rvec = rvec

        # Get the gripper action corresponding to this policy step.
        gripper = gripper_actions[i]

        # Build the 7D action vector for this policy step.
        action = np.zeros(7, dtype=np.float32)
        action[:3] = position
        action[3:6] = rvec
        # action[3:6] = current_orientation_angle
        action[6] = gripper

        # Determine the indices in the final all_actions array where this action is repeated.
        start_idx = i * steps_per_policy
        end_idx = start_idx + steps_per_policy
        all_actions[start_idx:end_idx, :] = np.tile(action, (steps_per_policy, 1))

    return all_actions


def poses_to_delta_actions(
    poses, 
    gripper_actions,
    translation_scaling=75.0,
    rotation_scaling=9.0,
    smooth=True
):
    """
    Convert a sequence of 4x4 pose matrices into delta actions (position + orientation).
    The orientation is represented as the axis-angle difference between consecutive poses.

    Steps:
    1) SMOOTH POSITIONS (optional):
       - We either smooth the positions using a user-defined `smooth_positions` function
         or directly take the translation part of each pose.
       - The result is `smoothed_positions` (N x 3).
    2) ORIENTATION DIFFERENCE:
       - For each consecutive pair of rotation matrices R_i -> R_i+1, compute the 
         quaternion difference q_delta = inv(q_i) * q_i+1, then convert to axis-angle.
       - Unify sign with the previous axis-angle (so orientation does not flip signs).
    3) BUILD DELTA ACTIONS:
       - Delta translation: smoothed_positions[i+1] - smoothed_positions[i]
       - Delta rotation: axis-angle of q_delta
       - Gripper action: from gripper_actions[i]
    4) BUFFER AT THE END:
       - Repeat the last action 10 times at the end.

    Args:
        poses (list of ndarray): Each element is a 4x4 transformation matrix.
        gripper_actions (array-like): 1D array of gripper values (length must match len(poses)).
        smooth (bool): If True, uses a smoothing function on positions.

    Returns:
        all_actions (ndarray): shape (num_samples - 1 + 10, 7)
                               Each row = [dx, dy, dz, rx, ry, rz, gripper]
    """
    
    num_samples = len(poses)
    if num_samples < 2:
        # Not enough poses to form a delta
        return np.zeros((0, 7), dtype=np.float32)

    # 1) Compute a smoothed set of positions
    if smooth:
        smoothed_positions = smooth_positions(poses, dist_threshold=0.15)
    else:
        smoothed_positions = np.array([pose[:3, 3] for pose in poses], dtype=np.float32)

    # We'll have num_actions = num_samples - 1
    num_actions = num_samples - 1
    all_actions = np.zeros((num_actions, 7), dtype=np.float32)

    for i in range(num_actions):
        # --- Position delta ---
        dx, dy, dz = smoothed_positions[i+1] - smoothed_positions[i]

        # --- Orientation delta: poses[i] -> poses[i+1] ---
        R_i  = poses[i][:3, :3]
        R_i1 = poses[i+1][:3, :3]

        q_i  = rotation_matrix_to_quaternion(R_i)
        q_i1 = rotation_matrix_to_quaternion(R_i1)

        q_i   = quat_normalize(q_i)
        q_i1  = quat_normalize(q_i1)
        q_inv = quat_inv(q_i)
        
        q_delta = quat_multiply(q_i1, q_inv)
        q_delta = quat_normalize(q_delta)

        # Convert to axis-angle, unify sign with previous
        rvec = quat2axisangle(q_delta)
        
        rvec *= rotation_scaling

        # Build the 7D delta action
        all_actions[i, :3]  = [dx, dy, dz]
        all_actions[i, :3]  *= translation_scaling
        all_actions[i, 3:6] = rvec
        all_actions[i, 6]   = gripper_actions[i]
        
    return all_actions


def poses_to_delta_actions_lr(
    poses, 
    gripper_actions,
    smooth=True,
    mapping_coef_file="reg_coef.npy",
    mapping_intercept_file="reg_intercept.npy"
):
    """
    Convert a sequence of 4x4 pose matrices into delta actions (position + orientation),
    then map these computed actions to the true actions using a linear regression mapping.
    The mapping is defined as: true_action = computed_action @ coef.T + intercept

    Steps:
    1) SMOOTH POSITIONS (optional):
       - Either smooth the positions using a user-defined `smooth_positions` function
         or directly take the translation part of each pose.
    2) ORIENTATION DIFFERENCE:
       - For each consecutive pair of rotation matrices R_i -> R_i+1,
         compute the quaternion difference (q_delta = inv(q_i) * q_i+1) and convert to axis-angle.
    3) BUILD DELTA ACTIONS:
       - Compute delta translation: smoothed_positions[i+1] - smoothed_positions[i]
       - Compute delta rotation: axis-angle representation of q_delta.
       - Set gripper action: from gripper_actions[i]
    4) MAP TO TRUE ACTIONS:
       - Load the regression parameters from .npy files and transform the computed actions.
    5) BUFFER AT THE END:
       - Append 10 copies of the last action.
       
    Args:
        poses (list of ndarray): Each element is a 4x4 transformation matrix.
        gripper_actions (array-like): 1D array of gripper values (length must match len(poses)).
        smooth (bool): If True, uses a smoothing function on positions.
        mapping_coef_file (str): Path to the .npy file containing the regression coefficients.
        mapping_intercept_file (str): Path to the .npy file containing the regression intercept.
    
    Returns:
        all_actions (ndarray): shape (num_samples - 1 + 10, 7)
                               Each row = [dx, dy, dz, rx, ry, rz, gripper]
    """
    import numpy as np

    num_samples = len(poses)
    if num_samples < 2:
        return np.zeros((0, 7), dtype=np.float32)

    # 1) Compute smoothed positions if needed.
    if smooth:
        smoothed_positions = smooth_positions(poses, dist_threshold=0.15)
    else:
        smoothed_positions = np.array([pose[:3, 3] for pose in poses], dtype=np.float32)

    num_actions = num_samples - 1
    computed_actions = np.zeros((num_actions, 7), dtype=np.float32)

    for i in range(num_actions):
        # --- Position delta ---
        dx, dy, dz = smoothed_positions[i+1] - smoothed_positions[i]

        # --- Orientation delta: poses[i] -> poses[i+1] ---
        R_i  = poses[i][:3, :3]
        R_i1 = poses[i+1][:3, :3]

        q_i  = rotation_matrix_to_quaternion(R_i)
        q_i1 = rotation_matrix_to_quaternion(R_i1)

        q_i   = quat_normalize(q_i)
        q_i1  = quat_normalize(q_i1)
        q_inv = quat_inv(q_i)
        
        # Note: the order here should match what you used during regression.
        q_delta = quat_multiply(q_i1, q_inv)
        q_delta = quat_normalize(q_delta)

        # Convert to axis-angle.
        rvec = quat2axisangle(q_delta)

        # Build the 7D computed action (without hand-tuned scaling).
        computed_actions[i, :3]  = [dx, dy, dz]
        computed_actions[i, 3:6] = rvec
        computed_actions[i, 6]   = gripper_actions[i]

    # 4) Load regression parameters and apply mapping:
    coef = np.load(mapping_coef_file)
    intercept = np.load(mapping_intercept_file)
    # Apply the mapping: for each computed action x, we compute y = x @ coef.T + intercept
    mapped_actions = computed_actions @ coef.T + intercept
    
    return mapped_actions