#####################################################################
# Add these lines BEFORE other Panda3D imports to silence logs:
#####################################################################
import logging

# Silence "Loading model ..." logs from your python code
logging.getLogger("action_extractor.megapose.action_identifier_megapose").setLevel(logging.WARNING)

import panda3d.core as p3d
# Suppress "Known pipe types", "Xlib extension missing", etc.
p3d.load_prc_file_data("", "notify-level-glxdisplay fatal\n")
p3d.load_prc_file_data("", "notify-level-x11display fatal\n")
p3d.load_prc_file_data("", "notify-level-gsg fatal\n")

#####################################################################
# Normal imports
#####################################################################
import os
import numpy as np
import torch
import math
import imageio
import cv2
import zarr
import shutil
from glob import glob
from tqdm import tqdm
from pathlib import Path
from zarr import DirectoryStore, ZipStore

import robomimic.utils.obs_utils as ObsUtils
from robomimic.utils.file_utils import get_env_metadata_from_dataset
from robomimic.utils.env_utils import create_env_from_metadata

from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder

# Megapose
from megapose.utils.load_model import NAMED_MODELS, load_named_model
from megapose.utils.logging import get_logger
logger = get_logger(__name__)

# Our local imports
from action_extractor.utils.dataset_utils import (
    hdf5_to_zarr_parallel_with_progress,
    directorystore_to_zarr_zip,
)
from action_extractor.megapose.action_identifier_megapose import (
    make_object_dataset_from_folder,
    bounding_box_center,   # might be unused
    find_color_bounding_box,
    pixel_to_world,
    ActionIdentifierMegapose
)
from robosuite.utils.camera_utils import (
    get_camera_extrinsic_matrix,
    get_camera_intrinsic_matrix,
)

from scipy.spatial.transform import Rotation as R


def combine_videos_quadrants(top_left_video_path, top_right_video_path, 
                             bottom_left_video_path, bottom_right_video_path, 
                             output_path):
    """Combines four videos into a single quadrant layout video."""
    import imageio

    top_left_reader = imageio.get_reader(top_left_video_path)
    top_right_reader = imageio.get_reader(top_right_video_path)
    bottom_left_reader = imageio.get_reader(bottom_left_video_path)
    bottom_right_reader = imageio.get_reader(bottom_right_video_path)

    fps = top_left_reader.get_meta_data()["fps"]  # or handle if they differ

    with imageio.get_writer(output_path, fps=fps) as writer:
        while True:
            try:
                tl_frame = top_left_reader.get_next_data()
                tr_frame = top_right_reader.get_next_data()
                bl_frame = bottom_left_reader.get_next_data()
                br_frame = bottom_right_reader.get_next_data()
            except (StopIteration, IndexError):
                # Means at least one reader had no more frames
                break

            # Combine frames in a quadrant layout
            top = np.hstack([tl_frame, tr_frame])
            bottom = np.hstack([bl_frame, br_frame])
            combined = np.vstack([top, bottom])

            writer.append_data(combined)

    top_left_reader.close()
    top_right_reader.close()
    bottom_left_reader.close()
    bottom_right_reader.close()


def convert_mp4_to_webp(input_path, output_path, quality=80):
    """Converts MP4 video to WebP format using ffmpeg."""
    import subprocess
    import shutil
    
    ffmpeg_path = shutil.which('ffmpeg')
    if ffmpeg_path is None:
        raise RuntimeError("ffmpeg not found. Install via: sudo apt-get install ffmpeg")
    
    cmd = [
        ffmpeg_path, '-i', input_path,
        '-c:v', 'libwebp',
        '-quality', str(quality),
        '-lossless', '0',
        '-compression_level', '6',
        '-qmin', '0',
        '-qmax', '100',
        '-preset', 'default',
        '-loop', '0',
        '-vsync', '0',
        '-f', 'webp',
        output_path
    ]
    subprocess.run(cmd, check=True)


def axisangle2quat(axis_angle):
    """
    Converts an axis-angle vector [rx, ry, rz] into a quaternion [x, y, z, w].
    """
    angle = np.linalg.norm(axis_angle)
    if angle < 1e-12:
        # nearly zero rotation
        return np.array([0, 0, 0, 1], dtype=np.float32)
    axis = axis_angle / angle
    half = angle * 0.5
    return np.concatenate([
        axis * np.sin(half),
        [np.cos(half)]
    ]).astype(np.float32)


def quat2axisangle(quat):
    """
    Convert quaternion [x, y, z, w] to axis-angle [rx, ry, rz].
    """
    w = quat[3]
    # clamp w
    if w > 1.0:
        w = 1.0
    elif w < -1.0:
        w = -1.0

    angle = 2.0 * math.acos(w)
    den = math.sqrt(1.0 - w * w)
    if den < 1e-12:
        return np.zeros(3, dtype=np.float32)

    axis = quat[:3] / den
    return axis * angle


def quat_multiply(q1, q0):
    """
    Multiply two quaternions q1 * q0 in xyzw form => xyzw
    """
    x0, y0, z0, w0 = q0
    x1, y1, z1, w1 = q1
    return np.array([
        x1*w0 + y1*z0 - z1*y0 + w1*x0,
        -x1*z0 + y1*w0 + z1*x0 + w1*y0,
        x1*y0 - y1*x0 + z1*w0 + w1*z0,
        -x1*x0 - y1*y0 - z1*z0 + w1*w0,
    ], dtype=np.float32)


def quat_inv(q):
    """
    Inverse of unit quaternion [x, y, z, w] is the conjugate => [-x, -y, -z, w].
    """
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float32)

def quat_conjugate(q):
    """Conjugate of a unit quaternion [w, x, y, z] => [w, -x, -y, -z]."""
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)

def rotation_matrix_to_angle_axis(R):
    """
    Convert a 3x3 rotation matrix R into its angle-axis representation
    (a 3D vector whose direction is the rotation axis and magnitude is the rotation angle).
    """
    # Numerical stability: clamp values for arccos
    trace_val = np.trace(R)
    theta = np.arccos(
        np.clip((trace_val - 1.0) / 2.0, -1.0, 1.0)
    )
    
    # If angle is very small, approximate as zero rotation.
    if np.isclose(theta, 0.0):
        return np.zeros(3)
    
    # Compute rotation axis using the classic formula
    # axis = (1/(2*sin(theta))) * [R[2,1] - R[1,2], R[0,2] - R[2,0], R[1,0] - R[0,1]]
    axis = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1]
    ])
    axis = axis / (2.0 * np.sin(theta))
    
    # Angle-axis form is axis * angle
    angle_axis = axis * theta
    return angle_axis

def rotation_matrix_to_quaternion(R):
    """
    Convert a 3x3 rotation matrix R into a quaternion [x, y, z, w].
    Assumes R is a proper rotation matrix (orthonormal, det=1).
    """
    M = np.asarray(R, dtype=np.float32)
    trace = M[0,0] + M[1,1] + M[2,2]

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (M[2,1] - M[1,2]) * s
        y = (M[0,2] - M[2,0]) * s
        z = (M[1,0] - M[0,1]) * s
    else:
        # Find the major diagonal element
        if M[0,0] > M[1,1] and M[0,0] > M[2,2]:
            s = 2.0 * np.sqrt(max(1e-12, 1.0 + M[0,0] - M[1,1] - M[2,2]))
            w = (M[2,1] - M[1,2]) / s
            x = 0.25 * s
            y = (M[0,1] + M[1,0]) / s
            z = (M[0,2] + M[2,0]) / s
        elif M[1,1] > M[2,2]:
            s = 2.0 * np.sqrt(max(1e-12, 1.0 + M[1,1] - M[0,0] - M[2,2]))
            w = (M[0,2] - M[2,0]) / s
            x = (M[0,1] + M[1,0]) / s
            y = 0.25 * s
            z = (M[1,2] + M[2,1]) / s
        else:
            s = 2.0 * np.sqrt(max(1e-12, 1.0 + M[2,2] - M[0,0] - M[1,1]))
            w = (M[1,0] - M[0,1]) / s
            x = (M[0,2] + M[2,0]) / s
            y = (M[1,2] + M[2,1]) / s
            z = 0.25 * s

    q = np.array([x, y, z, w], dtype=np.float32)
    return quat_normalize(q)

def quaternion_conjugate(q):
    """
    Conjugate (inverse for a unit quaternion): [w, x, y, z] -> [w, -x, -y, -z].
    Assumes q is a unit quaternion.
    """
    w, x, y, z = q
    return np.array([w, -x, -y, -z], dtype=float)

def quaternion_norm(q):
    """
    Compute the Euclidean norm of a quaternion.
    """
    return np.sqrt(np.dot(q, q))

def quaternion_normalize(q):
    """
    Normalize a quaternion to make it a unit quaternion.
    """
    norm = quaternion_norm(q)
    if norm < 1e-12:
        raise ValueError("Cannot normalize a near-zero quaternion.")
    return q / norm

def compute_hand_to_world_transform(q_world, q_hand):
    """
    Given:
      - q_world:  orientation of an object in the world frame  (as [w, x, y, z])
      - q_hand: orientation of the same object in the hand frame (as [w, x, y, z])
    Returns:
      - q_WO: the quaternion that transforms an orientation from the hand frame to the world frame.
      
    i.e. q_WO = q_world * inverse(q_hand)
    """
    # Ensure both quaternions are unit quaternions
    q_world  = quaternion_normalize(q_world)
    q_hand = quaternion_normalize(q_hand)
    
    q_hand_inv = quaternion_conjugate(q_hand)  # inverse of a unit quaternion
    q_WO = quat_multiply(q_world, q_hand_inv)
    return quaternion_normalize(q_WO)

def transform_hand_orientation_to_world(q_WO, q_in_hand):
    """
    Transform an arbitrary orientation q_in_hand (hand frame)
    into the world frame using q_WO.
    
    Returns: q_in_world = q_WO * q_in_hand
    """
    # Normalize for safety, especially if there's floating-point drift
    q_in_hand = quaternion_normalize(q_in_hand)
    q_out = quat_multiply(q_WO, q_in_hand)
    return quaternion_normalize(q_out)

def quat_normalize(q):
    """Normalize a quaternion."""
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        raise ValueError("Cannot normalize near-zero quaternion.")
    return q / norm

def load_ground_truth_poses_as_actions(obs_group, env_camera0):
    """
    Modified version:
      - We compute delta quaternions between consecutive steps, 
      - then 'add' (i.e. multiply) that delta with 'current_orientation'.
      - Then convert that updated orientation to axis-angle for each action.
    """

    # (N, 3)
    pos_array = obs_group["robot0_eef_pos"][:]   # end-effector positions
    # (N, 4) => [qx, qy, qz, qw] in the world frame
    quat_array = obs_group["robot0_eef_quat"][:] 

    num_samples = pos_array.shape[0]

    # The environment's initial eef orientation (assumed [w, x, y, z] but verify shape/order).
    # Some internal environment variable -- adjust if needed.
    current_orientation = env_camera0.env.env._eef_xquat.astype(np.float32)
    current_orientation = quat_normalize(current_orientation)

    # We will have one fewer action than the number of samples,
    # because we form deltas between consecutive frames.
    num_actions = num_samples - 1

    all_actions = np.zeros((num_actions, 7), dtype=np.float32)

    # Keep track of the previous axis-angle to unify sign across consecutive frames
    prev_rvec = None

    for i in range(num_actions):
        # Current and next quaternions from data, reorder them to [w, x, y, z]
        q_i   = quat_array[i]     #   [qw, qx, qy, qz]
        q_i1  = quat_array[i + 1] # next [qw, qx, qy, qz]
        q_i   = quat_normalize(q_i)
        q_i1  = quat_normalize(q_i1)

        # 1) compute delta: how do we go from q_i to q_i1?
        #    delta = q_i1 * inv(q_i)
        q_delta = quat_multiply(quat_inv(q_i), q_i1)
        q_delta = quat_normalize(q_delta)

        # 2) "add" this delta to current_orientation (i.e. multiply in quaternion space)
        current_orientation = quat_multiply(current_orientation, q_delta)
        current_orientation = quat_normalize(current_orientation)

        # 3) Convert to axis-angle
        rvec = quat2axisangle(current_orientation)

        # 4) optional sign unify (avoid ± flips in axis representation)
        if prev_rvec is not None and np.dot(rvec, prev_rvec) < 0:
            rvec = -rvec
        prev_rvec = rvec

        # For position, you can choose either pos_array[i] or pos_array[i+1]. 
        # Typically we take the next position: 
        px, py, pz = pos_array[i+1]

        all_actions[i, 0:3] = [px, py, pz]
        all_actions[i, 3:6] = rvec
        all_actions[i, 6]   = 1.0  # e.g., keep gripper = open = 1

    return all_actions

def _smooth_positions(poses, jump_threshold):
    """
    Given a list of 4x4 SE(3) poses, produce a smoothed array of shape (N,3)
    that avoids large jumps. We'll do a single pass with the following logic:

    - out[0] = poses[0].translation
    - For i in [0 .. N-2]:
        1) Check dist between out[i] and poses[i+1].translation.
        2) If <= jump_threshold, set out[i+1] to poses[i+1].translation (no big jump).
        3) If > jump_threshold, search for a j in [i+2..end] such that
           distance to out[i] is <= jump_threshold. If found, linearly
           interpolate from i..j. If not found, we do a fallback approach:
             - If i>0, we do velocity-based extrapolation from out[i-1]->out[i].
             - If i=0 (the first is an outlier), we do velocity-based interpolation
               from the second pose onward or just hold the second pose, etc.

    This ensures each step doesn't exceed the threshold unless we do an
    interpolation to jump over it. 
    Additionally, we handle the special case: 
    if i=0 is too far from *all* subsequent poses, we assume the first
    is the outlier and try to extrapolate from the next positions.
    """
    num_samples = len(poses)
    out = np.zeros((num_samples, 3), dtype=np.float32)
    out[0] = poses[0][:3, 3]  # the first pose's translation

    i = 0
    while i < num_samples - 1:
        # The target is the actual pose[i+1] translation
        target_pos = poses[i+1][:3, 3]
        dist_i_next = np.linalg.norm(target_pos - out[i])

        if dist_i_next <= jump_threshold:
            # No big jump => keep the actual next position
            out[i+1] = target_pos
            i += 1
        else:
            # There's a large jump from out[i] -> poses[i+1].
            # Search for j in [i+2 .. end] so that dist(out[i], poses[j]) <= jump_threshold
            j = None
            for test_idx in range(i+2, num_samples):
                test_pos = poses[test_idx][:3, 3]
                dist_i_test = np.linalg.norm(test_pos - out[i])
                if dist_i_test <= jump_threshold:
                    j = test_idx
                    break

            if j is None:
                # No good j found => fallback approach
                if i >= 1:
                    # We do velocity-based extrapolation from out[i-1] -> out[i]
                    velocity = out[i] - out[i-1]   # previous step
                    steps_left = (num_samples - 1) - i
                    for k in range(1, steps_left + 1):
                        frac = k / (steps_left + 1e-8)  # optional
                        out[i + k] = out[i] + frac * velocity
                    i = num_samples
                else:
                    # i=0 => means the very first pose is an outlier
                    if num_samples > 2:
                        # We'll try to define a velocity from poses[1]->poses[2]
                        pos1 = poses[1][:3, 3]
                        pos2 = poses[2][:3, 3]
                        velocity_12 = pos2 - pos1  # direction from second to third pose
                        # Fill out[0], out[1], ... by stepping from pos1 forward
                        out[0] = pos1  # override first
                        out[1] = pos1
                        steps_left = num_samples - 2
                        for k in range(1, steps_left+1):
                            frac = k / float(steps_left)
                            out[k+1] = pos1 + frac * velocity_12
                        i = num_samples
                    else:
                        # Only 2 frames total => just hold the second
                        out[1] = poses[1][:3, 3]
                        i = num_samples
            else:
                # We found a j s.t. dist(out[i], poses[j]) <= jump_threshold
                # We'll linearly interpolate from out[i] to poses[j] over (j - i) steps
                j_pos = poses[j][:3, 3]
                steps = j - i
                for m in range(i+1, j+1):
                    frac = (m - i) / float(steps)
                    out[m] = out[i] + frac * (j_pos - out[i])
                i = j

    return out

def _smooth_positions_side(
    poses_side,
    threshold=3.0,
    axes_to_smooth=(2,)  # e.g. (0,2) if you want to smooth x and z
):
    """
    Given a list/array of 4x4 SE(3) side-camera poses (N of them),
    produce an (N, 3) array of translations [x, y, z] in which we optionally
    detect outliers + interpolate for the specified 'axes_to_smooth'.

    For each axis in axes_to_smooth:
      1) Extract its values => arr[i] = poses_side[i][:3,3][axis].
      2) Detect outliers with median + MAD using 'threshold' * MAD.
      3) Mark outliers as NaN, linearly interpolate them.
      4) If leading or trailing frames are NaN, clamp to nearest valid.

    For any axis *not* in axes_to_smooth, we simply copy raw values from poses_side.

    Args:
      poses_side: shape (N, 4, 4), list or np.array of side-camera transforms.
      threshold: how many "MADs" away from median() to consider an outlier.
      axes_to_smooth: tuple of axes (0 => x, 1 => y, 2 => z) to smooth.

    Returns:
      out: shape (N, 3),
        out[i,0] => x,
        out[i,1] => y,
        out[i,2] => z.
      The selected axes have outliers removed and linearly interpolated.
      Non-selected axes are copied verbatim from the original poses_side.
    """

    N = len(poses_side)
    out = np.zeros((N, 3), dtype=np.float32)
    if N == 0:
        return out

    # --- Step 1: Extract raw X, Y, Z arrays ---
    x_vals = np.array([p[0,3] for p in poses_side], dtype=np.float32)
    y_vals = np.array([p[1,3] for p in poses_side], dtype=np.float32)
    z_vals = np.array([p[2,3] for p in poses_side], dtype=np.float32)

    # We'll build a dictionary for convenience
    #  axis=0 => x_vals, axis=1 => y_vals, axis=2 => z_vals
    raw_data = {
        0: x_vals,
        1: y_vals,
        2: z_vals,
    }

    # This will store the final smoothed 1D arrays for x,y,z
    cleaned_data = {
        0: raw_data[0].copy(),
        1: raw_data[1].copy(),
        2: raw_data[2].copy(),
    }

    # --- Step 2: For each axis we want to smooth, detect outliers + interpolate ---
    for axis in axes_to_smooth:
        arr = raw_data[axis].copy()  # raw 1D data for this axis
        arr_clean = _remove_outliers_and_interpolate_1d(arr, threshold=threshold)
        cleaned_data[axis] = arr_clean

    # --- Step 3: Build final (N,3) output from cleaned_data ---
    out[:, 0] = cleaned_data[0]
    out[:, 1] = cleaned_data[1]
    out[:, 2] = cleaned_data[2]

    return out


def _remove_outliers_and_interpolate_1d(values, threshold=3.0):
    """
    1) Detect outliers via median + MAD => mark as NaN
    2) Replace leading/trailing NaNs by clamping to nearest valid
    3) Interpolate interior NaN blocks linearly

    :param values: 1D np.ndarray of shape (N,)
    :param threshold: how many MADs from the median => outlier
    :return: 1D np.ndarray (N,) with outliers replaced by linear interpolation
    """
    out_arr = values.copy()
    N = len(values)
    if N == 0:
        return out_arr

    # -- Step A: median + MAD outlier detection
    median_val = np.median(out_arr)
    abs_dev = np.abs(out_arr - median_val)
    mad_val = np.median(abs_dev)

    if mad_val < 1e-12:
        # all nearly the same => no outliers
        return out_arr

    outlier_mask = (abs_dev > threshold * mad_val)
    out_arr[outlier_mask] = np.nan

    # -- Step B: handle case all outliers
    valid_mask = ~np.isnan(out_arr)
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) == 0:
        # fill everything with median_val or just return
        out_arr[:] = median_val
        return out_arr

    # -- Step C: clamp leading/trailing NaNs
    first_valid = valid_indices[0]
    last_valid = valid_indices[-1]

    if first_valid > 0:
        out_arr[:first_valid] = out_arr[first_valid]
    if last_valid < (N-1):
        out_arr[last_valid+1:] = out_arr[last_valid]

    # -- Step D: linearly interpolate each contiguous NaN block
    nan_indices = np.where(np.isnan(out_arr))[0]
    idx_ptr = 0
    while idx_ptr < len(nan_indices):
        start_invalid = nan_indices[idx_ptr]
        block_start = start_invalid
        block_end = start_invalid

        # find the contiguous block of NaNs
        while (block_end + 1 < N) and np.isnan(out_arr[block_end + 1]):
            block_end += 1

        # block [block_start..block_end]
        left_valid = block_start - 1
        right_valid = block_end + 1

        # skip to next block
        idx_ptr += (block_end - block_start + 1)

        # check boundaries
        if left_valid < 0 or right_valid >= N:
            # means we've already clamped them, so skip
            continue

        if np.isnan(out_arr[left_valid]) or np.isnan(out_arr[right_valid]):
            # can't interpolate if either side is invalid
            continue

        left_val = out_arr[left_valid]
        right_val = out_arr[right_valid]
        span = (block_end - block_start) + 1

        for k in range(span):
            frac = float(k+1)/(span+1)
            out_arr[block_start + k] = left_val + frac*(right_val - left_val)

    return out_arr


def poses_to_absolute_actions(
    poses, 
    poses_side, 
    fingers_distances, 
    env_camera0,
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
    smoothed_positions = _smooth_positions_side(poses, axes_to_smooth=(0,1,2))
    smoothed_positions_side = _smooth_positions_side(poses_side, axes_to_smooth=(0,1,2))
    # for i in range(len(poses)):
    #     print(poses[i][:3,3], smoothed_positions[i])
        
    # exit()
        

    # 2) Start from environment's known initial eef quaternion 
    #    (assuming it's [w, x, y, z], confirm shape/order as needed).
    starting_orientation = env_camera0.env.env._eef_xquat.astype(np.float32)
    current_orientation = env_camera0.env.env._eef_xquat.astype(np.float32)
    current_orientation = quat_normalize(current_orientation)
    current_position = env_camera0.env.env._eef_xpos.astype(np.float32)

    # We'll have num_actions = num_samples - 1
    num_actions = num_samples - 1
    all_actions = np.zeros((num_actions, 7), dtype=np.float32)

    prev_rvec = None
    z_offset = None

    for i in range(num_actions):
        # --- Orientation from poses[i] -> poses[i+1] (the "real" rotation) ---
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

        # --- Position from the precomputed 'smoothed_positions' ---
        if z_offset is None:
            z_offset = smoothed_positions[i][2] - current_position[2]
            
        px, py, pz = smoothed_positions[i+1]
        px, py, pz  = smoothed_positions_side[i+1]
        
        # print(f"pz_front: {pz_front}, pz_side: {pz}")
        pz -= z_offset
        
        print('px:', px, 'py:', py, 'pz:', pz)

        # Build the 7D action
        all_actions[i, :3]  = [px, py, pz]
        all_actions[i, 3:6] = quat2axisangle(starting_orientation)
        all_actions[i, 6]   = 1.0  # e.g., "gripper = open"

    return all_actions

def imitate_trajectory_with_action_identifier(
    dataset_path="/home/yilong/Documents/policy_data/lift/lift_smaller_2000",
    hand_mesh_dir="/home/yilong/Documents/action_extractor/action_extractor/megapose/panda_hand_mesh",
    output_dir="/home/yilong/Documents/action_extractor/debug/megapose_lift_smaller_2000",
    num_demos=100,
    save_webp=False,
    cameras=["frontview_image", "sideview_image"],
    batch_size=40,
):
    # 0) Output dir
    os.makedirs(output_dir, exist_ok=True)

    # 1) Build object dataset
    hand_object_dataset = make_object_dataset_from_folder(Path(hand_mesh_dir))

    # 2) Load model once
    model_name = "megapose-1.0-RGB-multi-hypothesis"
    model_info = NAMED_MODELS[model_name]
    logger.info(f"Loading model {model_name} once at script start.")
    hand_pose_estimator = load_named_model(model_name, hand_object_dataset).cuda()

    # 3) Preprocess dataset => zarr
    sequence_dirs = glob(f"{dataset_path}/**/*.hdf5", recursive=True)
    for seq_dir in sequence_dirs:
        ds_dir = seq_dir.replace(".hdf5", ".zarr")
        zarr_path = seq_dir.replace(".hdf5", ".zarr.zip")
        if not os.path.exists(zarr_path):
            hdf5_to_zarr_parallel_with_progress(seq_dir, max_workers=16)
            store = DirectoryStore(ds_dir)
            root_z = zarr.group(store, overwrite=False)
            store.close()
            directorystore_to_zarr_zip(ds_dir, zarr_path)
            shutil.rmtree(ds_dir)

    # 4) Collect Zarr files
    zarr_files = glob(f"{dataset_path}/**/*.zarr.zip", recursive=True)
    stores = [ZipStore(zarr_file, mode="r") for zarr_file in zarr_files]
    roots = [zarr.group(store) for store in stores]

    # 5) Create environment
    env_meta = get_env_metadata_from_dataset(dataset_path=sequence_dirs[0])
    env_meta['env_kwargs']['controller_configs']['control_delta'] = False
    env_meta['env_kwargs']['controller_configs']['type'] = 'OSC_POSE'

    obs_modality_specs = {
        "obs": {
            "rgb": cameras,
            "depth": [f"{cam.split('_')[0]}_depth" for cam in cameras],
        }
    }
    ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs)

    env_camera0 = create_env_from_metadata(env_meta=env_meta, render_offscreen=True)

    example_image = roots[0]["data"]["demo_0"]["obs"]["frontview_image"][0]
    camera_height, camera_width = example_image.shape[:2]

    frontview_K = get_camera_intrinsic_matrix(env_camera0.env.sim,
                                              camera_name="frontview",
                                              camera_height=camera_height,
                                              camera_width=camera_width)
    sideview_K   = get_camera_intrinsic_matrix(env_camera0.env.sim,
                                               camera_name="sideview",
                                               camera_height=camera_height,
                                               camera_width=camera_width)
    frontview_R  = get_camera_extrinsic_matrix(env_camera0.env.sim, camera_name="frontview")
    sideview_R   = get_camera_extrinsic_matrix(env_camera0.env.sim, camera_name="sideview")

    env_camera0 = VideoRecordingWrapper(
        env_camera0,
        video_recoder=VideoRecorder.create_h264(fps=20, codec="h264", input_pix_fmt="rgb24", crf=22),
        steps_per_render=1,
        width=camera_width,
        height=camera_height,
        mode="rgb_array",
        camera_name=cameras[0].split("_")[0],
    )

    env_camera1 = create_env_from_metadata(env_meta=env_meta, render_offscreen=True)
    env_camera1 = VideoRecordingWrapper(
        env_camera1,
        video_recoder=VideoRecorder.create_h264(fps=20, codec="h264", input_pix_fmt="rgb24", crf=22),
        steps_per_render=1,
        width=camera_width,
        height=camera_height,
        mode="rgb_array",
        camera_name=cameras[1].split("_")[0],
    )

    # 6) Optionally build the ActionIdentifierMegapose
    action_identifier = ActionIdentifierMegapose(
        pose_estimator=hand_pose_estimator,
        frontview_R=frontview_R,
        frontview_K=frontview_K,
        sideview_R=sideview_R,
        sideview_K=sideview_K,
        model_info=model_info,
        batch_size=batch_size,
        scale_translation=80.0,
    )

    n_success = 0
    total_n = 0
    results = []

    # 7) Loop over demos
    for root_z in roots:
        demos = list(root_z["data"].keys())[:num_demos] if num_demos else list(root_z["data"].keys())
        for demo in tqdm(demos, desc="Processing demos"):
            demo_id = demo.replace("demo_", "")
            upper_left_video_path  = os.path.join(output_dir, f"{demo_id}_upper_left.mp4")
            upper_right_video_path = os.path.join(output_dir, f"{demo_id}_upper_right.mp4")
            lower_left_video_path  = os.path.join(output_dir, f"{demo_id}_lower_left.mp4")
            lower_right_video_path = os.path.join(output_dir, f"{demo_id}_lower_right.mp4")
            combined_video_path    = os.path.join(output_dir, f"{demo_id}_combined.mp4")

            obs_group   = root_z["data"][demo]["obs"]
            num_samples = obs_group["frontview_image"].shape[0]

            # Left videos
            upper_left_frames = [obs_group["frontview_image"][i] for i in range(num_samples)]
            lower_left_frames = [obs_group["sideview_image"][i] for i in range(num_samples)]
            with imageio.get_writer(upper_left_video_path, fps=20) as writer:
                for frame in upper_left_frames:
                    writer.append_data(frame)
            with imageio.get_writer(lower_left_video_path, fps=20) as writer:
                for frame in lower_left_frames:
                    writer.append_data(frame)

            # ---- We want ground-truth absolute actions in eef_site coords ----
            front_frames_list = [obs_group["frontview_image"][i] for i in range(num_samples)]
            front_depth_list = [obs_group["frontview_depth"][i] for i in range(num_samples)]
            side_frames_list = [obs_group["sideview_image"][i] for i in range(num_samples)]
            
            cache_file = "hand_poses_cache.npz"
            if os.path.exists(cache_file):
                # Load from disk
                print(f"Loading cached poses from {cache_file} ...")
                data = np.load(cache_file, allow_pickle=True)
                all_hand_poses_world = data["all_hand_poses_world"]
                all_fingers_distances = data["all_fingers_distances"]
                all_hand_poses_world_from_side = data["all_hand_poses_world_from_side"]
            else:
                # No cache => run the expensive inference once
                print("No cache found. Running inference to get all_hand_poses_world...")
                (all_hand_poses_world,
                all_fingers_distances,
                all_hand_poses_world_from_side) = action_identifier.get_all_hand_poses_finger_distances_with_side(
                    front_frames_list,
                    front_depth_list=front_depth_list,
                    side_frames_list=side_frames_list
                )
                # Save to disk for future runs
                np.savez(
                    cache_file,
                    all_hand_poses_world=all_hand_poses_world,
                    all_fingers_distances=all_fingers_distances,
                    all_hand_poses_world_from_side=all_hand_poses_world_from_side
                )
           
            actions_for_demo = poses_to_absolute_actions(all_hand_poses_world, all_hand_poses_world_from_side, all_fingers_distances, env_camera0)
            # actions_for_demo = load_ground_truth_poses_as_actions(obs_group, env_camera0)
            

            initial_state = root_z["data"][demo]["states"][0]

            # 1) top-right video
            env_camera0.reset()
            env_camera0.reset_to({"states": initial_state})
            env_camera0.file_path = upper_right_video_path
            env_camera0.step_count = 0
            for action in actions_for_demo:
                env_camera0.step(action)
            env_camera0.video_recoder.stop()
            env_camera0.file_path = None

            # 2) bottom-right video
            env_camera1.reset()
            env_camera1.reset_to({"states": initial_state})
            env_camera1.file_path = lower_right_video_path
            env_camera1.step_count = 0
            for action in actions_for_demo:
                env_camera1.step(action)
            env_camera1.video_recoder.stop()
            env_camera1.file_path = None

            # success check
            success = env_camera0.is_success()["task"]
            if success:
                n_success += 1
            total_n += 1
            results.append(f"{demo}: {'success' if success else 'failed'}")

            # combine quadrant
            combine_videos_quadrants(
                upper_left_video_path,
                upper_right_video_path,
                lower_left_video_path,
                lower_right_video_path,
                combined_video_path
            )
            os.remove(upper_left_video_path)
            os.remove(upper_right_video_path)
            os.remove(lower_left_video_path)
            os.remove(lower_right_video_path)

    success_rate = (n_success / total_n)*100 if total_n else 0
    results.append(f"\nFinal Success Rate: {n_success}/{total_n} => {success_rate:.2f}%")

    with open(os.path.join(output_dir, "trajectory_results.txt"), "w") as f:
        f.write("\n".join(results))

    if save_webp:
        print("Converting to webp...")
        mp4_files = glob(os.path.join(output_dir, "*.mp4"))
        for mp4_file in tqdm(mp4_files):
            webp_file = mp4_file.replace(".mp4", ".webp")
            try:
                convert_mp4_to_webp(mp4_file, webp_file)
                os.remove(mp4_file)
            except Exception as e:
                print(f"Error converting {mp4_file}: {e}")

    print(f"Wrote results to {os.path.join(output_dir, 'trajectory_results.txt')}")


if __name__ == "__main__":
    imitate_trajectory_with_action_identifier(
        dataset_path="/home/yilong/Documents/policy_data/square_d0/raw/test/test",
        hand_mesh_dir="/home/yilong/Documents/action_extractor/action_extractor/megapose/panda_hand_mesh",
        output_dir="/home/yilong/Documents/action_extractor/debug/megapose_gt",
        num_demos=3,
        save_webp=False,
        batch_size=40
    )
