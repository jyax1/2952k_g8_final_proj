import numpy as np
import math

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