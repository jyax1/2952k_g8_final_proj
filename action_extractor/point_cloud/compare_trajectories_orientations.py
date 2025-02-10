#!/usr/bin/env python

import argparse
import numpy as np

#######################################
# Quaternion / Rotation Helper Routines
#######################################
def rotation_matrix_to_quaternion(R):
    """
    Convert a 3x3 rotation matrix to a unit quaternion [x, y, z, w].
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
    norm_q = np.linalg.norm(q)
    if norm_q < 1e-12:
        raise ValueError("Invalid rotation matrix => near-zero norm quaternion.")
    return q / norm_q

def unify_quaternion_sign(q, reference):
    """
    Ensure q lies on the same 'hemisphere' as reference
    by flipping sign if necessary, to avoid +/- q confusion.
    """
    if np.dot(q, reference) < 0:
        return -q
    return q

def quaternion_angle(q1, q2):
    """
    Compute the angle difference (in radians) between two unit quaternions q1, q2.
    Sign-unify them so +q/-q are not mistaken for 180° difference.
    """
    # normalize
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    # unify sign
    q1 = unify_quaternion_sign(q1, q2)
    dot_val = np.dot(q1, q2)
    dot_val = np.clip(dot_val, -1.0, 1.0)
    angle = 2.0 * np.arccos(dot_val)
    return angle


#######################################
# Main
#######################################
def main(file1, file2):
    # Hard-coded base orientation:
    q_base = np.array([2.8445084e-02, 9.9697667e-01, -4.8518830e-04, 7.2306558e-02], dtype=np.float32)
    # Normalize base orientation just in case
    q_base /= np.linalg.norm(q_base)

    # Load Nx4x4 arrays
    poses1 = np.load(file1)  # shape (N, 4, 4)
    poses2 = np.load(file2)  # shape (N, 4, 4)

    if poses1.shape != poses2.shape:
        raise ValueError(f"Different shapes: {poses1.shape} vs {poses2.shape}")
    if poses1.shape[1:] != (4, 4):
        raise ValueError("Poses must be Nx4x4 arrays.")

    N = poses1.shape[0]
    print(f"Comparing {N} pose pairs with a base orientation...")

    for i in range(10):
        # Extract the 3x3 rotation
        R1 = poses1[i, :3, :3]
        R2 = poses2[i, :3, :3]

        # Convert each to quaternion
        q1 = rotation_matrix_to_quaternion(R1)
        q2 = rotation_matrix_to_quaternion(R2)

        # 1) difference between q1 and the base orientation
        rad_q1 = quaternion_angle(q1, q_base)
        deg_q1 = np.degrees(rad_q1)

        # 2) difference between q2 and the base orientation
        rad_q2 = quaternion_angle(q2, q_base)
        deg_q2 = np.degrees(rad_q2)

        # 3) difference between q1 and q2
        rad_12 = quaternion_angle(q1, q2)
        deg_12 = np.degrees(rad_12)

        print(f"Pose {i}: ")
        print(f"   - angle(q1, base) = {deg_q1:7.3f} deg")
        print(f"   - angle(q2, base) = {deg_q2:7.3f} deg")
        print(f"   - angle(q1, q2)   = {deg_12:7.3f} deg")

    print("Done.")

if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare Nx4x4 poses from two files. "
                    "Print orientation differences from a fixed base orientation "
                    "and from each other."
    )
    parser.add_argument("file1", help="Path to first .npy file of Nx4x4 poses.")
    parser.add_argument("file2", help="Path to second .npy file of Nx4x4 poses.")
    args = parser.parse_args()

    main(args.file1, args.file2)


#!/usr/bin/env python

# import numpy as np

# def rotation_matrix_to_quaternion(R):
#     """
#     Convert a 3x3 rotation matrix to a unit quaternion [x, y, z, w].
#     Assumes R is a proper rotation matrix (orthonormal, det=1).
#     """
#     M = np.asarray(R, dtype=np.float32)
#     trace = M[0,0] + M[1,1] + M[2,2]
#     if trace > 0:
#         s = 0.5 / np.sqrt(trace + 1.0)
#         w = 0.25 / s
#         x = (M[2,1] - M[1,2]) * s
#         y = (M[0,2] - M[2,0]) * s
#         z = (M[1,0] - M[0,1]) * s
#     else:
#         if M[0,0] > M[1,1] and M[0,0] > M[2,2]:
#             s = 2.0 * np.sqrt(max(1e-12, 1.0 + M[0,0] - M[1,1] - M[2,2]))
#             w = (M[2,1] - M[1,2]) / s
#             x = 0.25 * s
#             y = (M[0,1] + M[1,0]) / s
#             z = (M[0,2] + M[2,0]) / s
#         elif M[1,1] > M[2,2]:
#             s = 2.0 * np.sqrt(max(1e-12, 1.0 + M[1,1] - M[0,0] - M[2,2]))
#             w = (M[0,2] - M[2,0]) / s
#             x = (M[0,1] + M[1,0]) / s
#             y = 0.25 * s
#             z = (M[1,2] + M[2,1]) / s
#         else:
#             s = 2.0 * np.sqrt(max(1e-12, 1.0 + M[2,2] - M[0,0] - M[1,1]))
#             w = (M[1,0] - M[0,1]) / s
#             x = (M[0,2] + M[2,0]) / s
#             y = (M[1,2] + M[2,1]) / s
#             z = 0.25 * s

#     q = np.array([x, y, z, w], dtype=np.float32)
#     norm_q = np.linalg.norm(q)
#     if norm_q < 1e-12:
#         raise ValueError("Invalid rotation matrix => near-zero norm quaternion.")
#     return q / norm_q

# if __name__ == "__main__":
#     import sys

#     if len(sys.argv) < 2:
#         print(f"Usage: {sys.argv[0]} file1.npy")
#         sys.exit(1)

#     file1 = sys.argv[1]
#     # Load Nx4x4 array
#     poses1 = np.load(file1)
#     if poses1.shape[1:] != (4,4):
#         raise ValueError("Expected Nx4x4 array of transforms.")

#     # Extract rotation from the first pose
#     R = poses1[0, :3, :3]

#     # Convert to quaternion
#     q = rotation_matrix_to_quaternion(R)

#     # Print the [x, y, z, w] quaternion
#     print("First orientation in file1 (as quaternion [x, y, z, w]):")
#     print(q)
