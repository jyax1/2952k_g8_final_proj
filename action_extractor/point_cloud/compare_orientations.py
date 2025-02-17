import math

def quaternion_orientation_difference_degrees(q1, q2):
    """
    Calculate the orientation difference in degrees between two quaternions q1 and q2.

    Args:
        q1 (tuple/list): The first quaternion, formatted as (w, x, y, z).
        q2 (tuple/list): The second quaternion, formatted as (w, x, y, z).

    Returns:
        float: The difference in orientation (degrees) between q1 and q2.
    """
    # 1. Normalize each quaternion
    def normalize(q):
        w, x, y, z = q
        norm = math.sqrt(w**2 + x**2 + y**2 + z**2)
        return (w / norm, x / norm, y / norm, z / norm)
    
    q1_norm = normalize(q1)
    q2_norm = normalize(q2)

    # 2. Compute the dot product q1 • q2 (where q = (w, x, y, z))
    dot = (
        q1_norm[0] * q2_norm[0] +
        q1_norm[1] * q2_norm[1] +
        q1_norm[2] * q2_norm[2] +
        q1_norm[3] * q2_norm[3]
    )

    # 3. Clamp the dot product to prevent numerical issues in acos (must be in [-1, 1])
    dot = max(-1.0, min(1.0, dot))

    # 4. Calculate the angle between the two quaternions in radians
    angle_radians = 2.0 * math.acos(dot)

    # 5. Convert radians to degrees
    angle_degrees = math.degrees(angle_radians)

    return angle_degrees


# Example usage
if __name__ == "__main__":
    # Suppose we have two quaternions (in w, x, y, z format)
    q_a = (0.7122174,  0.69890285, 0.04987878, 0.04234895)  # ~ Rotated 45 degrees about X
    q_b = (0.61854268,  0.78458513,  0.03966061, -0.01606732)      # Identity quaternion

    difference_deg = quaternion_orientation_difference_degrees(q_a, q_b)
    print(f"Difference in orientation (degrees): {difference_deg:.2f}")