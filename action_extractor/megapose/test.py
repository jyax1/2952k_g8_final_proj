import open3d as o3d
import numpy as np

def get_ply_dimensions(ply_path):
    # Load the point cloud or mesh
    pcd = o3d.io.read_point_cloud(ply_path)  # Use read_triangle_mesh() if it's a mesh

    # Compute the axis-aligned bounding box (AABB)
    aabb = pcd.get_axis_aligned_bounding_box()

    # Get dimensions (width, height, length)
    dimensions = aabb.get_extent()  # [width, height, length]

    return dimensions

# Example usage
ply_file = "action_extractor/megapose/panda_hand_mesh/panda-hand.ply"
dims = get_ply_dimensions(ply_file)
print(f"Width: {dims[0]:.3f}, Height: {dims[1]:.3f}, Length: {dims[2]:.3f}")

