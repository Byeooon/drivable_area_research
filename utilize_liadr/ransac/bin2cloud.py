import numpy as np
import open3d as o3d

def read_bin_point_cloud(filename):
    points = np.fromfile(filename, dtype=np.float32).reshape(-1, 5)[:, :3]  # using only x, y, z
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd