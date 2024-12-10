import open3d as o3d
import numpy as np

def apply_ransac(pcd):
    pcd = o3d.io.read_point_cloud(pcd)
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0, 0, 0])

    # origin pcd
    # o3d.visualization.draw_geometries([pcd, axes])

    plane_model, inliers = pcd.segment_plane(distance_threshold=0.3, ransac_n=3, num_iterations=500)

    inlier_cloud = pcd.select_by_index(inliers)
    outlier_cloud = pcd.select_by_index(inliers, invert=True)

    inlier_cloud.paint_uniform_color([0.5, 0.5, 0.5])
    outlier_cloud.paint_uniform_color([1, 0, 0])

    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud, axes])