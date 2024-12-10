import numpy as np
import open3d as o3d
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures

def apply_offroad_ransac(pcd):
    pcd = o3d.io.read_point_cloud(pcd)
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0, origin=[0, 0, 0])
    
    # degrees for polynomial feature ( only 2 in offroad)
    degress = [2]

    max_inliers = []
    max_inliers_degree = None
        
    for degree in degress:    
        pcd_points = np.array(pcd.points)

        X = pcd_points[:, :2]
        Z = pcd_points[:, 2]
        
        # Polynomial features for quadratic model
        poly_features = PolynomialFeatures(degree=degree, include_bias=True)
        X_poly = poly_features.fit_transform(X)
        
        # Apply RANSAC
        ransac = RANSACRegressor(min_samples=3*degree, residual_threshold=0.2, max_trials=500)
        ransac.fit(X_poly, Z)
        
        # Retrieve the inlier mask, then find the indices of inliers
        inlier_mask = ransac.inlier_mask_
        inliers = np.nonzero(inlier_mask)[0]

        if len(max_inliers) < len(inliers):        
            max_inliers = inliers
            max_inliers_degree = degree
            # Print the coefficients and intercept
            plane_model = ransac.estimator_.coef_
            # intercept = ransac.estimator_.intercept_

    inlier_cloud = pcd.select_by_index(inliers)
    outlier_cloud = pcd.select_by_index(inliers, invert=True)

    inlier_cloud.paint_uniform_color([0.5, 0.5, 0.5])
    outlier_cloud.paint_uniform_color([1, 0, 0])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud, axes])