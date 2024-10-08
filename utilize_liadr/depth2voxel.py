import numpy as np
import cv2
import open3d as o3d
import sys

class Depth2Voxel:
    def __init__(self, calibration_file):
        # Load calibration parameters
        self.K, self.RT, self.lidar_R, self.lidar_T = self.load_calibration(calibration_file)

    def load_calibration(self, calibration_file):
        with open(calibration_file, 'r') as f:
            lines = f.readlines()
            cam_K = np.array([float(x) for x in lines[0].split()[1:]]).reshape((3, 3))
            cam_RT = np.array([float(x) for x in lines[1].split()[1:]]).reshape((4, 4))
            lidar_R = np.array([float(x) for x in lines[2].split()[1:]]).reshape((3, 3))
            lidar_T = np.array([float(x) for x in lines[3].split()[1:]]).reshape((3, 1))
        return cam_K, cam_RT, lidar_R, lidar_T

    def load_depth_map(self, depth_map_file):
        # Load the depth map from a PNG file
        depth_map = cv2.imread(depth_map_file, cv2.IMREAD_UNCHANGED)
        
        # If the depth map is stored in 16-bit format, convert it to meters
        if depth_map.dtype == np.uint16:
            depth_map = depth_map.astype(np.float32) / 1000.0  # Assuming the depth map is in millimeters
        return depth_map

    def depth_to_point_cloud(self, depth_map):
        # Image dimensions
        h, w = depth_map.shape
        
        # Generate pixel grid coordinates
        i, j = np.indices((h, w))
        pixel_coords = np.stack((j, i, np.ones_like(i)), axis=-1)  # shape: (h, w, 3)
        
        # Reshape pixel coordinates for matrix operations
        pixel_coords = pixel_coords.reshape(-1, 3).T  # shape: (3, h * w)
        
        # Invert K matrix for transformation
        K_inv = np.linalg.inv(self.K)
        
        # Calculate normalized image coordinates
        normalized_coords = K_inv @ pixel_coords  # shape: (3, h * w)
        
        # Reshape depth map and convert it to 3D points
        depth_flat = depth_map.reshape(-1)  # shape: (h * w,)
        points_3D = normalized_coords * depth_flat  # Broadcasting depth across normalized coordinates
        
        # Apply camera extrinsics (cam_RT)
        points_3D_hom = np.vstack((points_3D, np.ones((1, points_3D.shape[1]))))  # Homogeneous coordinates
        points_3D_world = self.RT @ points_3D_hom  # shape: (4, h * w)
        
        # Convert to 3D points by removing homogeneous coordinate
        points_3D_world = points_3D_world[:3].T  # shape: (h * w, 3)

        return points_3D_world

    def create_point_cloud(self, points_3D):
        # Convert points to Open3D format
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_3D)
        return pcd

    def visualize_point_cloud(self, points_3D):
        # Create the point cloud
        pcd = self.create_point_cloud(points_3D)
        
        # Visualize the point cloud
        o3d.visualization.draw_geometries([pcd])
        
    def convert_to_voxel_grid(self, points_3D, voxel_size=0.05):
        # Convert point cloud to Open3D format
        pcd = self.create_point_cloud(points_3D)
        
        # Convert point cloud to Voxel Grid
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
        
        return voxel_grid

    def visualize_voxel_grid(self, voxel_grid):
        # Visualize the Voxel Grid
        o3d.visualization.draw_geometries([voxel_grid])