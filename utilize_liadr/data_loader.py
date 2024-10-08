import os
import cv2
import open3d as o3d
import numpy as np

class DataLoader:
    def __init__(self, image_dir, pcd_dir):
        self.image_dir = image_dir
        self.pcd_dir = pcd_dir

    def read_bin_point_cloud(self, filename):
        points = np.fromfile(filename, dtype=np.float32).reshape(-1, 5)[:,:4]
        return points

    def load_and_display_point_cloud(self):
        pcd_files = sorted([f for f in os.listdir(self.pcd_dir) if f.endswith('.bin')])
        
        for pcd_file in pcd_files:
            pcd_path = os.path.join(self.pcd_dir, pcd_file)
            points = self.read_bin_point_cloud(pcd_path)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points[:, :3])
            axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0]) # add axis

            o3d.visualization.draw_geometries([pcd, axis])

    def load_and_display_images(self):
        image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        for image_file in image_files:
            image_path = os.path.join(self.image_dir, image_file)
            image = cv2.imread(image_path)
            
            cv2.imshow('Image', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
