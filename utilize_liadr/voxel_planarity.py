import open3d as o3d
import numpy as np

class VoxelPlanarityCalculator:
    def __init__(self, voxel_grid, point_cloud, planarity_threshold):
        self.voxel_grid = voxel_grid
        self.point_cloud = np.asarray(point_cloud.points) if not isinstance(point_cloud, np.ndarray) else point_cloud
        self.planarity_threshold = planarity_threshold
        self.planarity_dict = {}

    def calculate_planarity(self):
        voxel_dict = {}

        # 포인트를 복셀에 할당
        for point in self.point_cloud:
            voxel_index = tuple(self.voxel_grid.get_voxel(point))
            voxel_dict.setdefault(voxel_index, []).append(point)

        # 각 복셀에 대해 평탄도 계산
        for voxel_index, voxel_points in voxel_dict.items():
            voxel_points = np.array(voxel_points)
            
            if len(voxel_points) > 10:
                points_mean = np.mean(voxel_points, axis=0)
                centered_points = voxel_points - points_mean

                cov_matrix = np.cov(centered_points, rowvar=False)
                eigenvalues, _ = np.linalg.eigh(cov_matrix)
                eigenvalues = np.sort(eigenvalues)[::-1]

                planarity = min(((eigenvalues[1] - eigenvalues[2]) * 2) / eigenvalues[0], 1.0) if eigenvalues[0] > 0 else 0
            else:
                planarity = 0

            self.planarity_dict[voxel_index] = planarity

        return self.planarity_dict

    def visualize_planarity(self):
        colors = []

        for point in self.point_cloud:
            voxel_index = tuple(self.voxel_grid.get_voxel(point))
            planarity = self.planarity_dict.get(voxel_index, 0)
            
            # 평탄도가 임계값 이상인 복셀만 빨간색으로 표시, 나머지는 검은색
            color = [1, 0, 0] if planarity >= self.planarity_threshold else [0, 0, 0]
            colors.append(color)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.point_cloud)
        pcd.colors = o3d.utility.Vector3dVector(np.array(colors))

        o3d.visualization.draw_geometries([pcd])