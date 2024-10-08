from data_loader import DataLoader
from depth2voxel import Depth2Voxel
from voxel_planarity import VoxelPlanarityCalculator

import sys

if __name__ == "__main__":
    calibration_file = "data/calib/1623170280811.txt"
    depth_map_file = "data/dense_depth_map/1623170280811.png"
    
    converter = Depth2Voxel(calibration_file)
    depth_map = converter.load_depth_map(depth_map_file)
    points_3D = converter.depth_to_point_cloud(depth_map)
    voxel_grid = converter.convert_to_voxel_grid(points_3D, voxel_size=0.25)

    calculator = VoxelPlanarityCalculator(voxel_grid, points_3D, planarity_threshold=0.7)
    planarity_dict = calculator.calculate_planarity()

    calculator.visualize_planarity()