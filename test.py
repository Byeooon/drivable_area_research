import numpy as np
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
import os
import pandas as pd

def visualize_heightmap(heightmap_data):
    # 데이터를 2D 배열로 변환
    x_min, y_min = heightmap_data.index.min()
    x_max, y_max = heightmap_data.index.max()
    
    grid_shape = (x_max - x_min + 1, y_max - y_min + 1)
    grid = np.full(grid_shape, np.nan)
    
    for (x, y), z in heightmap_data.items():
        grid[x - x_min, y - y_min] = z
    
    # 높이맵 시각화
    plt.figure(figsize=(12, 10))
    im = plt.imshow(grid, cmap='viridis', interpolation='nearest')
    plt.colorbar(im, label='Height (z)')
    plt.title('Heightmap Visualization')
    plt.xlabel('Y')
    plt.ylabel('X')
    plt.gca().invert_yaxis()  # y축을 뒤집어 원점을 좌상단으로
    plt.show()

def load_depth_map(file_path, depth_scale=1000.0):
    depth_map = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    return depth_map.astype(np.float32)

def read_calibration(file_path):
    calib_data = {}
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('cam_K:'):
                calib_data['K'] = np.array(line.split(':')[1].split()).astype(float).reshape(3, 3)
            elif line.startswith('cam_RT:'):
                calib_data['RT'] = np.array(line.split(':')[1].split()).astype(float).reshape(4, 4)
            elif line.startswith('lidar_R:'):
                calib_data['lidar_R'] = np.array(line.split(':')[1].split()).astype(float).reshape(3, 3)
            elif line.startswith('lidar_T:'):
                calib_data['lidar_T'] = np.array(line.split(':')[1].split()).astype(float)
    R = calib_data['RT'][:3, :3]
    T = calib_data['RT'][:3, 3]
    return calib_data['K'], R, T, calib_data['lidar_R'], calib_data['lidar_T']

def depth_to_pointcloud(depth_map, K, R, T):
    height, width = depth_map.shape
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    x = x.flatten()
    y = y.flatten()
    z = depth_map.flatten()
    valid = z > 0
    x, y, z = x[valid], y[valid], z[valid]
    points_cam = np.vstack((x, y, np.ones_like(x)))
    points_cam = np.linalg.inv(K) @ points_cam
    points_cam *= z[np.newaxis, :]
    points_world = R.T @ (points_cam - T[:, np.newaxis])
    return points_world.T

def visualize_depth_map(depth_map):
    plt.imshow(depth_map, cmap='viridis')
    plt.colorbar(label='Depth')
    plt.title('Depth Map')
    plt.show()

def visualize_pointcloud(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd])

def visualize_top_view(points):
    plt.scatter(points[:, 0], points[:, 1], s=1)
    plt.title('Top View of Point Cloud')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.show()

def main():
    base_path = '/home/julio981007/HDD/orfd/training'
    img_name = '1623171106467'
    
    depth_map_path = os.path.join(base_path, 'dense_depth', img_name + '.png')
    depth_scale = 1000.0
    depth_map = load_depth_map(depth_map_path, depth_scale)
    
    # Visualize depth map
    visualize_depth_map(depth_map)
    
    calib_file = os.path.join(base_path, 'calib', img_name + '.txt')
    K, R, T, _, _ = read_calibration(calib_file)
    
    point_cloud = depth_to_pointcloud(depth_map, K, R, T)
    
    # # Visualize the 3D point cloud
    # visualize_pointcloud(point_cloud)
    
    # # Visualize top view (x-y projection)
    # visualize_top_view(point_cloud[:, :2])  # Only use x and y coordinates
    #################################################################################
    heightmap_resolution = 0.02
    # generate some random 3D points
    points_df = pd.DataFrame(point_cloud, columns = ['x','y','z'])
    #didn't know if you wanted to keep the x and y columns so I made new ones.
    points_df['x_normalized'] = (points_df['x']/heightmap_resolution).astype(int)
    points_df['y_normalized'] = (points_df['y']/heightmap_resolution).astype(int)
    tmp = points_df.groupby(['x_normalized','y_normalized'])['z'].max()
    visualize_heightmap(tmp)
    # print(tmp)

if __name__ == "__main__":
    main()

'''
import numpy as np
import pandas as pd

heightmap_resolution = 0.02

# generate some random 3D points
points =  np.array([[x,y,z] for x in np.random.uniform(0,2,100) for y in np.random.uniform(0,2,100) for z in np.random.uniform(0,2,100)])
points_df = pd.DataFrame(points, columns = ['x','y','z'])
#didn't know if you wanted to keep the x and y columns so I made new ones.
points_df['x_normalized'] = (points_df['x']/heightmap_resolution).astype(int)
points_df['y_normalized'] = (points_df['y']/heightmap_resolution).astype(int)
tmp = points_df.groupby(['x_normalized','y_normalized'])['z'].max()

print(tmp)
'''