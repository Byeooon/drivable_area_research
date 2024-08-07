import numpy as np
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
import os
import pandas as pd
import sys

# def visualize_heightmap(heightmap_data):
#     # 데이터를 2D 배열로 변환
#     x_min, y_min = heightmap_data.index.min()
#     x_max, y_max = heightmap_data.index.max()
    
#     grid_shape = (x_max - x_min + 1, y_max - y_min + 1)
#     grid = np.full(grid_shape, np.nan)
    
#     for (x, y), z in heightmap_data.items():
#         grid[x - x_min, y - y_min] = z
    
#     # 높이맵 시각화
#     plt.figure(figsize=(12, 10))
#     im = plt.imshow(grid, cmap='viridis', interpolation='nearest')
#     plt.colorbar(im, label='Height (z)')
#     plt.title('Heightmap Visualization')
#     plt.xlabel('Y')
#     plt.ylabel('X')
#     plt.gca().invert_yaxis()  # y축을 뒤집어 원점을 좌상단으로
#     plt.show()
    
def visualize_heightmap(heightmap_data):
    # 산점도로 시각화
    plt.figure(figsize=(12, 10))
    x_coords, y_coords = zip(*heightmap_data.index)
    plt.scatter(y_coords, x_coords, c=heightmap_data.values, cmap='viridis')
    plt.colorbar(label='Height (z)')
    plt.title('Heightmap Visualization')
    plt.xlabel('Y')
    plt.ylabel('X')
    plt.gca().invert_yaxis()  # y축을 뒤집어 원점을 좌상단으로
    plt.show()

def load_depth_map(file_path):
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

def world_to_image(points_3d, K, R, T):
    """
    3D 월드 좌표를 2D 이미지 좌표로 변환합니다.
    """
    # points_3d_homogeneous = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))
    points_cam = R @ points_3d.T + T[:, np.newaxis]
    points_2d_homogeneous = K @ points_cam
    points_2d = points_2d_homogeneous[:2] / points_2d_homogeneous[2]
    return points_2d.T

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
    
def visualize_height_on_image(image, points_2d, height_values, original_shape):
    """
    높이 값을 이미지 위에 시각화합니다.
    """
    # 이미지 크기에 맞게 좌표 조정
    points_2d[:, 0] = points_2d[:, 0] * image.shape[1] / original_shape[1]
    points_2d[:, 1] = points_2d[:, 1] * image.shape[0] / original_shape[0]

    # 정수 좌표로 변환
    points_2d = points_2d.astype(int)

    # 유효한 좌표만 선택
    valid_points = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < image.shape[1]) & \
                   (points_2d[:, 1] >= 0) & (points_2d[:, 1] < image.shape[0])
    points_2d = points_2d[valid_points]
    height_values = height_values[valid_points]

    # 높이 값을 컬러로 매핑
    normalized_heights = (height_values - height_values.min()) / (height_values.max() - height_values.min())
    colors = plt.cm.viridis(normalized_heights)[:, :3]

    # 이미지에 점 그리기
    for (x, y), color in zip(points_2d, colors):
        cv2.circle(image, (x, y), 1, color * 255, -1)

    # 결과 시각화
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax.set_title('Height Values Projected onto Image')
    ax.axis('off')
    
    # 컬러바 추가
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=height_values.min(), vmax=height_values.max()))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Height')
    
    plt.show()

def main():
    base_path = '/home/julio981007/HDD/orfd/training'
    img_name = '1623171106467'
    
    # 원본 이미지 로드
    img_path = os.path.join(base_path, 'image_data', img_name + '.png')
    original_image = cv2.imread(img_path)
    original_shape = original_image.shape[:2]
    
    depth_map_path = os.path.join(base_path, 'dense_depth', img_name + '.png')
    depth_map = load_depth_map(depth_map_path)
    
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
    heightmap_resolution = 1 # 0.02
    # generate some random 3D points
    points_df = pd.DataFrame(point_cloud, columns = ['x','y','z'])
    #didn't know if you wanted to keep the x and y columns so I made new ones.
    points_df['x_normalized'] = (points_df['x']/heightmap_resolution).astype(float)
    points_df['y_normalized'] = (points_df['y']/heightmap_resolution).astype(float)
    tmp = points_df.groupby(['x_normalized','y_normalized'])['z'].max()
    visualize_heightmap(tmp)
    points_df['height'] = tmp.values
    indices = np.array(tmp.index.tolist())
    #################################################################################
    # 3D 포인트와 높이 값 추출
    points_3d = points_df[['x', 'y', 'z']].values
    height_values = points_df['height'].values

    print(R)
    # 3D 포인트를 2D 이미지 좌표로 변환
    points_2d = world_to_image(points_3d, K, R, T)
    
    # 높이 값을 이미지에 투영하여 시각화
    visualize_height_on_image(original_image, points_2d, height_values, original_shape)

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