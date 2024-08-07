from __future__ import absolute_import, division, print_function

import os
import copy
import glob
import numpy as np
from collections import Counter
from scipy import interpolate
import skimage.transform
import cv2
if not ("DISPLAY" in os.environ):
    import matplotlib as mpl
    mpl.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import mayavi.mlab
import pandas as pd
import open3d as o3d
from tqdm import tqdm

import multiprocessing as mp
import sys

cmap = plt.cm.jet
cmap2 = plt.cm.nipy_spectral


def show_open3d_pcd(pcd, show_origin=True, origin_size=3, show_grid=True):
    cloud = o3d.geometry.PointCloud()
    v3d = o3d.utility.Vector3dVector
    
    if isinstance(pcd, type(cloud)):
        pass
    elif isinstance(pcd, np.ndarray):
        cloud.points = v3d(pcd)
        
    coord = o3d.geometry.TriangleMesh().create_coordinate_frame(size=origin_size, origin=np.array([0.0, 0.0, 0.0]))
    
    # set front, lookat, up, zoom to change initial view
    o3d.visualization.draw_geometries([cloud, coord])

def show_velo(pointcloud):
    x = pointcloud[:, 0]  # x position of point
    y = pointcloud[:, 1]  # y position of point
    z = pointcloud[:, 2]  # z position of point
    
    d = np.sqrt(x ** 2 + y ** 2)  # Map Distance from sensor
    
    degr = np.degrees(np.arctan(z / d))
    
    vals = 'height'
    if vals == "height":
        col = z
    else:
        col = d
    
    fig = mayavi.mlab.figure(bgcolor=(0, 0, 0), size=(640, 500))
    mayavi.mlab.points3d(x, y, z,
                        col,  # Values used for Color
                        mode="point",
                        colormap='spectral',  # 'bone', 'copper', 'gnuplot'
                        # color=(0, 1, 0),   # Used a fixed (r,g,b) instead
                        figure=fig,
                        )
    
    mayavi.mlab.show()

def depth_colorize(depth):
    depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
    depth = 255 * cmap(depth)[:, :, :3]  # H, W, C
    return depth.astype('uint8')

def load_velodyne_points(filename):
    """Load 3D point cloud from KITTI file format
    (adapted from https://github.com/hunse/kitti)
    """
    #points = np.fromfile(filename, dtype=np.float32).reshape(-1, 4) #kitti
    points = np.fromfile(filename, dtype=np.float32).reshape(-1, 5)[:,:4]  #jk hesai40 data
    points[:, 3] = 1.0  # homogeneous
    return points


def read_calib_file(path):
    """Read KITTI calibration file
    (from https://github.com/hunse/kitti)
    """
    float_chars = set("0123456789.e+- ")
    data = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                # try to cast to float array
                try:
                    data[key] = np.array(list(map(float, value.split(' '))))
                except ValueError:
                    # casting error: data[key] already eq. value, so pass
                    pass

    return data


def sub2ind(matrixSize, rowSub, colSub):
    """Convert row, col matrix subscripts to linear indices
    """
    m, n = matrixSize
    return rowSub * (n-1) + colSub - 1

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

def generate_depth_map(velo_filename, depth_filename, calib_filename):
    """Generate a depth map from jk hesai40 data
    """
    calib_filename = '/home/julio981007/HDD/orfd/training/calib/1619778700681.txt'
    
    K, R, T, _, _ = read_calibration(calib_filename)
    
    depth_map = load_depth_map(depth_filename)
    point_cloud = depth_to_pointcloud(depth_map, K, R, T)

    # set image shape
    im_shape = np.array([720,1280],dtype=np.int32)

    # load velodyne points
    # velo = load_velodyne_points(velo_filename)
    # show_velo(velo)
    
    velo = point_cloud
    
    # camera parameter
    K = np.array([ 1472.919866, 0.000000, 614.779599, 0.000000, 1452.953534, 353.800982, 0.000000, 0.000000, 1.000000 ]).reshape(3,3)
    
    RT = np.array([ 9.9954895655531806e-01, -2.9636912774399400e-02,
       4.8514791948404291e-03, 5.4418414831161499e-02,
       2.6634465658627177e-03, -7.3426033150586920e-02,
       -9.9729710904432078e-01, -1.2367740273475647e-01,
       2.9913032303096956e-02, 9.9686020637648665e-01,
       -7.3313978486113582e-02, -1.0199587792158127e-01, 0., 0., 0., 1. ]).reshape(4,4)
    
    '''
    # the default RT of camera is camera to car, so inv is needed 
    print('R before:',RT)
    RTC = np.linalg.inv(RT)
    print('R after:',RTC)
    '''
    
    # velodyne parameter from velodyne to car
    R_velo = np.array([-0.996842,-0.0793231,0.00385075,
        0.0794014,-0.994533,0.067813,
        -0.00154944,0.0679046,0.997691]).reshape(3,3)
    '''
    base_path = '/home/julio981007/HDD/gurka/0'
    img_name = '000237'
    depth_map_path = os.path.join(base_path, 'dense_depth', img_name + '.png')
    depth_map = load_depth_map(depth_map_path)
    '''
    R_velo = np.array([
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.]
            ]).reshape(3,3)
    
    T_velo = np.array([0, 0, 0]).reshape(3,1)
    
    # project velodyne to car coordinate
    velo = R_velo @ velo[:,:3].T + T_velo
    velo = velo.T

    # remove all behind image plane (approximation)
    velo = velo[velo[:, 1] >= 0, :]
    #show_velo(velo)

    '''
    One_array = np.ones(velo.shape[0]).reshape(velo.shape[0],1)
    velo = np.hstack((velo,One_array))
    print('velo.shape{}'.format(velo.shape))
    '''

    # project the points to the camera
    # projection from car to image
    '''
    # version 1:
    P_car2im = K @ RT[:3, :4]
    velo_pts_im = np.dot(P_car2im, velo.T).T
    '''

    # version 2:
    R = RT[:3,:3]
    T = RT[:3,-1].reshape(3,1)

    velo_pts_cam = (R @ velo.T + T).T
    #show_velo(velo_pts_cam)

    velo_pts_im = (K @ velo_pts_cam.T).T
    #show_velo(velo_pts_im)
    #velo_pts_im = np.array((velo_pts_im[:,0],velo_pts_im[:,1],velo_pts_im[:,2])).T#*(-1)

    velo_pts_im[:, :2] = velo_pts_im[:, :2] / velo_pts_im[:, 2][..., np.newaxis]
    
    #################################################################################
    heightmap_resolution = 1 # 0.02
    # generate some random 3D points
    points_df = pd.DataFrame(velo, columns = ['x','y','z'])
    #didn't know if you wanted to keep the x and y columns so I made new ones.
    points_df['x_normalized'] = (points_df['x']/heightmap_resolution).astype(float)
    points_df['y_normalized'] = (points_df['y']/heightmap_resolution).astype(float)
    tmp = points_df.groupby(['x_normalized','y_normalized'])['z'].max()
    
    points_df['height'] = tmp.values
    
    # 3D 포인트와 높이 값 추출
    points_3d = points_df[['x', 'y', 'z']].values
    height_values = points_df['z'].values
    
    velo_pts_im = np.c_[velo_pts_im, height_values]
    #################################################################################

    # check if in bounds
    # use minus 1 to get the exact same value as KITTI matlab code
    velo_pts_im[:, 0] = np.round(velo_pts_im[:, 0]) - 1
    velo_pts_im[:, 1] = np.round(velo_pts_im[:, 1]) - 1
    #velo_pts_im[:, 0] 
    val_inds = (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 1] >= 0)
    val_inds = val_inds & (velo_pts_im[:, 0] < im_shape[1]) & (velo_pts_im[:, 1] < im_shape[0])
    velo_pts_im = velo_pts_im[val_inds, :]
    
    # project to image
    depth = np.zeros((im_shape[:2]))
    height = np.zeros((im_shape[:2]))
    depth[velo_pts_im[:, 1].astype(np.int32), velo_pts_im[:, 0].astype(np.int32)] =  velo_pts_im[:, 2]#*(-1)
    height[velo_pts_im[:, 1].astype(np.int32), velo_pts_im[:, 0].astype(np.int32)] =  velo_pts_im[:, 3]#*(-1)

    # find the duplicate points and choose the closest depth
    inds = sub2ind(depth.shape, velo_pts_im[:, 1], velo_pts_im[:, 0])
    dupe_inds = [item for item, count in Counter(inds).items() if count > 1]
    for dd in dupe_inds:
        pts = np.where(inds == dd)[0]
        x_loc = int(velo_pts_im[pts[0], 0])
        y_loc = int(velo_pts_im[pts[0], 1])
        depth[y_loc, x_loc] = velo_pts_im[pts, 2].min()
        height[y_loc, x_loc] = velo_pts_im[pts, 3].min()
    depth[depth < 0] = 0
    # print('depth.max:{}, depth.min:{}:'.format(depth.max(),depth.min()))

    # interpolate the depth map to fill in holes
    depth_inter = lin_interp(im_shape, velo_pts_im)
    #depth_inter = lstsq_interp(im_shape, velo_pts_im,valid=False)
    
    depth_inter = height

    '''
    # transfor depth to pointclouds
    depth_nonzero = np.where(depth_inter>0)
    #depth_nonzero = np.where(depth>0)
    u = depth_nonzero[1]
    v = depth_nonzero[0]
    z = depth_inter[v,u]
    
    #print('depth_nonzero:',depth_nonzero)
    #print('u:',u)
    #print('v:',v)
    #print('z:',z)

    #uvz = np.vstack((u,v,z))
    uvz = np.vstack((u,v,z))
    
    velo_c = np.linalg.inv(K) @ uvz
    print('velo_c:',velo_c,velo_c.shape)

    velo_w = R.T @ (velo_c - T)
    print('velo_w:',velo_w,velo_w.shape)

    show_velo(velo_w.T)
    '''
    return depth,depth_inter

def lin_interp(shape, xyd):
    from scipy.interpolate import LinearNDInterpolator

    m, n = shape
    ij, d = xyd[:, 1::-1], xyd[:, 2]
    f = LinearNDInterpolator(ij, d, fill_value=0)
    J, I = np.meshgrid(np.arange(n), np.arange(m))
    IJ = np.vstack([I.flatten(), J.flatten()]).T
    disparity = f(IJ).reshape(shape)
    return disparity


def main(base_path):
    """
    process hesai40 lidar to sparse depth image and dense image
    """
    path1_list = glob.glob(base_path+'/*')
    for path2 in tqdm(path1_list):
        lidar_path = os.path.join(path2,'lidar_data')
        rgb_path = os.path.join(path2,'image_data')

        sparse_depth_dir = os.path.join(path2,'sparse_depth')
        dense_depth_dir = os.path.join(path2,'dense_depth')
        height_dir = os.path.join(path2,'height')
        calib_dir = os.path.join(path2,'calib')
        # if not os.path.exists(sparse_depth_dir):
        #     os.mkdir(sparse_depth_dir)
        # if not os.path.exists(dense_depth_dir):
        #     os.mkdir(dense_depth_dir)
        if not os.path.exists(height_dir):
            os.mkdir(height_dir)
        else:
            raise Exception('The directory {} already exists.'.format(height_dir))

        # depth img name and dirs
        # lidar_data_list = glob.glob(lidar_path+'/*')
        rgb_data_list = glob.glob(rgb_path+'/*')
        for rgb_filename in tqdm(rgb_data_list):
            # print('velo_filename:',velo_filename)
            img_name = rgb_filename.split('/')[-1].split('.')[0]
            lidar_dir = os.path.join(lidar_path,img_name+'.bin')
            sparse_depth_save_dir = os.path.join(sparse_depth_dir,img_name+'.png')
            dense_depth_save_dir = os.path.join(dense_depth_dir,img_name+'.png')
            rgb_img_dir = os.path.join(rgb_path,img_name+'.png')
            
            calib_save_dir = os.path.join(calib_dir,img_name+'.txt')
            height_save_dir = os.path.join(height_dir,img_name+'.tiff')
        

            # TODO: prepare for multi process 
            # lidar to depth
            sparse_depth_pred, dense_depth_pred = generate_depth_map(lidar_dir, dense_depth_save_dir, calib_save_dir)

            sparse_depth = copy.deepcopy(sparse_depth_pred)
            dense_depth = copy.deepcopy(dense_depth_pred)
            
            # save depth in unit16 format
            img = (sparse_depth * 256.0).astype('uint16')
            img_buffer = img.tobytes()
            imgsave = Image.new("I", img.T.shape)
            imgsave.frombytes(img_buffer, 'raw', "I;16")
            # imgsave.save(sparse_depth_save_dir)

            img = (dense_depth * 256.0).astype('uint16')
            img_buffer = img.tobytes()
            imgsave = Image.new("I", img.T.shape)
            imgsave.frombytes(img_buffer, 'raw', "I;16")
            # imgsave.save(dense_depth_save_dir)
            
            cv2.imwrite(height_save_dir, dense_depth)
            
            '''
            # show depth 
            sparse_depth = depth_colorize(sparse_depth)
            sparse_depth = cv2.cvtColor(sparse_depth, cv2.COLOR_RGB2BGR)

            dense_depth = depth_colorize(dense_depth)
            dense_depth = cv2.cvtColor(dense_depth, cv2.COLOR_RGB2BGR)
            
            rgb = cv2.imread(rgb_img_dir) # the correspoing rgb img

            depth = np.concatenate((rgb,dense_depth,sparse_depth),axis=1)
            depth = cv2.resize(depth, (0,0),fx=0.5, fy=0.5)

            
            #cv2.imshow('depth_gt',depth_new)
            cv2.imshow('depth',depth)
            
            # show image with sparse to check the project consistency
            rgb = np.transpose(rgb, (2, 0, 1))
            ind = sparse_depth_pred>0
            rgb[0][ind] = 0 #sparse_depth[ind]
            rgb[1][ind] = 0 #sparse_depth[ind]
            rgb[2][ind] = 0  #sparse_depth[ind]
            rgb = rgb.transpose(1,2,0)

            cv2.imshow('rgb+sparse depth',rgb)
            cv2.waitKey(1)
            '''


if __name__=='__main__':
    base_path = '/home/julio981007/HDD/gurka' 
    main(base_path)