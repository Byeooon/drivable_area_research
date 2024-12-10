import glob
from PIL import Image

from ransac import apply_ransac
from offroad_ransac import apply_offroad_ransac

hdx_pcd_path = '/home/byeooon/Desktop/utilize_liadr/hdx_data/hdx_pcd/*.pcd'
pcd_files = glob.glob(hdx_pcd_path)

hdx_img_path = '/home/byeooon/Desktop/utilize_liadr/hdx_data/hdx_img/*.png'
img_files = glob.glob(hdx_img_path)

if __name__ == '__main__' :
    # for pcd in pcd_files:
        # print('now pcd is ... ', pcd)
        apply_ransac('/home/byeooon/Desktop/utilize_liadr/ransac/plane.pcd')
        apply_ransac('/home/byeooon/Desktop/utilize_liadr/ransac/sep_plane.pcd')
        # apply_ransac('/home/byeooon/Desktop/utilize_liadr/ransac/slope.pcd')

        # apply_offroad_ransac('/home/byeooon/Desktop/utilize_liadr/ransac/plane.pcd')
        # apply_offroad_ransac('/home/byeooon/Desktop/utilize_liadr/ransac/sep_plane.pcd')
        # apply_offroad_ransac('/home/byeooon/Desktop/utilize_liadr/ransac/slope.pcd')