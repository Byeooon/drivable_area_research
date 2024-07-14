import cv2
import numpy as np
import matplotlib.pylab as plt
import os
from tqdm import tqdm
import sys
from natsort import natsorted

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
    # else:
    #     raise Exception('Already folder exists')

base_path = '/media/imlab/HDD/gurka'
folders = ['0']

video_name = "video.mp4"
alpha = 0.6
fps = 10

for folder in folders:
    img_path = os.path.join(base_path, f'{folder}/image')
    label_path = os.path.join(base_path, f'{folder}/pseudo_labeling')

    save_path = os.path.join(base_path, f'{folder}/png_to_video')
    makedirs(save_path)

    img_list = natsorted([file for file in os.listdir(img_path) if file.endswith('.png')])
    
    frame_list = []
    for i in tqdm(img_list):
        img_name = i
        image = cv2.imread(os.path.join(img_path, f'{img_name}'))
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        label = cv2.imread(os.path.join(label_path, f'{img_name}'), cv2.IMREAD_GRAYSCALE)
        label_rgb = np.zeros_like(image)
        label_rgb[:,:,2] = label  # OpenCV에서는 BGR 순서이므로 인덱스 2가 R 채널입니다.

        dst = cv2.addWeighted(image, alpha, label_rgb, (1-alpha), 0)
        frame_list.append(dst)
        
        # cv2.imwrite('tmp.png', dst)

        # plt.imshow(dst)
        # plt.savefig('tmp.png')
        # sys.exit()
    h, w, c = frame_list[0].shape
    # out = cv2.VideoWriter(filename=os.path.join(save_path, video_name), fourcc=cv2.VideoWriter_fourcc(*'FMP4'), fps=fps, frameSize=size)
    out = cv2.VideoWriter(filename=os.path.join(save_path, video_name), fourcc=cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps=fps, frameSize=(w, h))
    
    for img in frame_list:
        out.write(img)
    out.release()