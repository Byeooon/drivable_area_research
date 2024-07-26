import cv2
import os
import sys
from tqdm import tqdm

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
    # else:
    #     raise Exception('Folder already exists')

def main():
    makedirs(save_path)

    vidcap = cv2.VideoCapture(os.path.join(base_path, f'{video_name}.mp4'))
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    success, image = vidcap.read()
    count = 0
    frame_count = 0

    with tqdm(total=total_frames, desc="Processing frames") as pbar:
        while success:
            if frame_count % (video_fps/10) == 0:  # 3프레임마다 한 번씩 이미지 저장 (30fps -> 10fps)
                cv2.imwrite(os.path.join(save_path, "%06d.png") % count, image)
                count += 1
                print('Saved frame:', count)
            
            success, image = vidcap.read()
            frame_count += 1
            pbar.update(1)  # 진행 바 업데이트
        print("Finished! Converted video to frames at 10fps")

if __name__ == '__main__':
    base_path = '/home/julio981007/HDD/gurka/0'
    video_name = 'The last video from here Caterpillar 972M XE wheel loaders'
    save_path = os.path.join(base_path, 'image_data')
    video_fps = 60
    
    main()