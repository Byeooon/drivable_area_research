import cv2
import os

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        raise Exception('Folder already exists')

base_path = '/media/imlab/HDD/gurka/0'

video_name = 'The last video from here Caterpillar 972M XE wheel loaders'

save_path = os.path.join(base_path, 'image')

makedirs(save_path)

vidcap = cv2.VideoCapture(os.path.join(base_path, f'{video_name}.mp4'))
success, image = vidcap.read()
count = 0
frame_count = 0

while success:
    if frame_count % 3 == 0:  # 3프레임마다 한 번씩 이미지 저장 (30fps -> 10fps)
        cv2.imwrite(os.path.join(save_path, "%06d.png") % count, image)
        count += 1
        print('Saved frame:', count)
    
    success, image = vidcap.read()
    frame_count += 1
print("Finished! Converted video to frames at 10fps")

'''
while success:
    cv2.imwrite(os.path.join(save_path, "%06d.png") % count, image)
    success,image = vidcap.read()
    print('Read a new frame: ', success)
    count += 1

print("finish! convert video to frame")
'''