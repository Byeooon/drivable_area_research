from pyexpat import ExpatError
from pytube import YouTube
import sys

def on_complete(stream, file_path):
    print(stream)
    print(file_path)


def on_progress(stream, chunk, bytes_remaining):
    print(100 - (bytes_remaining / stream.filesize * 100))

# 저장할 경로 설정
SAVE_PATH = "/media/imlab/HDD/gurka/" # 경로 설정 

# 다운로드할 유튜브 링크 리스트에 담기
link= open('/home/imlab/drivable_area_research/data/youtube.txt', 'r')

# for문 돌려서 다운로드받기
for i in link: 
    try:    
        # Youtube객채 생성
        yt = YouTube(url=i, on_complete_callback=on_complete, on_progress_callback=on_progress)
        yt_stream = yt.streams
        
    except: 
        # 예외처리
        print("Connection Error") 
    
    # 모든 파일 mp4 저장으로 설정
    mp4files = yt_stream.filter(file_extension='mp4')
    if len(mp4files)==0:
        raise Exception('Not found video')
    
    my_stream = mp4files.order_by('resolution').desc().first()
    
    try: 
        # 비디오 다운받기
        my_stream.download(SAVE_PATH)
    except: 
        print("에러 발생!") 
print('다운 완료!')