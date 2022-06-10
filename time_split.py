'''
本module 为时间分割模块
'''

import os 
import multiprocessing
import subprocess
import argparse
from def_parser import *
from multiprocessing import Pool

# 切割出MP4
time_list = [6,7,8,9,10]
parser = basic_parser()
args = parser.parse_args()
result_folder = '/data/wenxuan/croped_mp4s'
yuvs_folder =  '/data/wenxuan/yuvs'
# 得到 yuv文件
for video in ALL_VIDEOS:
    crop_video_folder = os.path.join(result_folder, video)
    if not os.path.exists(crop_video_folder):
        os.mkdir(crop_video_folder)
    yuv_folder = os.path.join(yuvs_folder, video)
    if not os.path.exists(yuv_folder):
        os.mkdir(yuv_folder)
    for time in time_list:
        comm = 'ffmpeg -i %s.mp4 -y -ss 00:00:%02d -t 1 %s'%(os.path.join(args.mp4_30fps_folder, video), time,os.path.join(crop_video_folder, '%02dto%02d_%s.mp4'%(time, time+1, video)))
        comm1 = 'ffmpeg -i %s %s'%(os.path.join(crop_video_folder, '%02dto%02d_%s.mp4'%(time, time+1, video)), os.path.join(yuv_folder, '%02dto%02d_%s.yuv'%(time, time+1, video)) )
        subprocess.run(comm, shell=True)
        subprocess.run(comm1, shell=True)