import os 
import subprocess
import argparse
import numpy as np
from multiprocessing import Pool
from def_parser import *

parser = basic_parser()
args = parser.parse_args()


def run_comm(comm):
    subprocess.run(comm, shell=True, stdout=open('/dev/null', 'w'), stderr=subprocess.STDOUT, check=True)
def get_left_top(WIDTH, HEIGHT, tile_width, tile_height):
    '''
    x对应w
    y对应height
    '''
    x_list = []
    y_list = []
    for x in range(0,WIDTH, tile_width):
        for y in range(0, HEIGHT, tile_height):
            x_list.append(x)
            y_list.append(y)
    return x_list, y_list
#TODO: data is not right
qp = list(range(18, 43))


# ,'F5Fighter','BTSRun'
ALL_VIDEOS =  [ 'Symphony']
time_stamp = ['09to10', '10to11']
croped_yuvs = []
for video in ALL_VIDEOS:
    for time in time_stamp:
        croped_yuvs.append(time + '_' + video + '.yuv')
for yuv in croped_yuvs:
    t, yuv_name = yuv.split('_')
    # 找到对应yuv
    yuv_path = os.path.join(args.input_yuv_folder, yuv_name[:-4], yuv)
    if not os.path.exists(os.path.join(args.croped_yuv_folder,  yuv[:-4])):
        os.mkdir(os.path.join(args.croped_yuv_folder,  yuv[:-4]))
    if not os.path.exists(os.path.join(args.encode_folder,  yuv[:-4])):
        os.mkdir(os.path.join(args.encode_folder,  yuv[:-4]))
    comm_list = []
    for w in W_size:
        for h in H_size:
            #进行crop操作
            if not os.path.exists(os.path.join(args.croped_yuv_folder,  yuv[:-4], '%04dx%04d'%(w,h))):
                os.mkdir(os.path.join(args.croped_yuv_folder,  yuv[:-4], '%04dx%04d'%(w,h)))
            x_list, y_list = get_left_top(args.WIDTH, args.HEIGHT, w, h)
            for i in range(len(x_list)):
                comm = ('%s -y -pixel_format yuv420p -f rawvideo -video_size 3840x1920 -i %s'
                ' -vf crop=w=%d:h=%d:x=%d:y=%d %s/%04d_%04d_%04d_%04d.yuv'%(args.ffmpeg_path, yuv_path, w, h,x_list[i], y_list[i],os.path.join(args.croped_yuv_folder,  yuv[:-4], '%04dx%04d'%(w,h)), x_list[i], y_list[i],w, h))
                comm_list.append(comm)
    with Pool(args.cpu_num) as p:
        p.map(run_comm, comm_list)
    print("{} done!".format(yuv))
                   

                