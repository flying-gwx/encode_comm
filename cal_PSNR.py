'''
编码计算视口中tile的PSNR值
后续再根据视口中的PSNR值得到vpsnr值
'''
from http import client
import os
from multiprocessing import Pool
import subprocess
from def_parser import *
import numpy as np

from lib.psnrlib import return_all_tile_union
'''
假设crop_yuv已经存在
'''
def run_comm(comm):
    subprocess.run(comm, shell=True, stdout=open('/dev/null', 'w'), stderr=subprocess.STDOUT, check=True)

parser = basic_parser()
args  = parser.parse_args()
# AirShow Done
videos =  [ 'Symphony']
# videos = ['F5Fighter']
Notencode_video = ['Surfing','Waterskiing', 'WaitingForLove', 'LOL']
time_stamp = ['06to07','07to08', '08to09','09to10', '10to11']
is_tile_in_viewpoint_folder = '/data/wenxuan/GCN_data/tile_in_viewpoint_npy'
qp = list(range(18, 43))
client_num = 40
yuv_list = []
for video in videos:
    for time in time_stamp:
        yuv_list.append(time + '_' + video + '.yuv')

for yuv in yuv_list:
    psnr_yuv_path = os.path.join(args.PSNR_path, yuv[:-4])
    if not os.path.exists(psnr_yuv_path):
        os.mkdir(psnr_yuv_path)
    for w in W_size:
        for h in H_size:
            # 读取在视口中的tile
            path = os.path.join(is_tile_in_viewpoint_folder, yuv[:-4], '%04dx%04d.npy'%(w, h))
            tiles = np.load(path, allow_pickle=True).item()
            # 对所有用户的tile求交集
            hevcs = []
            all_tile = return_all_tile_union(tiles)
            for encode_tile in all_tile:
                hevcs.append(encode_tile + '.hevc')
            comm_list = []
            for tmp_qp in qp:
                # 对tmp_file 进行编码求PSNR
                if not os.path.exists(os.path.join(psnr_yuv_path,  '%04d_%04dx%04d'%(tmp_qp, w,h))):
                    os.mkdir(os.path.join(psnr_yuv_path,  '%04d_%04dx%04d'%(tmp_qp, w,h)))
                yuv_path = os.path.join(args.croped_yuv_folder, yuv[:-4], '%04dx%04d'%(w,h))
                hevc_path = os.path.join(args.encode_folder, yuv[:-4], '%04d_%04dx%04d'%(tmp_qp, w,h))
                for hevc in hevcs:
                    if not os.path.exists(os.path.join(psnr_yuv_path,  '%04d_%04dx%04d'%(tmp_qp, w,h), hevc[:-5]+'.log')):
                        comm = ("ffmpeg -r 30 -i %s -f rawvideo -s %dx%d "
                        "-r 30 -pix_fmt yuv420p -i %s -lavfi psnr='%s' -f null -"%(os.path.join(hevc_path, hevc), w, h, os.path.join(yuv_path, '%s.yuv'%(hevc[:-5])), os.path.join(psnr_yuv_path,  '%04d_%04dx%04d'%(tmp_qp, w,h), hevc[:-5]+'.log'))
                        )
                        comm_list.append(comm)
            with Pool(args.cpu_num) as p:
                p.map(run_comm, comm_list)
    print("{} done!".format(yuv))
    subprocess.run('rm -r %s'%(os.path.join(args.croped_yuv_folder,  yuv[:-4])),shell= True)
    print("{} deleted!".format(os.path.join(args.croped_yuv_folder,  yuv[:-4])))

