'''
根据hevc比较yuv
'''
import os
from multiprocessing import Pool
import subprocess
from def_parser import *

'''
假设crop_yuv已经存在
'''
def run_comm(comm):
    subprocess.run(comm, shell=True)

parser = basic_parser()
args  = parser.parse_args()
yuv_list = os.listdir(args.croped_yuv_folder)
videos = ['StarryPolar', 'F5Fighter','BTSRun']
# 先测试一个视频试试 AirShow
time_stamp = ['06to07', '07to08', '08to09', '09to10', '10to11']
yuv_list = []
for video in videos:
    for time in time_stamp:
        yuv_list.append(time + '_' + video + '.yuv')


qp = list(range(18, 43))

for yuv in yuv_list:
    psnr_yuv_path = os.path.join(args.PSNR_path, yuv[:-4])
    if not os.path.exists(psnr_yuv_path):
        os.mkdir(psnr_yuv_path)
    comm_list = []
    for w in W_size:
        for h in H_size:
            for tmp_qp in qp:
                    
                    # hvec 文件夹命名：
                    # 记录每个hevc的size
                    if not os.path.exists(os.path.join(psnr_yuv_path,  '%04d_%04dx%04d'%(tmp_qp, w,h))):
                        os.mkdir(os.path.join(psnr_yuv_path,  '%04d_%04dx%04d'%(tmp_qp, w,h)))
                    
                    yuv_path = os.path.join(args.croped_yuv_folder, yuv[:-4], '%04dx%04d'%(w,h))
                    hevc_path = os.path.join(args.encode_folder, yuv[:-4], '%04d_%04dx%04d'%(tmp_qp, w,h))
                    hevcs = os.listdir(hevc_path)
                    hevcs.sort()
                    for hevc in hevcs:
                        comm = ("ffmpeg -r 30 -i %s -s %dx%d "
                        "-r 30 -pix_fmt yuv420p -i %s -lavfi psnr='%s' -f null -"%(os.path.join(hevc_path, hevc), w, h, os.path.join(yuv_path, '%s.yuv'%(hevc[:-5])), os.path.join(psnr_yuv_path,  '%04d_%04dx%04d'%(tmp_qp, w,h), hevc[:-5]+'.log'))
                        )
                        comm_list.append(comm)
    # 开多线程编码
    with Pool(args.cpu_num) as p:
        p.map(run_comm, comm_list)
    print("{} done!".format(yuv))
    subprocess.run('rm -r %s'%(os.path.join(args.croped_yuv_folder,  yuv[:-4])),shell= True)
    print("{} deleted!".format(os.path.join(args.croped_yuv_folder,  yuv[:-4])))
