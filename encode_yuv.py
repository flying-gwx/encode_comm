import os 
import multiprocessing
import subprocess
import argparse
from def_parser import basic_parser
from pathos.multiprocessing import ProcessingPool as Pool
'''
把crop好的yuv编码为hevc文件
'''
parser = basic_parser()
args = parser.parse_args()
def run_comm(comm):
    subprocess.run(comm, shell=True, stdout=open('/dev/null', 'w'), stderr=subprocess.STDOUT, check=True)

qp = list(range(18, 43))
# 读取特定yuv,进行编码，文件命名:time_name_qp_width_height


W_n = [30,15,12,10,8,6,5,4,3,2]
H_n = [30,15,12,10,8,6,5,4,3,2]
W_size =[]
H_size = []
for i in range(len(W_n)):
    W_size.append(int(args.WIDTH / W_n[i]))
for j in range(len(H_n)):
    H_size.append(int(args.HEIGHT / H_n[j]))

# 将crop后的文件删除

videos =  [ 'Symphony']
# # 先测试一个视频试试 AirShow
not_done = []
time_stamp = ['06to07','07to08', '08to09', '09to10', '10to11']
for video in videos:
    for time in time_stamp:
        not_done.append(time + '_' + video + '.yuv')
for yuv in not_done:
    comm_list = []
    for w in W_size:
        for h in H_size:
            for tmp_qp in qp:
                    # hvec 文件夹命名：
                    # 记录每个hevc的size
                    #TODO: 记录每个hevc的PSNR值
                    if not os.path.exists(os.path.join(args.encode_folder,  yuv[:-4],'%04d_%04dx%04d'%(tmp_qp,w,h))):
                        os.mkdir(os.path.join(args.encode_folder,  yuv[:-4],'%04d_%04dx%04d'%(tmp_qp,w,h)))
                    tile_names = os.listdir(os.path.join(args.croped_yuv_folder,  yuv[:-4], '%04dx%04d'%(w,h)))
                    tile_names.sort()
                    assert len(tile_names) !=0, "{} not croped!".format(yuv)
                    for tile in tile_names:
                        '''
                        我把nopsnr选项去掉了
                        '''
                        comm = ('kvazaar  -i %s  --input-res %dx%d --frames 30  --no-psnr '  
                        ' --qp %d --input-fps 30 -o %s'%(os.path.join(os.path.join(args.croped_yuv_folder,  yuv[:-4], '%04dx%04d'%(w,h), tile)), w,h, tmp_qp, os.path.join(args.encode_folder,  yuv[:-4],'%04d_%04dx%04d'%(tmp_qp,w,h), tile[:-4]+'.hevc')))
                        comm_list.append(comm)
    # 开多线程编码
    with Pool(args.cpu_num) as p:
        p.map(run_comm, comm_list)
    print("{} done!".format(yuv))
    

