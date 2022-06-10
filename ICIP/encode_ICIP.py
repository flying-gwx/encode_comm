'''
运行ICIP的分割和编码命令
输入: 
pixel 形式的u，v
带宽分配：Mb，[0.5, 1, 2, 4, 5]

输出：
在该带宽下根据这种分块得到的VPSNR值
'''
# 对6到10秒进行相同
import numpy as np
import subprocess
from ICIP_parser import *
import sys
sys.path.append('../')
from PSNR import *

parser = ICIP_result_parser()
args = parser.parse_args()
u = np.array([])
v = np.array([])


tmp_u = np.zeros(np.size(u) + 2, dtype=np.int16)
tmp_v = np.zeros(np.size(v) + 2, dtype=np.int16)
tmp_u[-1] = args.WIDTH -1 
tmp_v[-1] = args.HEIGHT - 1
tmp_u[1:-1] = u
tmp_v[1:-1] = v
u = tmp_u
v = tmp_v
for time in time_stamp:
    # 每秒的视频都剪一下
    origin_yuv = os.path.join(args.input_yuv_path, time + '_'+ args.video+'.yuv',)
    for h in range( np.size(v) - 1 ):
        for w in range(np.size(u) -1):
            begin_h = v[h]
            begin_w = u[w]
            width = u[w+1] - u[w]
            height = v[h+1] - v[h]
            comm = ('ffmpeg -y -pixel_format yuv420p -f rawvideo -video_size 3840x1920 -i %s'
            ' -vf crop=w=%d:h=%d:x=%d:y=%d %s/%04d_%04d_%04d_%04d.yuv')%(origin_yuv, width, height, begin_w, begin_h)
            subprocess.run(comm, shell=True)

# 根据视点判断哪些tile在里面

    
