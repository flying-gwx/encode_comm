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
mp4_folder = '/data/wenxuan/4k_30fps'
viewpoint_root = '/data/wenxuan/head-tracking-master'
save_path = '/data/wenxuan/ICIP'
mp4_path = os.path.join(save_path, 'encode')
psnr_path = os.path.join(save_path, 'PSNR')
device = torch.device(args.device)
if type(u) is not np.ndarray:
    u = np.array(u)
    v = np.array(v)
tmp_u = np.zeros(np.size(u) + 2, dtype=np.int16)
tmp_v = np.zeros(np.size(v) + 2, dtype=np.int16)
tmp_u[-1] = args.WIDTH
tmp_v[-1] = args.HEIGHT
tmp_u[1:-1] = u
tmp_v[1:-1] = v
u = tmp_u
v = tmp_v

# 编码这些tile
# 输入u，v viewpoint


for tmp_time in time_stamp:
    # 每秒的视频都剪一下
    origin_yuv = os.path.join(args.input_yuv_path, tmp_time + '_'+ args.video+'.yuv')
    # croped_yuv保存位置
    t_begin, t_end = tmp_time.split('to')
    t_begin = int(t_begin)
    t_end = int(t_end)
    total_viewpoint = Get_Viewpoint(viewpoint_root, os.path.join(mp4_folder, args.video + '.mp4'), args.video)
    viewpoint = total_viewpoint[:, 30*t_begin:30*t_end, :]
    clientNum = viewpoint.shape[0]
    frameNum = viewpoint.shape[1]
    '''
    每个用户的比特分配量不同
    '''
    user_vmse_list = []
    for i in range(clientNum):
        # 储存编码的begin_h, begin_w, width, height, 所占视口像素数
        view_tile = {}
        user_mse = 0
        all_mask = torch.zeros(args.HEIGHT, args.WIDTH, dtype=torch.float32).to(args.device)
        one = torch.ones_like(all_mask)
        for j in range(frameNum):        
            #得到GOP级别的mask
            mask = Get_Mask(args.HEIGHT, args.WIDTH, viewpoint[i, j, 1], viewpoint[i, j, 0]).to('cuda:0')
            all_mask = all_mask + mask
        all_mask = torch.where(all_mask > 0, one, all_mask)
        pixels = torch.sum(all_mask)
        for h in range( np.size(v) - 1 ):
            for w in range(np.size(u) -1):
                begin_h = v[h]
                begin_w = u[w]
                width = u[w+1] - u[w]
                height = v[h+1] - v[h]
                # 判断是否在视点内
                tmp_pixel = torch.sum(mask[begin_h:begin_h+height, begin_w:begin_w + width])
                if tmp_pixel > 0:
                    view_tile['%04d_%04d_%04d_%04d'%(begin_w, begin_h,width, height)] = tmp_pixel/pixels
        # 把key值求交集
        
        tiles_num = len(view_tile.keys())
        # 切割出要被看见的部分
        for key in view_tile.keys():
            all_data = key.split('_')
            begin_w = int(all_data[0])
            begin_h = int(all_data[1])
            width = int(all_data[2])
            height = int(all_data[3])
            comm = ('ffmpeg -y -pixel_format yuv420p -f rawvideo -video_size 3840x1920 -i %s'
            ' -vf crop=w=%d:h=%d:x=%d:y=%d %s')%(origin_yuv, width, height, begin_w, begin_h, os.path.join(save_path, 'croped_yuv', '%s.yuv'%(key)) )
            subprocess.run(comm, shell=True)
            # 分配比特率，编码
        mse_list = []
        for band in BANDWIDTH:
            bit_per_tile = int(band/tiles_num)
            for key in view_tile.keys():
                all_data = key.split('_')
                begin_w = int(all_data[0])
                begin_h = int(all_data[1])
                width = int(all_data[2])
                height = int(all_data[3])
                # 对每块tile使用相同比特率进行编码
                comm = ('kvazaar  -i %s  --input-res %dx%d --frames 30  --no-psnr --no-info '  
                        ' --bitrate %d --input-fps 30 -o %s'%(os.path.join(save_path, 'croped_yuv','%s.yuv'%(key)), w,h, bit_per_tile, os.path.join(mp4_path,'%s.hevc'%(key) )))
                
            # ffmpeg计算psnr
            for key in view_tile.keys():
                comm = ("ffmpeg -r 30 -i %s -s %dx%d "
                    "-r 30 -pix_fmt yuv420p -i %s -lavfi psnr='%s' -f null -"%(os.path.join(mp4_path,'%s.hevc'%(key) ), w, h, os.path.join(save_path, 'croped_yuv','%s.yuv'%(key)), os.path.join(psnr_path,  key+'.log'))
                    )
            #读取mse，计算vmse,得到该用户的vmse值
            mse = 0
            for key in view_tile.keys():
                mse += read_average_mse(os.path.join(psnr_path,  key+'.log'))*view_tile[key]
            # 清除log， yuv和mp4部分
            mse_list.append(mse)
            subprocess.run('rm %s/*'%(os.path.join(save_path, 'croped_yuv')), shell=True)
            subprocess.run('rm %s/*'%(mp4_path))
            subprocess.run('rm %s/*'%(psnr_path))
        user_vmse_list.append(mse_list)
        # 输出一个与bandwidth等长的numpy, 对所有user的mse取均值
        user_vmse_npy = np.stack(user_vmse_list, axis = 0)
        result_mse = np.array(len(BANDWIDTH))
        for band_id in range(len(BANDWIDTH)):
            result_mse[band_id] = np.mean(user_vmse_npy[:, band_id])
        print('{} result_mse is {}'.format(origin_yuv, result_mse))
        vpsnr = 10*np.log10(255**2/ result_mse)
        np.save('{}.npy'.format(tmp_time + '_'+ args.video), vpsnr)
