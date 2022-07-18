'''
读取 is_tile_in_folder
读取 PSNR
计算 vpsnr
以 字典形式保存
'''
from PSNR import *
import os
import numpy as np
from def_parser import *
import time
from lib.psnrlib import *
from lib.sizelib import *
# Get_Mask
# 判断tile与Mask是否有重合
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





if __name__ == "__main__":
    parser = basic_parser()
    args = parser.parse_args()
    # 对 5to6_Videos 计算 psnr
    mp4_folder = '/data/wenxuan/4k_30fps'
    viewpoint_root = '/data/wenxuan/head-tracking-master'
    save_path = '/data/wenxuan/GCN_data/vpsnr_folder/'
  #  ALL_VIDEOS = [ 'BTSRun','AirShow','Surfing', 'Waterskiing',  'F5Fighter', 'StarryPolar', 'BlueWorld',  'WaitingForLove',  'LOL']
    ALL_VIDEOS =[ 'Symphony']
    time_stamp = ['06to07','07to08', '08to09', '09to10', '10to11']
    cal_videos = []
    for video in ALL_VIDEOS:
        for tmp_time in time_stamp:
            cal_videos.append(tmp_time + '_' + video)
    for video in cal_videos:
        # 读取 视口具体包括的tile
        parameters = os.listdir(os.path.join(args.PSNR_path, video))
        parameters.sort()
        PSNR_folder = os.path.join(args.PSNR_path, video )
        psnr_result = {}
        for para in parameters:
            qp, width, height = split_qp_height_width(para)
            tile_in_viewport = np.load(os.path.join(args.viewport_tiles, video, '%04dx%04d.npy'%(width, height)), allow_pickle=True).item()
            psnr_logs_path = os.path.join(PSNR_folder, para)
            average_mse = return_vmse(tile_in_viewport, psnr_logs_path)
            psnr = 10 * np.log10(255.0 **2 / average_mse)
            psnr_result[para] = psnr
        np.save(os.path.join(save_path,'{}.npy'.format(video)) ,psnr_result)
        print("{} done!".format(video))



        
       