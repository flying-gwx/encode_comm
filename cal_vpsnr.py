
from PSNR import *
import os
import torch
import numpy as np
from def_parser import *
import time


def read_average_mse(log_path):
    '''
    读取log中的mse, 返回tile的平均mse
    '''
    with open(log_path, mode = 'r') as f:
        lines = f.readlines()
    mse = []
    for i in range(len(lines)):
        # 找到mse_avg后面的数
        data = lines[i].split(lines[i])
        tmp_mse = data[1].split(':')[1]
        mse.append(float(tmp_mse))
    return np.mean(np.array(mse))


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
    saved_path = '/data/wenxuan/GCN_data/is_tile_in_viewpoint_folder'
    ALL_VIDEOS = ['AirShow','Surfing', 'Waterskiing',  'F5Fighter', 'StarryPolar', 'BlueWorld', 'BTSRun',  'WaitingForLove',  'LOL']
    TRAIN_VIDEOS = ['Surfing','AirShow', 'StarryPolar', 'BlueWorld', 'LOL', 'Dota2', 'BTSRun']
    for i in range(len(ALL_VIDEOS)):
        ALL_VIDEOS[i] = '5to6_' + ALL_VIDEOS[i]
    for video in ALL_VIDEOS:
        t1 = time.time()
        t, video_name = video.split('_')
        total_viewpoint = Get_Viewpoint(viewpoint_root, os.path.join(mp4_folder, video_name + '.mp4'), video_name)
        if t == "5to6":
            viewpoint = total_viewpoint[:, 150:180, :]
        else:
            viewpoint = total_viewpoint[:, 450:480, :] 
        clientNum = viewpoint.shape[0]
        frameNum = viewpoint.shape[1]
        video_dict = {}
        for w in W_size:
            for h in H_size:
                    x_list, y_list = get_left_top(args.WIDTH, args.HEIGHT, w, h)
                    w_h_dict = {}
                        # 判断tile是否在Mask内
                    for i in range(clientNum):
                        for j in range(frameNum):
                            user_frame_dict = {}
                            mask = Get_Mask(args.HEIGHT, args.WIDTH, viewpoint[i, j, 1], viewpoint[i, j, 0]).to('cuda:0')
                            for tile_idx in range(len(x_list)):
                                parameter = '%04d_%04d_%04d_%04d'%(x_list[tile_idx], y_list[tile_idx], w, h)
                                if (mask[y_list[tile_idx]:y_list[tile_idx] + h, x_list[tile_idx]:x_list[tile_idx]+w]).any():
                                    user_frame_dict[parameter] = 1
                            w_h_dict['%02d_%02d'%(i,j)] = user_frame_dict
                    video_dict['%04dx%04d'%(w,h)] = w_h_dict
        np.save(os.path.join(saved_path , '{}.npy'.format(video)), video_dict)
        print("a video {} saved".format(video))