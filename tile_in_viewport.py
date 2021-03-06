'''
判断 tile是否在视口内
输出: 字典， key为分块的位置与大小， value为(40)数组, 1代表在， 0代表不在
'''
from PSNR import *
import os
import torch
import numpy as np
from def_parser import *
import time


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
    mp4_folder = '/data/wenxuan/4k_30fps'
    viewpoint_root = '/data/wenxuan/head-tracking-master'
    saved_path = '/data/wenxuan/GCN_data/tile_in_viewpoint_npy'
  #  ALL_VIDEOS = ['AirShow','Surfing', 'Waterskiing',  'F5Fighter', 'StarryPolar', 'BlueWorld', 'BTSRun',  'WaitingForLove',  'LOL']
    ALL_VIDEOS = ['Symphony']
    use_videos = []
    for video in ALL_VIDEOS:
        for tmp_time in time_stamp:
            use_videos.append(tmp_time + '_' +video )
    for video in use_videos:
        t_1 = time.time()
        t, video_name = video.split('_')
        t_begin, t_end = t.split('to')
        t_begin = int(t_begin)
        t_end = int(t_end)
        total_viewpoint = Get_Viewpoint(viewpoint_root, os.path.join(mp4_folder, video_name + '.mp4'), video_name)
        viewpoint = total_viewpoint[:, 30*t_begin:30*t_end, :]
        clientNum = viewpoint.shape[0]
        frameNum = viewpoint.shape[1]
        if not os.path.exists(os.path.join(saved_path, video)):
            os.mkdir(os.path.join(saved_path, video))
        for w in W_size:
            for h in H_size:
                x_list, y_list = get_left_top(args.WIDTH, args.HEIGHT, w, h)
                w_h_dict = {}
                    # 判断tile是否在Mask内
                for i in range(clientNum):
                    for j in range(frameNum):
                        user_frame_dict = {}
                        mask = Get_Mask(args.HEIGHT, args.WIDTH, viewpoint[i, j, 1], viewpoint[i, j, 0]).to(args.device)
                        # 转为float32,防止数值过大
                        pixels = torch.sum(mask)
                        for tile_idx in range(len(x_list)):
                            parameter = '%04d_%04d_%04d_%04d'%(x_list[tile_idx], y_list[tile_idx], w, h)
                            if (mask[y_list[tile_idx]:y_list[tile_idx] + h, x_list[tile_idx]:x_list[tile_idx]+w]).any():
                                # 得到比例
                                view_part = torch.sum(mask[y_list[tile_idx]:y_list[tile_idx] + h, x_list[tile_idx]:x_list[tile_idx]+w])/pixels
                                if args.device == 'cpu':
                                    user_frame_dict[parameter] = view_part.numpy()
                                else:
                                    user_frame_dict[parameter] = view_part.cpu().numpy()
                        w_h_dict['%02d_%02d'%(i,j)] = user_frame_dict
                np.save(os.path.join(saved_path , video,  '%04dx%04d.npy'%(w,h)), w_h_dict)
        print('a video time is {}'.format(time.time() - t_1))
        print("a video {} saved".format(video))