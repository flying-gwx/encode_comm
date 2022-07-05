
import os
import numpy as np
from def_parser import basic_parser
import numpy as np
from lib.sizelib import *
from datautilis.view_label import *
import time

if __name__ == "__main__":
    parser = basic_parser()
    args = parser.parse_args()
 #   ALL_VIDEOS = ['WaitingForLove','AirShow','Surfing', 'Waterskiing', 'F5Fighter', 'StarryPolar', 'BlueWorld', 'BTSRun',  'LOL']
    ALL_VIDEOS = [ 'Surfing', 'Waterskiing']
    time_stamp = ['07to08', '08to09', '09to10', '10to11']
    tile_in_viewpoint_path = '/data/wenxuan/GCN_data/tile_in_viewpoint_npy'
    # 读入video
    for tmp_time in time_stamp:
        for video in ALL_VIDEOS:
            video_name = tmp_time + '_' + video
            transport = {}
            size_dict = np.load(os.path.join(args.size_dict_folder, video_name + '.npy'), allow_pickle=True).item()
         #   is_tile_in_viewpint = np.load(os.path.join(tile_in_viewpoint_path, video_name + '.npy'), allow_pickle=True).item()
            for qp_width_height, tile_size in size_dict.items():
                t1 = time.time()
                qp, width, height = split_qp_height_width(qp_width_height)
                is_tile_in_viewpint = np.load(os.path.join(tile_in_viewpoint_path, video_name, '%04dx%04d.npy'%(width, height)),allow_pickle=True).item()
                all_user_sizes = return_viewport_tile_size(tile_size, is_tile_in_viewpint)
                all_user_pixels = return_viewport_tile_pixels(is_tile_in_viewpint)
                average_band = 8*np.mean(all_user_sizes)
                average_pixel = np.mean(all_user_pixels)
                average_eta = average_band / average_pixel
                transport[qp_width_height] = np.array([average_eta, average_band, average_pixel])
       #         print("cal culate time is {}".format(time.time() - t1))
            all_eta = []
            video_eta = []
            for key, value in transport.items():
                all_eta.append(value[0])
                video_eta.append(size_dict[key]['bits/pixel'])
            view_error(all_eta, title='{} bits/pixels'.format(video), save_name='{}_eta.png'.format(video_name))
            relative_error = np.abs(np.array(all_eta) - np.array(video_eta))/np.array(video_eta) * 100
            view_error(relative_error, title = '{} percentage error(%)'.format(video_name), save_name='{}_percentage_error.png'.format(video_name))
            np.save(os.path.join(args.transport_result_path, video_name + '.npy'), transport)
            print("{} done!".format(video_name))