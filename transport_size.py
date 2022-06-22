
import os
import numpy as np
from def_parser import basic_parser
import numpy as np
from lib.sizelib import *

if __name__ == "__main__":
    parser = basic_parser()
    args = parser.parse_args()
    ALL_VIDEOS = ['AirShow','Surfing', 'Waterskiing', 'F5Fighter', 'StarryPolar', 'BlueWorld', 'BTSRun',  'WaitingForLove',  'LOL']
    time_stamp = ['06to07', '07to08', '08to09', '09to10', '10to11']
    tile_in_viewpoint_path = '/data/wenxuan/GCN_data/is_tile_in_viewpoint_folder'
    # 读入video
    for time in time_stamp:
        for video in ALL_VIDEOS:
            video_name = time + '_' + video
            transport = {}
            size_dict = np.load(os.path.join(args.size_dict_folder, video_name + '.npy'), allow_pickle=True).item()
         #   is_tile_in_viewpint = np.load(os.path.join(tile_in_viewpoint_path, video_name + '.npy'), allow_pickle=True).item()
            for qp_width_height, tile_size in size_dict.items():
                qp, width, height = split_qp_height_width(qp_width_height)
                is_tile_in_viewpint = np.load(os.path.join(tile_in_viewpoint_path, video_name, '%04dx%04d.npy'%(width, height)),allow_pickle=True).item()
                all_user_sizes = return_viewport_tile_size(tile_size, is_tile_in_viewpint)
                all_user_pixels = return_viewport_tile_pixels(is_tile_in_viewpint)
                average_band = 8*np.mean(all_user_sizes)
                average_pixel = np.mean(all_user_pixels)
                average_eta = average_band / average_pixel
                transport[qp_width_height] = np.array([average_eta, average_band, average_pixel])
            np.save(os.path.join(args.transport_result_path, video_name + '.npy'), transport)
            print("{} done!".format(video_name))
            # sobel_image = np.load(os.path.join('/data/wenxuan/png_sobel', '{}.npy'.format(video_name)),allow_pickle=True).item()
            # sobel_value = sobel_image['01']
            # for i in range(2, 31):
            #     sobel_value = sobel_value + sobel_image['%02d'%(i)]
            # 假设事先不知道视点先验结果
            


                # 看一下用高sobel区域的eta和实际的eta比差多少