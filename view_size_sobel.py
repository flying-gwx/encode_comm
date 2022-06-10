'''
查看 具体的eta值和实际得到的eta值是否不太一样
'''

import os
import numpy as np
from def_parser import basic_parser
import matplotlib.pyplot as plt
# 读入tile_size

parser = basic_parser()
args = parser.parse_args()
tile_in_viewpoint_path = '/data/wenxuan/GCN_data/is_tile_in_viewpoint_folder'
video_name = '5to6_AirShow'
size_dict = np.load(os.path.join(args.size_dict_folder, video_name + '.npy'), allow_pickle=True).item()
is_tile_in_viewpint = np.load(os.path.join(tile_in_viewpoint_path, video_name + '.npy'), allow_pickle=True).item()

true_eta = []
estimated_eta = []
for key, all_tile_value in size_dict.items():
    qp, width_height = key.split('_')
    estimated_eta.append(all_tile_value['bits/pixel'])
    in_viewpoint_tiles = is_tile_in_viewpint[width_height]
    # 所有在视点内的tile
    tmp_eta = 0
    for user_frame in in_viewpoint_tiles.keys():
        true_tiles = in_viewpoint_tiles[user_frame]
        # 把这些tile的size加起来
        width, height = width_height.split('x')
        width = float(width)
        height = float(height)
        pixels_num = len(true_tiles) * width * height
        size = 0
        for x_y_w_h in true_tiles.keys():
            size+= all_tile_value[x_y_w_h]
        tmp_eta += 8*size/pixels_num
    tmp_eta = tmp_eta/len(in_viewpoint_tiles.keys())
    true_eta.append(true_eta)

plt.plot(estimated_eta, label = 'estimated')
plt.plot(true_eta, label = 'true')
plt.legend()
plt.savefig('view_eta.png')
