'''
查看 具体的eta值和实际得到的eta值是否不太一样
'''
import os
from matplotlib import markers
import numpy as np
from def_parser import *
import matplotlib.pyplot as plt
import numpy as np
from lib.sizelib import *

# 读入tile_size


def view_video_result(view_qp, view_width_height, video_name):
    '''
    可视化每个tile被看到的次数以及每个tile具体的byte数量
    '''
    true_eta = []
    estimated_eta = []
    for key, all_tile_value in size_dict.items():
        qp, width_height = key.split('_')
        estimated_eta.append(all_tile_value['bits/pixel'])
        tmp_eta = 0
        tile_view_num = {}
        for tmp_key in all_tile_value.keys():
            tile_view_num[tmp_key] = 0
        for user in range(40):
            user_tile = {}
            for frame in range(30):
                user_tile = user_tile | in_viewpoint_tiles['%02d_%02d'%(user, frame)].keys()
            for tile  in user_tile:
                tile_view_num[tile] += 1
            width, height = width_height.split('x')
            width = float(width)
            height = float(height)
            pixels_num = len(user_tile) * width * height
            size = 0
            for x_y_w_h in user_tile:
                size+= all_tile_value[x_y_w_h]
            tmp_eta += 8*size/pixels_num
        if qp == view_qp and width_height ==view_width_height:
            del tile_view_num['bits/pixel']
            del tile_view_num['byte_size']
            visio_tiles_in_viewpoint(tile_view_num, width_height, video_name)
            # 可视化tile的byte数量
            del all_tile_value['bits/pixel']
            del all_tile_value['byte_size']
            view_bytes(all_tile_value, width_height, video_name, qp )
        tmp_eta = tmp_eta/40
        true_eta.append(tmp_eta)

    estimated_eta_npy = np.array(estimated_eta)
    true_eta_npy = np.array(true_eta)
    relative_error = np.abs(true_eta_npy - estimated_eta_npy)*100 / true_eta_npy
    index = int(view_qp) - 18
    plt.plot(relative_error[index*100:index*100 + 100], marker = 'o', label = 'video average')
    plt.ylabel('%')
    plt.xlabel('coding parameter index')
    plt.title('bits/pixel percentage error(qp = {})'.format(int(view_qp)))
    plt.savefig('{}_{}_view_eta.png'.format(video_name, int(view_qp)))


def select_vis_data( data_dict, vis_tile_parameter,  same_width = True):
    result = {}
    for key, value in data_dict.items():
        qp, width, height = split_qp_height_width(key)
        if same_width:
            if width == vis_tile_parameter:
                result[height] = value
        else:
            if height == vis_tile_parameter:
                result[width] = value
    keys = sorted(result.keys())
    values = []
    for key in keys:
        values.append(result[key])
    return keys, values



def vis_difference(x_input, split_results, estimate_results, transport_results, save_name):
    plt.plot(x_input, split_results, label = 'split estimation', marker = 'o')
    plt.plot(x_input, estimate_results, label = 'no split estimation', marker = '*')
    plt.plot(x_input,transport_results, label = 'transport size', marker = '+')
    plt.legend()
    plt.xlabel('')
    plt.ylabel('average bits')
    plt.savefig(save_name)
    plt.close()

if __name__ =="__main__":
    parser = basic_parser()
    args = parser.parse_args()
    view_qp = '0040'
    view_width = 384
    view_height = 384
    tile_in_viewpoint_path = '/data/wenxuan/GCN_data/tile_in_viewpoint_npy'
    video_name = '06to07_BTSRun'
    size_dict = np.load(os.path.join(args.size_dict_folder, video_name + '.npy'), allow_pickle=True).item()
    size_keys = list(size_dict.keys())
    for qp_width_height in size_keys:
        qp, width, height = split_qp_height_width(qp_width_height)
        if qp != int(view_qp):
            del size_dict[qp_width_height]
    # eta划分为4个维度进行验证
    width_divider = [960, 1920, 2880]
  #  height_divider = np.array(height_divider)
    height_divider = [960]

    split_results = {}
    transport_results = {}
    estimate_results = {}
    for qp_width_height, tile_size in size_dict.items():
        qp, width, height = split_qp_height_width(qp_width_height)
        video_eta =  tile_size['bits/pixel']
        tile_size = remove_none_tile_information(tile_size)
        if width <= width_divider[0] and height <= height_divider[0]:
            etas = split_eta(tile_size, height_divider, width_divider, width, height)
        else: 
            etas = video_eta * np.ones((len(height_divider) + 1, len(width_divider) + 1))
        in_viewpoint_tiles = np.load(os.path.join(tile_in_viewpoint_path, video_name, '%04dx%04d.npy'%(width, height)), allow_pickle=True).item()  
        # 分割eta
        all_user_pixels = split_pixels(in_viewpoint_tiles, height_divider, width_divider)
            
        # all_user_pixels = return_latitude_pixels(in_viewpoint_tiles, height_divider)
        all_user_sizes = 8*return_viewport_tile_size(tile_size, in_viewpoint_tiles)
        result = all_user_pixels * etas[np.newaxis, :]
        result = np.sum(result, axis = (1,2))
        
        split_results[qp_width_height] = (np.mean(result))
        transport_results[qp_width_height] = (np.mean(all_user_sizes))
        etas = video_eta * np.ones((len(height_divider) + 1, len(width_divider) + 1))
        estimate_result =  all_user_pixels * etas[np.newaxis, :] 
        estimate_results[qp_width_height] = (np.mean(np.sum(estimate_result, axis = (1,2))))
   
    # 可视化 相同height
    _, witdh_split_results = select_vis_data(split_results, view_width, True)
    _, width_transport_results = select_vis_data(transport_results, view_width, True)
    height_list, width_estimate_results = select_vis_data(estimate_results, view_width, True)
    vis_difference(height_list, witdh_split_results, width_estimate_results, width_transport_results, '{}_{}_{}_qp_{}.png'.format(video_name,'width', view_width, int(view_qp)))
    print("done!")
    _, witdh_split_results = select_vis_data(split_results, view_height, False)
    _, width_transport_results = select_vis_data(transport_results, view_height, False)
    width_list, width_estimate_results = select_vis_data(estimate_results, view_height, False)
    vis_difference(width_list, witdh_split_results, width_estimate_results, width_transport_results, '{}_{}_{}_qp_{}.png'.format(video_name,'height', view_height, int(view_qp)))
    print("done!")