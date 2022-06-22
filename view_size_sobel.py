'''
查看 具体的eta值和实际得到的eta值是否不太一样
'''
import os
import numpy as np
from def_parser import basic_parser
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




if __name__ =="__main__":
    parser = basic_parser()
    args = parser.parse_args()
    view_qp = '0028'
    view_width_height = '0384x0192'
    tile_in_viewpoint_path = '/data/wenxuan/GCN_data/is_tile_in_viewpoint_folder'
    video_name = '5to6_BTSRun'
    size_dict = np.load(os.path.join(args.size_dict_folder, video_name + '.npy'), allow_pickle=True).item()

    # eta划分为4个维度进行验证
    sobel_image = np.load(os.path.join('/data/wenxuan/png_sobel', '{}.npy'.format(video_name)),allow_pickle=True).item()
    sobel_value = sobel_image['01']
    for i in range(2, 31):
        sobel_value = sobel_value + sobel_image['%02d'%(i)]
    plt.imsave('{}_sobel.png'.format(video_name), sobel_value, cmap = 'gray')
    height_divider = [480, 960, 1440]
    height_divider = np.array(height_divider)
    view_video_result('0020', '0384x0192', video_name)
    percentage_error = []
    for qp_width_height, tile_size in size_dict.items():
        
        qp, width, height = split_qp_height_width(qp_width_height)
        in_viewpoint_tiles = np.load(os.path.join(tile_in_viewpoint_path, video_name, '%04dx%04d.npy'%(width, height)), allow_pickle=True).item()  
        # 分割eta
        if height < 640:
            eta = split_eta_with_latitude(tile_size, height_divider, args.WIDTH, args.HEIGHT)
        else:
            eta = np.ones((len(height_divider) + 1)) * tile_size['byte_size']/args.WIDTH/ args.HEIGHT
        all_user_pixels = return_latitude_pixels(in_viewpoint_tiles, height_divider)
        all_user_sizes = return_viewport_tile_size(tile_size, in_viewpoint_tiles)
        result = all_user_pixels * eta[np.newaxis, :]
        result = np.sum(result, axis = 1)
        if qp == 28:
            percentage_error.append(np.abs((np.mean(result)-np.mean(all_user_sizes)))/np.mean(all_user_sizes)*100)
    plt.plot(percentage_error, label = 'split')
    plt.legend()
    plt.title('percentage error')
    plt.savefig('percentage_error_qp_28_{}.png'.format(video_name))
    plt.close()