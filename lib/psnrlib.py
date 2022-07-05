import numpy as np
import os

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

def read_average_mse(log_path):
    '''
    读取log中的mse, 返回tile的平均mse
    '''
    with open(log_path, mode = 'r') as f:
        lines = f.readlines()
    mse = []
    for i in range(len(lines)):
        # 找到mse_avg后面的数
        data = lines[i].split(' ')
        tmp_mse = data[1].split(':')[1]
        mse.append(float(tmp_mse))
    return np.mean(np.array(mse))

def read_frame_mse(log_path, frame_num):
    with open(log_path, mode = 'r') as f:
        lines = f.readlines()
        # 找到mse_avg后面的数
    data = lines[frame_num - 1].split(' ')
    tmp_mse = data[1].split(':')[1]
   
    return float(tmp_mse)

def return_vmse(tile_in_viewport, log_folder,  client_num = 40, frame_num = 30):
    average_mse = 0
    for user in range(client_num):
        user_mse = 0
        for frame in range(frame_num):
            view_tiles = tile_in_viewport['%02d_%02d'%(user, frame)]
            frame_mse = 0
            for tile, view_segment in view_tiles.items():
                mse = read_frame_mse(os.path.join(log_folder, tile + '.log') , frame+1)
                frame_mse += mse * view_segment
            user_mse += frame_mse/frame_num
        average_mse += user_mse/client_num
    return average_mse

def return_all_tile_union(all_view_tiles):
    tile = all_view_tiles['00_00'].keys()
    for key, value in all_view_tiles.items():
        tile = tile | all_view_tiles[key].keys()
    return tile