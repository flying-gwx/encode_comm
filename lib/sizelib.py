import os
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import ipdb

def split_parameter(parameter):
    data = parameter.spllit('_')
    qp = int(data[0])
    width, height = data[1].split('x')
    return qp, int(width), int(height)

def split_position_height_width(parameter):
    data = parameter.split('_')
    return int(data[0]), int(data[1]), int(data[2]), int(data[3])
    
def split_qp_height_width(parameter):
    data = parameter.split('_')
    qp = float(data[0])
    width, height = data[1].split('x')
    return qp, float(width), float(height)

def visio_tiles_in_viewpoint(all_tiles, tile_parameter, video):
    '''
    可视化tile被看到的tile数量
    '''  
    visio = np.zeros((1920, 3840))
    for key, value in all_tiles.items():
        x, y, width, height = split_position_height_width(key)
        visio[y:y+height, x:x+width] = value
    plt.imshow(visio,cmap='gray')
    plt.title('{} tile_size {}:number of times watched'.format(video,tile_parameter))
    plt.savefig('{}_{}.png'.format(video,tile_parameter))
    plt.close()

def view_bytes(tile_dict,tile_parameter, video, qp):
    visio = np.zeros((1920, 3840))
    for key, value in tile_dict.items():
        x, y, width, height = split_position_height_width(key)
        visio[y:y+height, x:x+width] = value
    plt.imshow(visio,cmap='gray')
    plt.title('{} tile_size {} qp {}:tile bytes'.format(video,tile_parameter, int(qp)))
    plt.savefig('{}_{}_bytes.png'.format(video,tile_parameter))
    plt.close()
'''
对size 进行划分得到不同维度的eta值
'''

'''
得到实际的视口的byte数量
'''
def return_viewport_tile_size(all_tile_size, in_viewpoint_tiles, clientNum = 40, frameNum = 30):
    '''
    in_viewpoint_tiles 中储存40x30的view_tile数据
    all_tile_size中保存着每个位置的tile_size
    返回：clientNum数组，被看到的tile的size，单位为byte
    '''
    all_user_size = []
    for user in range(clientNum):
        user_tile = {}
        user_size = 0
        for frame in range(frameNum):
            user_tile = user_tile | in_viewpoint_tiles['%02d_%02d'%(user, frame)].keys()
        for x_y_w_h in user_tile:
            user_size+= all_tile_size[x_y_w_h]
        all_user_size.append(user_size)
    return np.array(all_user_size)

def return_viewport_tile_pixels(in_viewpoint_tiles, clientNum = 40, frameNum = 30):
    all_user_pixels = []

    for user in range(clientNum):
            user_pixel = 0
            user_tile = {}
            for frame in range(frameNum):
                user_tile = user_tile | in_viewpoint_tiles['%02d_%02d'%(user, frame)].keys()
            for x_y_w_h in user_tile:
                x, y, w,h = split_position_height_width(x_y_w_h)
                user_pixel += w*h
            all_user_pixels.append(user_pixel)
    return np.array( all_user_pixels)


def return_index(height,divider):
    # 返回height 在 height_dvider中属于哪个部分
    if height < divider[0]:
        return 0
    for i in range(1, len(divider)):
        if height >= divider[i-1] and height < divider[i]:
            return i
    return len(divider)


def return_center(left_top, height, width):
    '''
    left_top: point width, point height

    '''
    return [left_top[0] + width/2, left_top[1] + height /2]

def return_divider_index(left_top, width, height,  width_divider, height_divider):
    center = return_center(left_top, height, width)
    width_index = return_index(center[0], width_divider)
    height_index = return_index(center[1], height_divider)
    return [width_index, height_index]

def split_eta_with_latitude(tile_size_dict, height_divider,width, height):
    '''
    size_dict中大小为byte
    height_split为高度分割线,根据height_split将size_dict中的tile分成几组，每组计算对应的eta
    划分完tile之后具体的eta计算
    '''
    # pixel_num需要重新计算得到
    pixelnum =  np.zeros((len(height_divider)+1)) 
    size = np.zeros((len(height_divider)+1))
    for key, value in tile_size_dict.items():
        # 只对posiiton_width_height的数据进行操作
        if len(key.split('_')) == 4:
            x, y, width, height =  split_position_height_width(key)
            index = return_divider_index(y+height/2, height_divider)
            size[index] += value   
            pixelnum[index] += 1*width* height
    assert pixelnum.all(), "{}, {}".format(width, height)
    return size/(pixelnum)

def remove_none_tile_information(size_dict):
    '''
    去除size dict中的键值
    '''
    del size_dict['byte_size']
    del size_dict['bits/pixel']
    return size_dict
def split_eta(tile_size_dict, height_divider,width_divider, width, height):
    '''
    size_dict中大小为byte
    height_split为高度分割线,根据height_split将size_dict中的tile分成几组，每组计算对应的eta
    划分完tile之后具体的eta计算
    '''
    # pixel_num需要重新计算得到
    pixelnum =  np.zeros((len(height_divider)+1, len(width_divider) + 1))
    size = np.zeros((len(height_divider)+1, len(width_divider) + 1))
    for key, value in tile_size_dict.items():
        # 只对posiiton_width_height的数据进行操作
        x, y, width, height =  split_position_height_width(key)
        index = return_divider_index([x,y], width, height, width_divider, height_divider = height_divider)
        size[index[1], index[0]] += 8*value   
        pixelnum[index[1], index[0]] += 1*width* height
  #  print("width:{}  height:{}  pixels:{}  ".format(width, height, pixelnum ))
    assert pixelnum.all(), "{}, {}".format(width, height)
    return size/(pixelnum)

def split_pixels(in_viewpoint_tiles,height_divider, width_divider,  clientNum = 40, frameNum = 30):
    all_user_pixelnum = []
    for user in range(clientNum):
            user_tile = {}
            pixelnum =  np.zeros((len(height_divider)+1, len(width_divider) + 1))
            for frame in range(frameNum):
                user_tile = user_tile | in_viewpoint_tiles['%02d_%02d'%(user, frame)].keys()
            for x_y_w_h in user_tile:
                x,y,w, h = split_position_height_width(x_y_w_h)
                index = return_divider_index([x,y], w, h, width_divider, height_divider = height_divider)
                pixelnum[index[1], index[0]] += 1*w*h
            all_user_pixelnum.append(pixelnum)
    return np.stack(all_user_pixelnum, axis = 0)

def return_latitude_pixels( in_viewpoint_tiles,height_divider,  clientNum = 40, frameNum = 30):
    '''
    in_viewpoint_tiles 中储存40x30的view_tile数据
    all_tile_size中保存着每个位置的tile_size
    返回：clientNum数组，被看到的tile的size，单位为byte
    '''
    all_user_pixelnum = []
    for user in range(clientNum):
            user_tile = {}
            user_pixels_latitude = np.zeros((len(height_divider)+1))
            for frame in range(frameNum):
                user_tile = user_tile | in_viewpoint_tiles['%02d_%02d'%(user, frame)].keys()
            for x_y_w_h in user_tile:
                x,y,w, h = split_position_height_width(x_y_w_h)
                index = return_divider_index(y+h/2, height_divider)
                user_pixels_latitude[index] +=1
            user_pixels_latitude = user_pixels_latitude * w* h
            all_user_pixelnum.append(user_pixels_latitude)
    return np.stack(all_user_pixelnum, axis = 0)
