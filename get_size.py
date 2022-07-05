import os 
import numpy as np
import csv
from def_parser import ALL_VIDEOS, basic_parser

parser = basic_parser()
args = parser.parse_args()
'''
读入
'''
encode_folder = args.encode_folder
size_dict_folder = args.size_dict_folder
videos = []
#ALL_VIDEOS = ['Surfing', 'Waterskiing', 'AirShow', 'F5Fighter', 'StarryPolar', 'BlueWorld', 'BTSRun',  'WaitingForLove',  'LOL']
ALL_VIDEOS = [ 'Surfing', 'Waterskiing']
time_stamp = ['07to08', '08to09', '09to10', '10to11']
for video in ALL_VIDEOS:
    for tmp_time in time_stamp:
        videos.append(tmp_time + '_' + video)
for video in videos:
    all_parameters = os.listdir(os.path.join(encode_folder, video))
    all_parameters.sort()
    video_dict = {}
    for parameter in all_parameters:
        # 读入文件中所有hevc的size，保存在字典中
        size_dict = {}
        hevcs = os.listdir(os.path.join(encode_folder, video, parameter))
        # 检查数据数量是否正确
        qp, tile_size = parameter.split('_')
        width, height = tile_size.split('x')
        assert len(hevcs) == int(args.WIDTH/int(width))*int(args.HEIGHT/int(height)), "video_name: {} parameter: {}".format(video, parameter)
        for hevc in hevcs:
            size_dict[hevc[:-5]] = os.path.getsize(os.path.join(encode_folder, video, parameter, hevc))
            assert size_dict[hevc[:-5]] != 0, "video_name: {} parameter: {} hevc {}".format(video, parameter, hevc)
        tmp_size = sum(size_dict.values())
        size_dict['byte_size'] = tmp_size
        size_dict['bits/pixel'] = 8*tmp_size/(args.HEIGHT*args.WIDTH)
        video_dict[parameter] = size_dict
    np.save(os.path.join(size_dict_folder, '%s.npy'%(video)),video_dict)
