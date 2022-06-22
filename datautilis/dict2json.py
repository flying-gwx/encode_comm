import json
from socket import TCP_LINGER2
import numpy as np
from multiprocessing import Pool
import os
import time


tmp_dict = {'aa_bb':0.0001, 'CC_DD':0.02}

def dict2json(tmp_dict, save_name):   
    result = json.dumps(tmp_dict)
    with open(save_name, 'w') as f:
        f.write(result)

def json2dict(json_name):
    with open(json_name, 'r') as f: 
        content = f.read()
    result = json.loads(content)
    return result

# 将字典转换为json
tile_in_viewpoint_path = '/data/wenxuan/GCN_data/is_tile_in_viewpoint_folder'
json_path =  '/data/wenxuan/GCN_data/is_tile_in_viewport_json'

video_name = '06to07_AirShow'
width = 1920
height = 960
t1 = time.time()
tmp_dict = np.load(os.path.join(tile_in_viewpoint_path, video_name, '%04dx%04d.npy'%(width, height)),allow_pickle=True).item()
print("npy load time is {}".format(time.time() - t1))
dict2json(tmp_dict, '%04dx%04d.json'%(width, height))
t2 = time.time()
result = json2dict('%04dx%04d.json'%(width, height))
print("json load time is {}".format(time.time() - TCP_LINGER2))
print("done!")