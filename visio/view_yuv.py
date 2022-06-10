
'''
对每个frame的byte数进行分析并画图
'''
import matplotlib.pyplot as plt
import numpy as np
from PSNR import *

result1 = Read_Yuv_Vid('./test5to6_AirShow.yuv', 1920, 3840, 'cuda:0')
result2 = Read_Yuv_Vid('/data/wenxuan/all_yuvs/5to6_AirShow.yuv', 1920, 3840, 'cuda:0')
print((result1 == result2).all())

with open('test.txt', mode = 'r') as f:
    lines = f.readlines()

video_name = ['5to6_A380_tile_1', '5to6_BTSRun_tile_1', '5to6_BTSRun_tile_2']
size_data = []
for i in range(len(lines)):
    if i % 31 != 0:
        size_data.append(float(lines[i]))
plt.plot(size_data[:30], label = video_name[0], marker = 'o')
plt.plot(size_data[30:60], label = video_name[1], marker = 'o')
# plt.plot(size_data[60:90], label = video_name[2], marker = 'o')
size_array = np.array(size_data)
mean_1 = (size_array[31] + size_array[40] + size_array[50]) /3 
mean_true = np.mean(size_array[31:60])
print(" my mean : {} true mean {}".format(mean_1, mean_true))
plt.legend()
plt.xlabel('frame_number')
plt.ylabel('size/byte')
plt.savefig('test.png')
print('done!')