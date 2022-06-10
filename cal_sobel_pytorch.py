import torch
import numpy as np
from torch import nn
from PIL import Image
import torch.nn.functional as F
import cv2
import os 
import matplotlib.pyplot as plt


def edge_conv2d(im):
    # 用nn.Conv2d定义卷积操作
    # 注意pytorch 为 B， N， W,H
    conv_op = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
    if im.shape[1] == 1:
        channel_num = 1
    else:
        channel_num = 3
    # 定义sobel算子参数
    
    sobel_kernel_horizonal = np.array([[-1, 0,1],[-2, 0,2],[-1,0,1]], dtype = 'float32')
    sobel_kernel_vertical = np.array([[1,2,1], [0,0,0],[-1,-2,-1]], dtype = 'float32')
    sobel_kernel_horizonal = sobel_kernel_horizonal.reshape((1, 1, 3, 3))
    sobel_kernel_vertical = sobel_kernel_vertical.reshape((1,1,3,3))
    # 将sobel算子转换为适配卷积操作的卷积核
    if channel_num == 3:
        sobel_kernel_vertical.repeat(sobel_kernel_vertical, 3 , axis = 1)
        sobel_kernel_vertical.repeat(sobel_kernel_vertical, 3 , axis = 0)
        sobel_kernel_horizonal.repeat(sobel_kernel_horizonal, 3, axis = 1)
        sobel_kernel_horizonal.repeat(sobel_kernel_horizonal, 3, axis = 0)
    # 卷积输出通道，这里我设置为3
    
    conv_op.weight.data = torch.from_numpy(sobel_kernel_horizonal)
    horizon_result = conv_op(im)
    conv_op.weight.data = torch.from_numpy(sobel_kernel_vertical)
    vertical_result = conv_op(im)
    horizon_result = horizon_result.squeeze()
    vertical_result = vertical_result.squeeze()
    result = torch.sqrt(horizon_result **2 + vertical_result**2)
    # 将输出转换为图片格式
    
    return horizon_result.detach().numpy(), vertical_result.detach().numpy(), result.detach().numpy()

if __name__ == "__main__":
    png_path = '/data/wenxuan/GCN_data/all_pngs'
    videos = os.listdir(png_path)
    for video in videos[76:]:
        sobel = {}
        for i in range(1, 31):
            if i > 1:
                im_1 = Image.open(os.path.join(png_path, video, '%02d.png'%(i))).convert('L')
                im = Image.open(os.path.join(png_path, video, '%02d.png'%(i - 1 ))).convert('L')
                im = np.array(im, dtype = 'float32')
                im_1 = np.array(im_1, dtype = 'float32')
                input = im_1 - im
            else:
                input = Image.open(os.path.join(png_path, video, '%02d.png'%(i))).convert('L')
                input = np.array(input, dtype='float32')
            input = torch.from_numpy(input.reshape(1,1, input.shape[0], input.shape[1]))
            horizon, vertical, result = edge_conv2d(input)

            sobel['%02d'%(i)] = result
        np.save( os.path.join('/data/wenxuan/png_sobel','%s.npy'%(video)),sobel)
