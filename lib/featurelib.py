import torch
import numpy as np
from torch import nn
from PIL import Image
from PSNR import Get_Mask, get_partial_viewpoint
import os
import matplotlib.pyplot as plt


def sobel_conv2d(im, device):
    # 用nn.Conv2d定义卷积操作
    # 注意pytorch 为 B， N， W,H
    conv_op = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
    if im.shape[1] == 1:
        channel_num = 1
    else:
        channel_num = 3
    # 定义sobel算子参数
    sobel_kernel_horizonal = np.array(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype="float32"
    )
    sobel_kernel_vertical = np.array(
        [[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype="float32"
    )
    sobel_kernel_horizonal = sobel_kernel_horizonal.reshape((1, 1, 3, 3))
    sobel_kernel_vertical = sobel_kernel_vertical.reshape((1, 1, 3, 3))
    # 将sobel算子转换为适配卷积操作的卷积核
    if channel_num == 3:
        sobel_kernel_vertical.repeat(sobel_kernel_vertical, 3, axis=1)
        sobel_kernel_vertical.repeat(sobel_kernel_vertical, 3, axis=0)
        sobel_kernel_horizonal.repeat(sobel_kernel_horizonal, 3, axis=1)
        sobel_kernel_horizonal.repeat(sobel_kernel_horizonal, 3, axis=0)
    # 卷积输出通道，这里我设置为3
    conv_op.weight.data = torch.from_numpy(sobel_kernel_horizonal).to(device)
    horizon_result = conv_op(im)
    conv_op.weight.data = torch.from_numpy(sobel_kernel_vertical).to(device)
    vertical_result = conv_op(im)
    horizon_result = horizon_result.squeeze()
    vertical_result = vertical_result.squeeze()
    result = torch.sqrt(horizon_result**2 + vertical_result**2)
    # 将输出转换为图片格式
    # 只对大于0的数据进行处理看看
    return horizon_result.detach(), vertical_result.detach(), result.detach()


def laplace_conv2d(im, device):
    conv_op = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
    if im.shape[1] == 1:
        channel_num = 1
    else:
        channel_num = 3
    laplace_kernel = np.array(
        [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype="float32"
    )
    laplace_kernel = laplace_kernel.reshape((1, 1, 3, 3))
    if channel_num == 3:
        laplace_kernel.repeat(laplace_kernel, 3, axis=1)
        laplace_kernel.repeat(laplace_kernel, 3, axis=0)
    conv_op.weight.data = torch.from_numpy(laplace_kernel).to(device)
    result = conv_op(im)
    return torch.abs(result.detach())


def gaussian_conv2d(im, device):
    conv_op = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
    if im.shape[1] == 1:
        channel_num = 1
    else:
        channel_num = 3
    gaussian_kernel = np.array(
        [
            [0.05854983, 0.09653235, 0.05854983],
            [0.09653235, 0.15915494, 0.09653235],
            [0.05854983, 0.09653235, 0.05854983],
        ],
        dtype="float32",
    )
    gaussian_kernel = gaussian_kernel / np.sum(gaussian_kernel)
    gaussian_kernel = gaussian_kernel.reshape((1, 1, 3, 3))
    if channel_num == 3:
        gaussian_kernel.repeat(gaussian_kernel, 3, axis=1)
        gaussian_kernel.repeat(gaussian_kernel, 3, axis=0)
    conv_op.weight.data = torch.from_numpy(gaussian_kernel).to(device)
    result = conv_op(im)
    return torch.abs(result.detach())

def roberts_conv2d(im, device):
    # 用nn.Conv2d定义卷积操作
    # 注意pytorch 为 B， N， W, H
    conv_op = nn.Conv2d(3, 3, kernel_size=2, padding=1, bias=False)
    if im.shape[1] == 1:
        channel_num = 1
    else:
        channel_num = 3
    # 定义sobel算子参数
    Robert_kernel_horizonal = np.array([[-1, 0], [0, 1]], dtype="float32")
    Robert_kernel_vertical = np.array([[0, -1], [1, 0]], dtype="float32")
    Robert_kernel_horizonal = Robert_kernel_horizonal.reshape((1, 1, 2, 2))
    Robert_kernel_vertical = Robert_kernel_vertical.reshape((1, 1, 2, 2))
    # 将sobel算子转换为适配卷积操作的卷积核
    if channel_num == 3:
        Robert_kernel_vertical.repeat(Robert_kernel_vertical, 3, axis=1)
        Robert_kernel_vertical.repeat(Robert_kernel_vertical, 3, axis=0)
        Robert_kernel_horizonal.repeat(Robert_kernel_horizonal, 3, axis=1)
        Robert_kernel_horizonal.repeat(Robert_kernel_horizonal, 3, axis=0)
    # 卷积输出通道，这里我设置为3
    conv_op.weight.data = torch.from_numpy(Robert_kernel_horizonal).to(device)
    horizon_result = conv_op(im)
    conv_op.weight.data = torch.from_numpy(Robert_kernel_vertical).to(device)
    vertical_result = conv_op(im)
    horizon_result = horizon_result.squeeze()
    vertical_result = vertical_result.squeeze()
    horizon_result = horizon_result[:-1, :-1]
    vertical_result = vertical_result[:-1, :-1]
    result = torch.sqrt(horizon_result**2 + vertical_result**2)
    # 将输出转换为图片格式
    # 只对大于0的数据进行处理看看
    return horizon_result.detach(), vertical_result.detach(), result.detach()


def cal_video_feature(png_path, video, device, method="sobel"):
    """
    output: frameNum, HEIGHT, WIDTH
    """
    output_feature = {}
    for i in range(1, 31):
        if i > 1:
            im_1 = Image.open(os.path.join(png_path, video, "%02d.png" % (i))).convert(
                "L"
            )
            im = Image.open(
                os.path.join(png_path, video, "%02d.png" % (i - 1))
            ).convert("L")
            im = np.array(im, dtype="float32")
            im_1 = np.array(im_1, dtype="float32")
            input = im_1 - im
        else:
            input = Image.open(os.path.join(png_path, video, "%02d.png" % (i))).convert(
                "L"
            )
            input = np.array(input, dtype="float32")
        input = torch.from_numpy(
            input.reshape(1, 1, input.shape[0], input.shape[1])
        ).to(device)
        if method == "sobel":
            horizon, vertical, result = sobel_conv2d(input, device)
        elif method == "laplace":
            result = laplace_conv2d(input, device)
        elif method == "robert":
            horizon, vertical, result = roberts_conv2d(input, device)
        elif method == 'gaussian_laplace':
            result = gaussian_conv2d(input, device)
            result = laplace_conv2d(result, device)
        else:
            raise NotImplementedError
        output_feature["%02d" % (i)] = result
    return output_feature


def cal_horizon_vertical(png_path, video, device, method="sobel"):
    output_feature = {}
    for i in range(1, 31):
        if i > 1:
            im_1 = Image.open(os.path.join(png_path, video, "%02d.png" % (i))).convert(
                "L"
            )
            im = Image.open(
                os.path.join(png_path, video, "%02d.png" % (i - 1))
            ).convert("L")
            im = np.array(im, dtype="float32")
            im_1 = np.array(im_1, dtype="float32")
            input = im_1 - im
        else:
            input = Image.open(os.path.join(png_path, video, "%02d.png" % (i))).convert(
                "L"
            )
            input = np.array(input, dtype="float32")
        input = torch.from_numpy(
            input.reshape(1, 1, input.shape[0], input.shape[1])
        ).to(device)
        if method == "sobel":
            horizon, vertical, result = sobel_conv2d(input, device)
        elif method == "robert":
            horizon, vertical, result = roberts_conv2d(input, device)
        output_feature["%02d" % (i)] = [torch.abs(horizon), torch.abs(vertical)]
    return output_feature


def get_GOP_mask(
    viewpoint_root, mp4_folder, video, device, HEIGHT=1920, WIDTH=3840, is_all_one=True
):
    """
    viewpoint_root:head-tracking_master
    mp4_folder: 4k_30fps
    video: 5to6_A380
    """
    #   np.save( os.path.join('/data/wenxuan/png_sobel','%s.npy'%(video)),sobel)
    viewpoint = get_partial_viewpoint(viewpoint_root, mp4_folder, video)
    clientNum = viewpoint.shape[0]
    frameNum = viewpoint.shape[1]
    all_mask = torch.zeros((HEIGHT, WIDTH), dtype=torch.float32).to(device)
    for i in range(clientNum):
        for j in range(frameNum):
            mask = Get_Mask(HEIGHT, WIDTH, viewpoint[i, j, 1], viewpoint[i, j, 0]).to(
                device
            )
            all_mask += mask
    if not is_all_one:
        all_mask = all_mask / torch.max(all_mask)
    else:
        ones = torch.ones_like(all_mask)
        all_mask = torch.where(all_mask > 0, ones, all_mask)
    return all_mask

if __name__ == "__main__":
    gaussian = np.array(
        [
            [0.05854983, 0.09653235, 0.05854983],
            [0.09653235, 0.15915494, 0.09653235],
            [0.05854983, 0.09653235, 0.05854983],
        ],
        dtype="float32",
    )
    plt.imshow(gaussian, cmap='gray')
    plt.savefig('gaussian.png')