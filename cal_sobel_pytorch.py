from email.policy import default
import torch
import numpy as np
from torch import nn
from PIL import Image
from PSNR import Get_Mask,get_partial_viewpoint
import os 
from def_parser import basic_parser
from lib.featurelib import *
import matplotlib.pyplot  as plt

def return_mean_median_std(feature, device):
    frame_result = []
    if device == 'cpu':
        frame_result.append(torch.mean(feature).numpy())
        frame_result.append(torch.median(feature).numpy())
        frame_result.append(torch.std(feature).numpy())
    else:
        frame_result.append(torch.mean(feature).cpu().numpy())
        frame_result.append(torch.median(feature).cpu().numpy())
        frame_result.append(torch.std(feature).cpu().numpy())
    return frame_result

def return_mean_std(feature, device):
    frame_result = []
    if device == 'cpu':
        frame_result.append(torch.mean(feature).numpy())
        frame_result.append(torch.std(feature).numpy())
    else:
        frame_result.append(torch.mean(feature).cpu().numpy())
        frame_result.append(torch.std(feature).cpu().numpy())
    return frame_result

def add_parser(parser):
    parser.add_argument('--mask_flag', default = 'weighted_mask')
    parser.add_argument('--feature_flag', default = 'laplace')
    return parser
if __name__ == "__main__":
    png_path = '/data/wenxuan/GCN_data/all_pngs'
    parser = basic_parser()
    args = parser.parse_args()
    mp4_folder = '/data/wenxuan/4k_30fps'
    viewpoint_root = '/data/wenxuan/head-tracking-master'
    feature_saved_path = '/data/wenxuan/GCN_data/no_mask_laplace_feature'
    if not os.path.exists(feature_saved_path):
        os.mkdir(feature_saved_path)
    mask_flag = 'no_mask'
    feature_flag = 'laplace'
    device = args.device
    is_normalized = True
    frameNum = 30
    ALL_VIDEOS = ['Symphony']
    
    videos = ['5to6_WesternSichuan']
    for video in ALL_VIDEOS:
        videos.append('06to07_' + video)
        videos.append('07to08_' + video)
        videos.append('08to09_' + video)
        videos.append('09to10_' + video)
        videos.append('10to11_' + video)
    for video in videos:
        if mask_flag == 'no_mask':
            all_mask = torch.ones((args.HEIGHT, args.WIDTH),device = args.device)
        elif mask_flag =='ones_mask':
            all_mask = get_GOP_mask(viewpoint_root, mp4_folder, video, device = args.device, is_all_one = True)
        elif mask_flag == 'weighted_mask':
            all_mask = get_GOP_mask(viewpoint_root, mp4_folder, video, device = args.device, is_all_one = False)
        
        feature = []
        if feature_flag == 'sobel':
            sobel = cal_video_feature(png_path, video, device=args.device)
            for frame in range(1,frameNum + 1):
                view_sobel = sobel['%02d'%(frame)] * all_mask
                sobel_result = return_mean_median_std(view_sobel, args.device)
                feature.append(sobel_result)
        elif feature_flag == 'laplace':
            laplace = cal_video_feature(png_path, video,device=args.device, method = 'laplace')
            
            for frame in range(1,frameNum + 1):
                view_laplace = laplace['%02d'%(frame)] * all_mask
                plt.imshow(view_laplace[0, 0, :,:].cpu().numpy(), cmap= 'gray')
                plt.colorbar()
                plt.savefig('lapalce.png')
                plt.close()
                laplace_result = return_mean_median_std(view_laplace, args.device)
                feature.append(laplace_result)
        elif feature_flag == 'robert':
            robert = cal_video_feature(png_path, video,device=args.device, method = 'robert')
            for frame in range(1,frameNum + 1):
                view_robert = robert['%02d'%(frame)] * all_mask
                robert_result = return_mean_median_std(view_robert, args.device)
                feature.append(robert_result)
        elif feature_flag == 'gaussian_laplace':
            laplace = cal_video_feature(png_path, video,device=args.device, method = 'gaussian_laplace')
            for frame in range(1,frameNum + 1):
                view_laplace = laplace['%02d'%(frame)] * all_mask
                laplace_result = return_mean_median_std(view_laplace, args.device)
                feature.append(laplace_result)
        feature = np.stack(feature, axis = 0)
        np.save(os.path.join(feature_saved_path, video + '.npy'), feature)

        print("feature saved!")