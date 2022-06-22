'''
this script aims to get the ep value and eta value
'''
import math
import argparse

from scipy.spatial.distance import num_obs_dm
from PSNR import Get_Viewpoint, write_txt,S_2D
import os 
import numpy as np
'''
viewpoint_used now is a numpy, get from Get_view_port 
'''
def factP_user(sw,sh,a,b,viewpoint_used,W,H):
    np_list=[]
    NP=0
    # 这里也没什么问题
    # xx 为 纬度， yy为经度，
    xx = viewpoint_used[..., 1].reshape( -1)
    yy = viewpoint_used[..., 0].reshape(-1)
    xx = xx.tolist()
    yy = yy.tolist()
    nmm_list = []
    #取最大的nmm， 最小的smm，最大的emm，最小的wmm，计算得到结果作为这1s需要传输的tile的大小
    # 计算得到相应的结果
    for p in range(len(xx)):
        nmm, smm, wmm, emm, jdcc = S_2D((math.pi*xx[p])/180, (math.pi*yy[p])/180, a, b)
        # 根据 mask的方式去判断传输多少Ep，并进行计算
        if 180-yy[p]<0.5*abs(jdcc):
            nw1=math.ceil(W/sw)-math.floor((((180+wmm)*W)/360)/sw)
            nw2=math.ceil((((180+emm)*W)/360)/sw)
            nw=nw1+nw2
            nh=math.ceil((((90+nmm)*H)/180)/sh)-math.floor((((90+smm)*H)/180)/sh)
            nn=nw*nh
            np=sw*sh*nn
        elif yy[p]-(-180)<0.5*abs(jdcc):
            nw1=math.ceil((((180+emm)*W)/360)/sw)
            nw2=math.ceil(W/sw)-math.floor((((180+wmm)*W)/360)/sw)
            nw=nw1+nw2
            nh = math.ceil((((90 + nmm) * H) / 180) / sh) - math.floor((((90 + smm) * H) / 180) / sh)
            nn = nw * nh
            np = sw * sh * nn
        else:
            nw=math.ceil((((180+emm)*W)/360)/sw)-math.floor((((180+wmm)*W)/360)/sw)
            nh = math.ceil((((90 + nmm) * H) / 180) / sh) - math.floor((((90 + smm) * H) / 180) / sh)
            nn = nw * nh
            np = sw * sh * nn
        np_list.append(np)
    for u in range(len(np_list)):
        NP=NP+(1/len(np_list))*np_list[u]
    return NP
def factP(sw,sh,a,b,viewpoint_used,W,H):
    np_list=[]
    NP=0
    # 这里也没什么问题
    # xx 为 纬度， yy为经度，
    xx = viewpoint_used[..., 1].reshape( -1)
    yy = viewpoint_used[..., 0].reshape(-1)
    xx = xx.tolist()
    yy = yy.tolist()
    for p in range(len(xx)):
        nmm, smm, wmm, emm, jdcc = S_2D((math.pi*xx[p])/180, (math.pi*yy[p])/180, a, b)
        if 180-yy[p]<0.5*abs(jdcc):
            nw1=math.ceil(W/sw)-math.floor((((180+wmm)*W)/360)/sw)
            nw2=math.ceil((((180+emm)*W)/360)/sw)
            nw=nw1+nw2
            nh=math.ceil((((90+nmm)*H)/180)/sh)-math.floor((((90+smm)*H)/180)/sh)
            nn=nw*nh
            np=sw*sh*nn
        elif yy[p]-(-180)<0.5*abs(jdcc):
            nw1=math.ceil((((180+emm)*W)/360)/sw)
            nw2=math.ceil(W/sw)-math.floor((((180+wmm)*W)/360)/sw)
            nw=nw1+nw2
            nh = math.ceil((((90 + nmm) * H) / 180) / sh) - math.floor((((90 + smm) * H) / 180) / sh)
            nn = nw * nh
            np = sw * sh * nn
        else:
            nw=math.ceil((((180+emm)*W)/360)/sw)-math.floor((((180+wmm)*W)/360)/sw)
            nh = math.ceil((((90 + nmm) * H) / 180) / sh) - math.floor((((90 + smm) * H) / 180) / sh)
            nn = nw * nh
            np = sw * sh * nn
        np_list.append(np)
    for u in range(len(np_list)):
        NP=NP+(1/len(np_list))*np_list[u]
    return NP

def ep_eta_main(hevc_path, viewpoint_used, result_path, W = 3840, H = 1920):
    hevc_name = hevc_path.split('/')[-1]
    
    q = hevc_name.split('_')[0]
    s_w = hevc_name.split('_')[1].split('x')[0]
    s_h = hevc_name.split('_')[1].split('x')[1][:-5]
    EP = factP(int(s_w), int(s_h), 0.5 * math.pi, 0.5 * math.pi, viewpoint_used, W, H)
    write_txt(result_path, [str(EP), str(eta)]) 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--yuvs_path', default='/data/wenxuan/all_yuvs')
    parser.add_argument('--W', type=int, default=3840)
    parser.add_argument('--H', type=int, default=1920)
    parser.add_argument('--viewpoint_root', type=str, default='/home/wenxuan/database/head-tracking-master')
    parser.add_argument('--result_path', type=str, default='/data/wenxuan/GCN_data/EP_dict')
    parser.add_argument('--mp4_root', default='/home/wenxuan/database/4k_30fps')
    args = parser.parse_args()
    W_n = [30,15,12,10,8,6,5,4,3,2]
    H_n = [30,15,12,10,8,6,5,4,3,2]
    W_size =[]
    H_size = []
    for i in range(len(W_n)):
        W_size.append(int(args.W / W_n[i]))
    for j in range(len(H_n)):
        H_size.append(int(args.H / H_n[j]))
    yuvs = os.listdir(args.yuvs_path)
    for yuv in yuvs:
        result_dict = {}
        t, vid_name = yuv[:-4].split('_')
        viewpoint = Get_Viewpoint(args.viewpoint_root, os.path.join(args.mp4_root, vid_name+'.mp4'), vid_name)
        if t == '5to6':
            viewpoint = viewpoint[:, 150:180, :]
        else:
            viewpoint = viewpoint[:, 450:480, :]
        for s_w in W_size:
            for s_h in H_size:
                EP = factP(int(s_w), int(s_h), 0.5 * math.pi, 0.5 * math.pi, viewpoint, args.W, args.H)
                result_dict['%04dx%04d'%(s_w, s_h)] = EP
        np.save(os.path.join(args.result_path, '{}.npy'.format(yuv[:-4])), result_dict)
    # 计算 Ep值并写入字典 
