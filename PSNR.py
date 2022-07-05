"""
this is the module for the Visual Attention-Aware Omnidirectional VideoStreaming Using Optimal Tilesfor Virtual Reality
"""
from re import I
from cv2 import cvtColor
import numpy as np
import math
import time
import argparse
import torch
import os
from scipy.interpolate import interp1d
import cv2
from PIL import Image
from numpy import cos, pi
from PIL import Image
import ipdb
#logging.basicConfig(filename='./ICASSP.log', level=logging.INFO)




def write_txt(txt_root, str_list):
    file_write_obj = open(txt_root, "w")
    for var in str_list:
        file_write_obj.writelines(var)
        file_write_obj.write("\n")
    file_write_obj.close()
    return


def imshow(arry):
    outputImg = Image.fromarray(np.uint8(arry * 255.0))
    outputImg = outputImg.convert("L")
    outputImg.show()
    return


def S_2D(w1, j1, a, b):
    if w1 >= 0.5 * (math.pi - b) and w1 <= 0.5 * math.pi:
        nm = 0.5 * math.pi
        nmm = (180 * nm) / math.pi
    elif w1 >= -(b / 2) and w1 < 0.5 * (math.pi - b):
        nm = w1 + (0.5 * b)
        nmm = (180 * nm) / math.pi
    else:
        aa = -math.sin(w1) - (math.tan(0.5 * b) * math.sin(0.5 * math.pi + w1))
        bb = math.sqrt(
            (math.cos(w1) + math.tan(0.5 * b) * math.cos(0.5 * math.pi + w1)) ** 2
            + (math.tan(0.5 * a)) ** 2
        )
        nm = -math.atan(aa / bb)
        nmm = (180 * nm) / math.pi
    if w1 >= -0.5 * math.pi and w1 <= 0.5 * (b - math.pi):
        sm = -0.5 * math.pi
        smm = (180 * sm) / math.pi
    elif w1 > 0.5 * (b - math.pi) and w1 <= 0.5 * b:
        sm = w1 - 0.5 * b
        smm = (180 * sm) / math.pi
    else:
        aa = math.sin(w1) - (math.tan(0.5 * b) * math.sin(0.5 * math.pi - w1))
        bb = math.sqrt(
            (math.cos(w1) + math.tan(0.5 * b) * math.cos(0.5 * math.pi - w1)) ** 2
            + (math.tan(0.5 * a)) ** 2
        )
        sm = math.atan(aa / bb)
        smm = (180 * sm) / math.pi
    if abs(w1) > 0.5 * (math.pi - b):
        jdcc = 360
        wmm = -180
        emm = 180
    else:
        aa = math.cos(abs(w1)) - (math.tan(0.5 * b) * math.cos(0.5 * math.pi - abs(w1)))
        bb = math.tan(0.5 * a)
        jdc = 2 * (0.5 * math.pi - math.atan(aa / bb))
        jdcc = (180 * jdc) / math.pi
        if (j1 - 0.5 * jdc) < -1 * math.pi:
            wm = (j1 - 0.5 * jdc) + 2 * math.pi
            wmm = (180 * wm) / math.pi
        else:
            wm = j1 - 0.5 * jdc
            wmm = (180 * wm) / math.pi

        if (j1 + 0.5 * jdc) > math.pi:
            em = (j1 + 0.5 * jdc) - 2 * math.pi
            emm = (180 * em) / math.pi
        else:
            em = j1 + 0.5 * jdc
            emm = (180 * em) / math.pi
    return nmm, smm, wmm, emm, jdcc


# w=(0*math.pi)/180
# j=(180*math.pi)/180
# print(S_2D(w,j,math.pi/2,math.pi/2))
def Get_Weight(w, h):
    result = torch.zeros((1, h, w, 1), dtype=torch.float16)
    for j in range(h):
        result[:, j, :, :] = cos((j + 0.5 - h / 2) * pi / h)
    return result


def Get_Mask(H, W, w, j, a=math.pi / 2, b=math.pi / 2):
    result = torch.zeros((H, W), dtype=torch.float32)
    nm, sm, wm, em, jdcc = S_2D(w, j, a, b)
    wm += 180
    em += 180
    nm += 90
    sm += 90
    wmm = math.ceil((wm * W) / 360)
    emm = math.ceil((em * W) / 360)
    nmm = math.ceil(H - ((nm * H) / 180))
    smm = math.ceil(H - ((sm * H) / 180))
    if wm < em:
        result[nmm:smm, wmm:emm] = 1
    else:
        result[nmm:smm, wmm:] = 1
        result[nmm:smm, :emm] = 1
    return result



# imshow(Get_Mask(1800,3600,w,j))
# imshow(Get_Weight(360,180))
def Read_Yuv_Vid(vid_root, H, W,device):  # 返回的是（frameNum,H,W,C）的numpy数组
    file_size = os.path.getsize(vid_root)
    # Number of frames: in YUV420 frame size in bytes is width*height*1.5
    n_frames = file_size // (W * H * 3 // 2)
    # Open 'input.yuv' a binary file.
    f = open(vid_root, "rb")
    result = torch.zeros((n_frames, H, W, 3), dtype=torch.float16,device=device)
    for i in range(n_frames):
        # Read Y, U and V color channels and reshape to height*1.5 x width numpy array
        yuv = np.frombuffer(f.read(W * H * 3 // 2), dtype=np.uint8).reshape(
            (H * 3 // 2, W)
        )
        # Convert YUV420 to BGR (for testing), applies BT.601 "Limited Range" conversion.
        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420).astype(np.float16)
        # print(bgr.shape)
        result[i] = torch.from_numpy(bgr)
    f.close()
    # cv2.destroyAllWindows()
    return result


def Get_Viewpoint(
    viewpoint_root, mp4Vid_root, vidName, clientNum=40
): 
    '''
    viewpoint_root: 具体视频文件夹的path
    mp4Vid_root: 具体mp4的path
    VidName:视频名字
    '''
 # 返回的是（clientNum,frameNum,2）的numpy数组
    if mp4Vid_root.split('/')[-1] =='LetsNotBeAloneTonight.mp4':
        mp4Vid_root= mp4Vid_root.replace('LetsNotBeAloneTonight.mp4', 'Let\'sNotBeAloneTonight.mp4')
    capture = cv2.VideoCapture(mp4Vid_root)
    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    if "'" in vidName:
        print(vidName + " is not right for viewpoint file, change it")
        vidName_list = vidName.split("'")
        vidName = ""
        for j in range(len(vidName_list)):
            vidName += vidName_list[j]
        print("now vidName is {}".format(vidName))
    # print(mp4Vid_root)
    frameNum = int(frame_count)
    result = np.zeros((clientNum, frameNum, 2), dtype=np.float16)
    for h in range(clientNum):  # 一共有40个人
        vpNum = 0
        vx = []  # 用于存储每个视点的纬度
        vy = []  # 用于存储每个视点的经度
        root_viewpoint = viewpoint_root + "/Subject_%d/%s.txt" % (h + 1, vidName)
        with open(root_viewpoint, "r") as f:
            for line in f:
                vpNum += 1
                vy.append(line.split()[0])
                # vx 经度
                vx.append(line.split()[1])
        x = np.linspace(0, frameNum, num=vpNum, endpoint=True)
        vx = np.array(vx)
        vy = np.array(vy)
        fx = interp1d(x, vx)
        fy = interp1d(x, vy)
        xnew = np.linspace(0, frameNum, num=frameNum, endpoint=True)
        # 实际上是先存储经度再存储纬度
        result[h, :, 0] = fx(xnew) * math.pi / 180
        result[h, :, 1] = fy(xnew) * math.pi / 180
    # print(result)
    return result

def get_partial_viewpoint(viewpoint_root, mp4_folder, time_video, clientNum=40):
    t, video_name = time_video.split('_')
    t_begin, t_end = t.split('to')
    t_begin = int(t_begin)
    t_end = int(t_end)
    total_viewpoint = Get_Viewpoint(viewpoint_root, os.path.join(mp4_folder, video_name + '.mp4'), video_name)
    viewpoint = total_viewpoint[:, 30*t_begin:30*t_end, :]
    return viewpoint

def Cal_WSPSNRv_WSMSEv(
    vid1_root, vid2_root, mp4Vid_root, W, H, viewpoint_root, yuvvidName, device
):
    t1 = time.time()
    vid1 = Read_Yuv_Vid(vid1_root, H, W, device)
    vid2 = Read_Yuv_Vid(vid2_root, H, W,device)
    t, vidName = yuvvidName.split("_")
    t2 = time.time()
    print(" yuv time is {}".format(t2 - t1))
    total_viewpoint = Get_Viewpoint(viewpoint_root, mp4Vid_root, vidName)
    if t == "5to6":
        viewpoint = total_viewpoint[:, 150:180, :]
    else:
        viewpoint = total_viewpoint[:, 450:480, :] 
    clientNum, frameNum, vpNum = viewpoint.shape
    weight = Get_Weight(W, H).to(device)
    WS_MSE = 0
    S_MSE = 0
    t3 = time.time()
    mask = torch.zeros((frameNum, H, W, 1), dtype=torch.float16, device=device)
    t4 = time.time()
    print('mask time is {}'.format(t4 - t3))
    for i in range(clientNum):
    #    mask = torch.zeros((frameNum, H, W, 1), dtype=torch.float16).to(device)
        for j in range(frameNum):
            # print('vx,vy',viewpoint[i,j,0],viewpoint[i,j,1])
            t1 = time.time()
            # 这里是先纬度，再经度没啥问题
            mask[j, :, :, 0] = Get_Mask(H, W, viewpoint[i, j, 1], viewpoint[i, j, 0])
        # print(vid1.shape)
        # print(vid2.shape)
        # print(mask.shape)
        # print(weight.shape)
        if torch.sum((mask).float()) == 0:
            print("cl", i)
            break
        # print(torch.sum((mask*weight).float()))
        WS_MSE += (
            torch.sum(((((vid1 - vid2) * mask) ** 2) * weight).float())
            / torch.sum((mask * weight * 3.0).float())
        ) / clientNum
        S_MSE += (
            torch.sum(((((vid1 - vid2) * mask) ** 2)).float())
            / torch.sum((mask * 3.0).float())
        ) / clientNum
        # MSE += (torch.mean((abs(vid1 - vid2)** 2).float())) / clientNum
        # print(WS_MSE.cpu().numpy())
        # print(MSE.cpu().numpy())
        # print(end-start)
        # break
    t5 = time.time()
    print(" cal time is {}".format(t5 - t4))
    WS_PSNR = 10 * np.log10((255.0 ** 2) / WS_MSE.cpu().numpy())
    S_PSNR = 10 * np.log10((255.0 ** 2) / S_MSE.cpu().numpy())

    # PSNR = 10 * np.log10((255.0**2) / MSE.cpu().numpy())
    return WS_PSNR, WS_MSE, S_PSNR, S_MSE

def Cal_WSPSNRv_WSMSEv_vidbased(
    vid1, vid2_root, mp4Vid_root, W, H, viewpoint_root, yuvvidName, device
):
    vid2 = Read_Yuv_Vid(vid2_root, H, W,device)
#    print("yuv time is {}".format(t2 - t1))
    t, vidName = yuvvidName.split("_")

    total_viewpoint = Get_Viewpoint(viewpoint_root, mp4Vid_root, vidName)
    if t == "5to6":
        viewpoint = total_viewpoint[:, 150:180, :]
    else:
        viewpoint = total_viewpoint[:, 450:480, :] 
    clientNum, frameNum, vpNum = viewpoint.shape

    weight = Get_Weight(W, H).to(device)
    WS_MSE = 0
    S_MSE = 0
    mask = torch.zeros((frameNum, H, W, 1), dtype=torch.float16, device=device)
    for i in range(clientNum):
    #    mask = torch.zeros((frameNum, H, W, 1), dtype=torch.float16).to(device)
        for j in range(frameNum):
            # print('vx,vy',viewpoint[i,j,0],viewpoint[i,j,1])
            t1 = time.time()
            mask[j, :, :, 0] = Get_Mask(H, W, viewpoint[i, j, 1], viewpoint[i, j, 0])
        if torch.sum((mask).float()) == 0:
            print("cl", i)
            break
        # print(torch.sum((mask*weight).float()))
        if mask.shape[0] != 30:
            print("ERROR!")
            print("yuvvidName {}".format(yuvvidName))
            print("mask shape is {}".format(mask.shape))
        WS_MSE += (
            torch.sum(((((vid1 - vid2) * mask) ** 2) * weight).float())
            / torch.sum((mask * weight * 3.0).float())
        ) / clientNum
        S_MSE += (
            torch.sum(((((vid1 - vid2) * mask) ** 2)).float())
            / torch.sum((mask * 3.0).float())
        ) / clientNum
        # MSE += (torch.mean((abs(vid1 - vid2)** 2).float())) / clientNum
        # print(WS_MSE.cpu().numpy())
        # print(MSE.cpu().numpy())
        # print(end-start)
        # break
    WS_MSE = torch.tensor(WS_MSE).to(device)
    S_MSE = torch.tensor(S_MSE).to(device)
    WS_PSNR = 10 * np.log10((255.0 ** 2) / WS_MSE.cpu().numpy())
    S_PSNR = 10 * np.log10((255.0 ** 2) / S_MSE.cpu().numpy())

    # PSNR = 10 * np.log10((255.0**2) / MSE.cpu().numpy())
    return WS_PSNR, WS_MSE, S_PSNR, S_MSE

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="manual to this script")
    parser.add_argument("--original_vid_path", type=str, default=None)
    parser.add_argument("--yuv_path", type=str, default=None)
    parser.add_argument("--mp4_path", type=str, default=None)
    parser.add_argument("--W", type=int, default=3840)
    parser.add_argument("--H", type=int, default=1920)
    parser.add_argument("--viewpoint_root", type=str, default=None)
    parser.add_argument("--folder", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--spsnr_path", type=str, default=None)
    args = parser.parse_args()
    WS_PSNR, WS_MSE, S_PSNR, S_MSE = Cal_WSPSNRv_WSMSEv(
        args.original_vid_path,
        args.yuv_path,
        args.mp4_path,
        args.H,
        args.W,
        args.viewpoint_root,
        args.folder,
        args.device,
    )
    # print(WS_PSNR, WS_MSE)

    write_txt(
        args.spsnr_path,
        [
            str(WS_PSNR),
            str(WS_MSE.cpu().numpy()),
            str(S_PSNR),
            str(S_MSE.cpu().numpy()),
        ],
    )
