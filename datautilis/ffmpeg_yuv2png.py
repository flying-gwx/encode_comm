import subprocess
import os 


# 5to6_Dubai.yuv数据有问题，不要用
yuvs_folder = '/data/wenxuan/yuvs'
videos = os.listdir(yuvs_folder)
pngs_path = '/data/wenxuan/GCN_data/all_pngs'

for video in videos:
    yuvs = os.listdir(os.path.join(yuvs_folder, video))
    yuvs.sort()
    for yuv in yuvs:
        if yuv[-4:] == '.yuv':
            tmp_png_path = os.path.join(pngs_path, yuv[:-4])
            if not os.path.exists(tmp_png_path):
                os.mkdir(tmp_png_path)
            comm = 'ffmpeg -pix_fmt yuv420p -s 3840x1920  -f rawvideo -i {} -vframes 30  {}/%2d.png'.format(os.path.join(yuvs_folder, video, yuv),tmp_png_path)
            subprocess.run(comm, shell=True, stdout=subprocess.DEVNULL, check=True)
            print("{} done!".format(yuv))