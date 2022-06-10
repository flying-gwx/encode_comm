from hashlib import sha1
import subprocess
import os 


# 5to6_Dubai.yuv数据有问题，不要用
yuvs_path = '/data/wenxuan/yuvs/BlueWorld'
pngs_path = '/data/wenxuan/GCN_data/all_pngs'

yuvs = os.listdir(yuvs_path)
yuvs.sort()
for yuv in yuvs:
    tmp_png_path = os.path.join(pngs_path, yuv[:-4])
    if not os.path.exists(tmp_png_path):
        os.mkdir(tmp_png_path)
    comm = 'ffmpeg -pix_fmt yuv420p -s 3840x1920  -f rawvideo -i {} -vframes 30  {}/%2d.png'.format(os.path.join(yuvs_path, yuv),tmp_png_path)
    subprocess.run(comm, shell=True)
    print("{} done!".format(yuv))