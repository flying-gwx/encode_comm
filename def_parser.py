import argparse
def basic_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ffmpeg_path', type=str, default =r'/home/wenxuan/ffmpeg-4.4.2/ffmpeg' )
    parser.add_argument('--croped_yuv_folder', type = str, default= r'/data/wenxuan/croped_yuvs')
    parser.add_argument('--input_yuv_folder', type = str, default= r'/data/wenxuan/all_yuvs')
    parser.add_argument('--encode_folder', type = str, default = '/data/wenxuan/encode')
    parser.add_argument('--HEIGHT', type=int, default = 1920)
    parser.add_argument('--WIDTH', type=int, default=3840)
    parser.add_argument('--cpu_num', type = int, default=10)
    parser.add_argument('--csv_folder', type = str, default='/data/wenxuan/origin_csv')
    parser.add_argument('--size_dict_folder', type = str, default='/data/wenxuan/GCN_data/size_dict_folder')
    parser.add_argument('--PSNR_path', default = '/data/wenxuan/PSNR')
    parser.add_argument('--mp4_30fps_folder', default='/data/wenxuan/4k_30fps')
    return parser
W_n = [30,15,12,10,8,6,5,4,3,2]
H_n = [30,15,12,10,8,6,5,4,3,2]
W_size =[]
H_size = []
#ALL_VIDEOS = ['Surfing', 'Waterskiing', 'AirShow', 'F5Fighter', 'StarryPolar', 'BlueWorld', 'BTSRun',  'WaitingForLove',  'LOL']
ALL_VIDEOS = ['StarryPolar', 'F5Fighter','BTSRun']
time_stamp = ['06to07', '07to08', '08to09', '09to10', '10to11']
for i in range(len(W_n)):
    W_size.append(int(3840 / W_n[i]))
for j in range(len(H_n)):
    H_size.append(int(1920 / H_n[j]))