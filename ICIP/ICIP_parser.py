import argparse
def ICIP_result_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ffmpeg_path', type=str, default =r'/home/wenxuan/ffmpeg-4.4.2/ffmpeg' )
    parser.add_argument('--croped_yuv_folder', type = str, default= r'/data/wenxuan/croped_yuvs')
    parser.add_argument('--input_yuv_folder', type = str, default= r'/data/wenxuan/yuvs/AirShow')
    parser.add_argument('--yuv_name', type = str, default='AirShow')
    parser.add_argument('--encode_folder', type = str, default = '/data/wenxuan/encode')
    parser.add_argument('--HEIGHT', type=int, default = 1920)
    parser.add_argument('--WIDTH', type=int, default=3840)
    parser.add_argument('--cpu_num', type = int, default=10)
    parser.add_argument('--csv_folder', type = str, default='/data/wenxuan/origin_csv')
    parser.add_argument('--size_dict_folder', type = str, default='/data/wenxuan/GCN_data/size_dict_folder')
    parser.add_argument('--PSNR_path', default = '/data/wenxuan/PSNR')
    parser.add_argument('--mp4_30fps_folder', default='/data/wenxuan/4k_30fps')
    parser.add_argument('--video', 'AirShow')
    parser.add_argument('--device', 'cuda:0')
    return parser

#ALL_VIDEOS = ['Surfing', 'Waterskiing', 'AirShow', 'F5Fighter', 'StarryPolar', 'BlueWorld', 'BTSRun',  'WaitingForLove',  'LOL']
ALL_VIDEOS = ['StarryPolar', 'F5Fighter','BTSRun']
time_stamp = ['5to6', '06to07', '07to08', '08to09', '09to10', '10to11']
# 为了和kvazaar相适应，设置单位为bps
BANDWIDTH = [5e5, 1e6, 2e6, 4e6,  5e6]
u = []
v = []