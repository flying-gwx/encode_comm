import os 
import multiprocessing
import subprocess
import argparse
from def_parser import basic_parser
from multiprocessing import Pool
'''
把crop好的yuv编码为hevc文件
'''
parser = basic_parser()
args = parser.parse_args()
def run_comm(comm):
    subprocess.run(comm, shell=True)

qp = list(range(18, 43))
# 读取特定yuv,进行编码，文件命名:time_name_qp_width_height

yuvs_finish = ['15to16_BTSRun.yuv', '15to16_Symphony.yuv', '15to16_Graffiti.yuv','15to16_WaitingForLove.yuv',
'15to16_Dubai.yuv',         '15to16_Rally.yuv' ,  '15to16_StarWars.yuv'  ,     
'15to16_BlueWorld.yuv',  '15to16_GalaxyOnFire.yuv',  '15to16_Skiing.yuv']

W_n = [30,15,12,10,8,6,5,4,3,2]
H_n = [30,15,12,10,8,6,5,4,3,2]
W_size =[]
H_size = []
for i in range(len(W_n)):
    W_size.append(int(args.WIDTH / W_n[i]))
for j in range(len(H_n)):
    H_size.append(int(args.HEIGHT / H_n[j]))
not_finish_yuvs = ['15to16_A380.yuv', '15to16_AcerPredator.yuv', '15to16_AirShow.yuv', '15to16_Antarctic.yuv', '15to16_BFG.yuv', 
'15to16_CMLauncher.yuv', '15to16_CS.yuv', '15to16_CandyCarnival.yuv', '15to16_Cryogenian.yuv', '15to16_Dota2.yuv', '15to16_DrivingInAlps.yuv', '15to16_Egypt.yuv', '15to16_F5Fighter.yuv', '15to16_Gliding.yuv', '15to16_Help.yuv', '15to16_HondaF1.yuv', '15to16_IRobot.yuv', '15to16_KasabianLive.yuv', '15to16_LOL.yuv', '15to16_LetsNotBeAloneTonight.yuv', '15to16_LoopUniverse.yuv', '15to16_MC.yuv', '15to16_MercedesBenz.yuv', '15to16_Parachuting.yuv', '15to16_Pokemon.yuv', '15to16_Predator.yuv', '15to16_ProjectSoul.yuv', '15to16_RingMan.yuv', '15to16_RioOlympics.yuv', '15to16_RollerCoaster.yuv', '15to16_StarryPolar.yuv', '15to16_SuperMario64.yuv', '15to16_Supercar.yuv', '15to16_Surfing.yuv', '15to16_Terminator.yuv', '15to16_VRBasketball.yuv', '15to16_Waterskiing.yuv', '15to16_WesternSichuan.yuv', '5to6_AcerPredator.yuv',  '5to6_A380.yuv',  '5to6_Help.yuv','5to6_AirShow.yuv', '5to6_Antarctic.yuv', '5to6_BFG.yuv', '5to6_BTSRun.yuv', '5to6_BlueWorld.yuv', '5to6_CMLauncher.yuv', '5to6_CS.yuv', '5to6_CandyCarnival.yuv', '5to6_Cryogenian.yuv', '5to6_Dota2.yuv', '5to6_DrivingInAlps.yuv', '5to6_Dubai.yuv', '5to6_Egypt.yuv', '5to6_F5Fighter.yuv', '5to6_GalaxyOnFire.yuv', '5to6_Gliding.yuv', '5to6_Graffiti.yuv', '5to6_HondaF1.yuv', '5to6_IRobot.yuv', '5to6_KasabianLive.yuv', '5to6_LOL.yuv', '5to6_LetsNotBeAloneTonight.yuv', '5to6_LoopUniverse.yuv', '5to6_MC.yuv', '5to6_MercedesBenz.yuv', '5to6_Parachuting.yuv', '5to6_Pokemon.yuv', '5to6_Predator.yuv', '5to6_ProjectSoul.yuv', '5to6_Rally.yuv', '5to6_RingMan.yuv', '5to6_RioOlympics.yuv', '5to6_RollerCoaster.yuv', '5to6_Skiing.yuv', '5to6_StarWars.yuv', '5to6_StarryPolar.yuv', '5to6_SuperMario64.yuv', '5to6_Supercar.yuv', '5to6_Surfing.yuv', '5to6_Symphony.yuv', '5to6_Terminator.yuv', '5to6_VRBasketball.yuv', '5to6_WaitingForLove.yuv', '5to6_Waterskiing.yuv', '5to6_WesternSichuan.yuv']

# 将crop后的文件删除

videos = ['StarryPolar', 'F5Fighter','BTSRun']
# 先测试一个视频试试 AirShow
not_done = []
time_stamp = ['06to07', '07to08', '08to09', '09to10', '10to11']
for video in videos:
    for time in time_stamp:
        not_done.append(time + '_' + video + '.yuv')

for yuv in not_done:

    comm_list = []
    for w in W_size:
        for h in H_size:
            for tmp_qp in qp:
                    # hvec 文件夹命名：
                    # 记录每个hevc的size
                    #TODO: 记录每个hevc的PSNR值
                    if not os.path.exists(os.path.join(args.encode_folder,  yuv[:-4],'%04d_%04dx%04d'%(tmp_qp,w,h))):
                        os.mkdir(os.path.join(args.encode_folder,  yuv[:-4],'%04d_%04dx%04d'%(tmp_qp,w,h)))
                    tile_names = os.listdir(os.path.join(args.croped_yuv_folder,  yuv[:-4], '%04dx%04d'%(w,h)))
                    tile_names.sort()
                    for tile in tile_names:
                        '''
                        我把nopsnr选项去掉了
                        '''
                        comm = ('kvazaar  -i %s  --input-res %dx%d --frames 30  --no-psnr --no-info '  
                        ' --qp %d --input-fps 30 -o %s'%(os.path.join(os.path.join(args.croped_yuv_folder,  yuv[:-4], '%04dx%04d'%(w,h), tile)), w,h, tmp_qp, os.path.join(args.encode_folder,  yuv[:-4],'%04d_%04dx%04d'%(tmp_qp,w,h), tile[:-4]+'.hevc')))
                        comm_list.append(comm)
    # 开多线程编码
    with Pool(args.cpu_num) as p:
        p.map(run_comm, comm_list)
    print("{} done!".format(yuv))
    subprocess.run('rm -r %s'%(os.path.join(args.croped_yuv_folder,  yuv[:-4])),shell= True)
    print("{} deleted!".format(os.path.join(args.croped_yuv_folder,  yuv[:-4])))

    

