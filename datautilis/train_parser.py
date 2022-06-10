import  argparse

def Get_basic_trainparser():
     parser = argparse.ArgumentParser(description='ICASSP train parser')
#      flags.DEFINE_integer('resize_W', 2048, 'Resize img.')
# flags.DEFINE_integer('resize_H', 1024, 'Resize img.')
# 不进行resize
     parser.add_argument('--resize_W', default=2048, type=int)
     parser.add_argument('--resize_H', default=1024, type=int)
     parser.add_argument('--image_folder', default='/home/lhr1/database/GCN_data/all_pngs')
     parser.add_argument('--csv_folder', default='/home/lhr1/database/GCN_data/all_csv/origin_csv')
     parser.add_argument('--project_datapath', default='/data/wenxuan/GCN_data/project_gray_images')
     parser.add_argument('--size_dict_folder', default='/data/wenxuan/GCN_data/size_dict_folder' )  
     parser.add_argument('--sobel_value_path', default='/data/wenxuan/GCN_data/sobel_value')
     parser.add_argument('--split_eta_dataset', default='/data/wenxuan/GCN_data/coding_efficiency')
     parser.add_argument('--epochs', default = 50, type = int)
     parser.add_argument('--batch_size', default = 64, type = int)
     parser.add_argument('--gpu', default='0', type = str)
     parser.add_argument('--dropout', default=1.0, type = float)
     parser.add_argument('--learning_rate', default=1e-5, type = float)
     parser.add_argument('--remuse', type = bool, default = False)
     return parser