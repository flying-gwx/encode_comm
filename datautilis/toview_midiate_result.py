
import numpy as np
import matplotlib.pyplot as plt
import os 
from train_parser import Get_basic_trainparser
from view_label import *
import sys
sys.path.append('..')
from datahelper import Datahelper, ALL_VIDEOS


def view_moment():
    dict_path = '/home/lhr1/database/GCN_data/npy_dict/video_sobel_mean_std_third_momentum.npy'
    dict_sobel_mean_std_third = np.load(dict_path, allow_pickle= True).item()
    dict_median = np.load('/home/lhr1/database/GCN_data/npy_dict/sobel_median.npy', allow_pickle= True).item()
    all_data = {}
    for key, value in dict_sobel_mean_std_third.items():

        tmp = np.array( dict_median[key])
        all_data[key] = np.append(tmp, value) 
    np.save('/home/lhr1/database/GCN_data/npy_dict/sobel_median_mean_std_third.npy', all_data)
   # 按照ALL_VIDEOS的顺序进行可视化
    parser = Get_basic_trainparser()
    args = parser.parse_args()
    visio_result = []
    labels_list = []
    median_result_npy = []
    for video in ALL_VIDEOS:
        dataset = Datahelper(args, video)
        _, labels = dataset.Return_origin_complexity_TrainDataset(tile_able = False, qp_mode='all')
        labels_list.append(np.mean(labels))
        visio_result.append(dict_sobel_mean_std_third[video])
        median_result_npy.append(dict_median[video])

    visio_result_npy = np.stack(visio_result, axis = 0)
    median_result_npy  = np.array(median_result_npy)
    max_sobel = np.max(visio_result_npy[:, 0])
    max_std = np.max(visio_result_npy[:, 1]) 
    max_third_moment = np.max(visio_result_npy[:, 2])  
    print("max value of sobel_mean, median, std_sobel, third_moment are{} {} {} {}".format(max_sobel, np.max(median_result_npy), max_std, max_third_moment))
    plt.plot(visio_result_npy[:, 0]/max_sobel, label = 'mean value')
    plt.plot(median_result_npy / np.max(median_result_npy), label = 'median')
    plt.plot(labels_list/max(labels_list), label = 'video label')
    plt.legend()
    plt.savefig('../../visualization/input_data_visualization/visio_mean_median.png')
    # plt.plot(visio_result_npy[:, 1]/max_std, label = 'std value')
    # plt.plot(visio_result_npy[:, 2]/max_third_moment, label = 'third moment value')
    # plt.plot(labels_list/max(labels_list), label = 'video label')
    # plt.xlabel('coding version')
    # plt.legend()
    # plt.savefig('../../visualization/input_data_visualization/visio_stastics.png')





def view_graphs():
    dict_path = '/home/lhr1/database/GCN_data/npy_dict/graph_dict.npy'
    dict = np.load(dict_path, allow_pickle= True).item()
    parser = Get_basic_trainparser()
    args = parser.parse_args()
    graph_mean = []
    
    labels_list = []
    for video in ALL_VIDEOS:
        dataset = Datahelper(args, video)
        _, labels = dataset.Return_origin_complexity_TrainDataset(tile_able = False, qp_mode='all')
        labels_list.append(np.mean(labels))
        graph_mean.append(dict[video][0])

    plt.plot(graph_mean, label = 'graphs')
    plt.plot(labels_list)



if __name__ == "__main__":
    parser = Get_basic_trainparser()
    args = parser.parse_args()
    dataset = Datahelper(args, '5to6_AcerPredator')
    dataset.return_position_dataset()
    view_moment()
    
