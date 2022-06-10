import numpy as np
import matplotlib.pyplot as plt
import os 
import scipy.io
from train_parser import Get_basic_trainparser
from view_label import *
import sys
sys.path.append('../')
from datahelper import Datahelper, ALL_VIDEOS

def check_graph_data():
    gray_project_path = '/home/lhr1/database/GCN_data/project_gray_images'
    mats = os.listdir(gray_project_path)
    for mat in mats:
        graph_point = scipy.io.loadmat("{}/{}".format(gray_project_path, mat))[
            "value"
        ]
        num_zeros = []
        for i in range(graph_point.shape[0]):
            num_zeros.append(np.sum(graph_point[i, ...] == 0))
        print( ' the std value of zero in video{} is {}'.format(mat[:-4],np.std(np.array(num_zeros))))


if __name__ == '__main__':
    parser = Get_basic_trainparser()
    args = parser.parse_args()
    check_graph_data()