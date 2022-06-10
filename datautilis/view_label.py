from logging import handlers
from matplotlib import projections
import numpy as np
import os
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
W_size=[256,320,384,448,512,640,768,960,1280,1920]
H_size=[64,128,192,256,320,384,448,640,960]

def view_sample(predictions,video_names, labels, title, index,mode, save_path):
    '''
    取出两个数据集，对其中low和high的部分分别进行可视化
    '''
    for video in video_names:
        prediction = predictions[video][index]
        label = labels[video][index]
        view_two_numpy(label, prediction, x_y_labels=['Coding Version ID', 'bits/pixel'], title = title, save_name=os.path.altsepjoin(save_path, '{}_{}.png'.format(mode,video)))

def view_box(data, title, save_name):
    '''
    画box图，
    '''
    len = data.shape[1]
    position = list(range(2, 2*len + 1, 2))
    fig, ax = plt.subplots()
    ax.boxplot(data, positions=position, widths = 1.5, patch_artist=True,showmeans = False,
    showfliers=False,
    medianprops={"color": "white", "linewidth": 0.5},
                boxprops={"facecolor": "C0", "edgecolor": "white",
                          "linewidth": 0.5},
                whiskerprops={"color": "C0", "linewidth": 1.5},
                capprops={"color": "C0", "linewidth": 1.5})
    plt.title(title)
    plt.show()
    plt.savefig(save_name)
    plt.close()

def viewhotmap(input_dict, data,  title, saved_name):
    '''
    将data以热力图的形式展现出来
    input_dict为x轴和y轴
    input_dict = {'s_w' 's_h', 'qp'}
    '''
    qp_max = 51
    keys = list(input_dict.keys())
    # 反归一化
    s_w = (input_dict['s_w'] * max(W_size)).astype(np.int)
    s_h = (input_dict['s_h'] * max(H_size)).astype(np.int)
    if 'qp' in keys:
        qp = (input_dict['qp'] * qp_max).astype(np.int)
        y_len = len(W_size) * len(H_size)
        qp_unique = list(np.unique(qp))
        qp_unique.sort()
        x_len = len(qp_unique)
        result = np.zeros((x_len, y_len))
        y_list = []
        for w in W_size:
            for h in H_size:
                y_list.append((w,h))
        
        for i in range(data.shape[0]):
            x = qp_unique.index(qp[i])
            y = y_list.index((s_w[i], s_h[i]))
            result[x_len - x - 1, y] = data[i]
        fig = plt.figure()
        ax = fig.add_subplot(111)
       # ax.set_xlabel('')
        ax.set_yticks(range(x_len))
        ax.set_yticklabels(qp_unique)
        ax.set_xticks(range(y_len))
        ax.set_xticklabels(y_list)
        im = ax.imshow(result)
        plt.colorbar(im)
        plt.title(title)
        plt.savefig(saved_name)
        plt.close()
    else:
        y_len = len(W_size)
        x_len = len(H_size)
    
    




def view_two_numpy(true_value, prediction_value, x_y_labels, title, save_name):
    '''
    save_name: 保存的名字
    title：图片title
    '''
    plt.plot(true_value, color = 'red', label = 'true_value' )
    plt.plot(prediction_value, color = 'blue', label = 'predictions_value')
    plt.xlabel(x_y_labels[0])
    plt.ylabel(x_y_labels[1])
    plt.legend()
    plt.title(title)
    plt.savefig(save_name)
    plt.close()

def view_error(error, title, save_name):
    plt.plot(error)
    plt.xlabel('Coding Version ID')
    plt.title(title)
    plt.savefig(save_name)
    plt.close()

def view_pie(percentage_dict, title, save_name):
    '''
    percentage_dict 中percentages 为 百分数*100
    '''
    names = percentage_dict.keys()
    percentage = percentage_dict.values()
    _,texts, autotexts = plt.pie(x = percentage, labels=names,autopct="%.2f%%") 
    plt.title(title)
    plt.savefig(save_name)
    plt.close()

def view_histogram(histogram, title, save_name):
    plt.plot(histogram)
    plt.title(title)
    plt.savefig(save_name)
    plt.close()

# def view_loss(train_loss, test_loss, save_path):
#     '''
#     输入为loss,设置可视化范围，如果差别过大，只取容易可视化的
#     '''
#     min_train_loss = min(train_loss)

def get_percentage_num(train_percentage_error):
    total_train_size = train_percentage_error.size
    num_5 = np.sum(train_percentage_error < 0.05)
    num_5_10 = np.sum((train_percentage_error > 0.05) & (train_percentage_error < 0.1))
    num_10_15 = np.sum((train_percentage_error > 0.1) & (train_percentage_error < 0.15))
    num_15_up = np.sum(train_percentage_error > 0.15)
    train_percentage_dict = {'<5%': 100*num_5/total_train_size, '5%-10%':num_5_10*100/total_train_size, '10%-15%': num_10_15 * 100/total_train_size, '>15%': num_15_up *100/total_train_size }
    return train_percentage_dict

def view_result(train_predictions, train_labels, saved_path, mode = 'train'):
    '''
    可视化模块
    mean percentage error
    mse error
    predictions and labels
    max error
    '''
    if not os.path.exists(os.path.join(saved_path, mode)):
        os.mkdir(os.path.join(saved_path, mode))
        print('{} is created!'.format(os.path.join(saved_path, mode)))
    if not os.path.exists(os.path.join(saved_path, mode, 'errors')):
        os.mkdir(os.path.join(saved_path, mode, 'errors'))
    if not os.path.exists(os.path.join(saved_path, mode, 'predictions')):
        os.mkdir(os.path.join(saved_path, mode, 'predictions'))
        
        
    max_error = 0
    max_key = None
    csv_number = train_labels[list(train_labels.keys())[0]].shape[0]
    train_percentage_error = np.zeros((csv_number, len(list(train_predictions.keys()))))
    train_absolute_error = np.zeros((csv_number, 1))
    i = 0
    train_prediction = np.zeros((csv_number, 1))
    train_true_value = np.zeros((csv_number, 1))
    for key in train_predictions.keys():
        train_prediction += train_predictions[key]
        train_true_value += train_labels[key]
        # mse error
        view_two_numpy(train_labels[key], train_predictions[key], x_y_labels=[ 'Coding Version ID', 'bits/pixel'], title = '{} {}'.format(mode, key), 
        save_name=os.path.join(saved_path, mode, 'predictions', '{}_prediction.png'.format(key)))
        absolute_error = np.abs(train_labels[key] - train_predictions[key])
        view_error(absolute_error, title = '{} absolute error'.format(key), save_name = os.path.join(saved_path, mode, 'errors', '{}_error.png'.format(key)))
        error = np.mean(absolute_error)
        train_absolute_error += absolute_error
        tmp = np.abs(train_predictions[key] - train_labels[key])/train_labels[key]
        train_percentage_error[:,i] = np.squeeze(tmp, axis = 1)
        i += 1 
        if error > max_error:
            max_key = key
    print("{} max error key is {}".format(mode, max_key))
    view_error(np.abs(train_labels[max_key] - train_predictions[max_key]), title = '{} max absolute error : {}'.format(mode, max_key), save_name =  os.path.join(saved_path,mode, '{}_max_mse_error.png'.format(mode)))

    train_absolute_error = train_absolute_error/len(list(train_predictions.keys()))
    view_error(train_absolute_error, title='{} mean absolute error'.format(mode), save_name = os.path.join(saved_path, mode, '{}_mean_absolute_error.png'.format(mode)))

    mean_percentage_error = np.mean(train_percentage_error, axis = 1)
    view_error(mean_percentage_error, title = '{} mean percentage error'.format(mode), save_name = os.path.join(saved_path, mode, '{}_mean_percentage_error.png'.format(mode)) )
    
    train_percentage_dict = get_percentage_num(train_percentage_error)
    view_pie(train_percentage_dict, title = '{}_percentage_pie'.format(mode), save_name = os.path.join(saved_path, mode, '{}_percentage_pie.png'.format(mode)))
    train_true_value = train_true_value/len(list(train_predictions.keys()))
    train_prediction = train_prediction/len(list(train_predictions.keys()))
    view_two_numpy( train_true_value,train_prediction, x_y_labels=[ 'Coding Version ID', 'bits/pixel'], title='average {} dataset'.format(mode), save_name=os.path.join(saved_path, mode, '{}_mean_predictions.png'.format(mode)))

