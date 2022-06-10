import  argparse
from cgi import test
import re
import os
import matplotlib.pyplot as plt
import numpy as np
 
def Get_parser():
    parser = argparse.ArgumentParser(description='loss visualization')
    parser.add_argument('--txt_path', type = str)
    parser.add_argument('--begin', type = int, default= 0)
    return parser

def read_txt(txt_name):
    loss = []
    with open(txt_name, mode='r') as f:
        lines = f.readlines()
    for i in lines:
        loss.append(float(re.findall(r"\d+\.?\d*",i)[0]))
    return loss


parser = Get_parser()
args = parser.parse_args()
txts = os.listdir(args.txt_path)
train_txt = ''
test_txt = ''

for i in txts:
    if i[-9:-4] == 'train':
        train_txt = i
    elif i[-8:-4] == 'test':
        test_txt = i

train_loss = read_txt(os.path.join(args.txt_path,train_txt))
test_loss = read_txt(os.path.join(args.txt_path,test_txt))
train_loss = np.array(train_loss)
test_loss = np.array(test_loss)



plt.plot(np.arange(args.begin, train_loss.size, 1), train_loss[args.begin:])
plt.plot(np.arange(args.begin, train_loss.size, 1), test_loss[args.begin:])
plt.legend(['train_loss', 'test_loss'])
plt.savefig(os.path.join(args.txt_path, 'train_test_loss.png'))
plt.close()
