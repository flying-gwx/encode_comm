import os
import csv
import argparse
import pandas as pd
from numpy import append

def get_base_parser():
    parser = argparse.ArgumentParser(description='ICASSP parser')
    parser.add_argument('--txts_path', type=str)
    parser.add_argument('--result_path', type=str)
    return parser


def read_txt(txt_path):
    with open(txt_path, 'r') as f:
        list_item = f.read().splitlines()
    return list_item
        

qp=[]
s_w=[]
s_h=[]
SPSNR=[]
WSPSNR=[]
eta=[]
ep=[]
parser = get_base_parser()
args = parser.parse_args()
all_txts = os.listdir(args.txts_path)
# sort保证不同机器读到的all_txts是一样的
all_txts.sort()
assert len(all_txts) == 4590 * 2 
csv_name = args.txts_path.split('/')[-1] + '.csv'
all_txts_SPSNR = []
all_txts_ep_eta = []


for txt in all_txts:
    if txt[-7:] == 'eta.txt':
        all_txts_ep_eta.append(txt)
    else :
        all_txts_SPSNR.append(txt)
assert (len(all_txts_ep_eta) == len(all_txts_SPSNR))

for i in range(len(all_txts_ep_eta)):
    parameter = all_txts_ep_eta[i].split('_')
    qp.append(parameter[0])
    s_w.append(parameter[1].split('x')[0])
    s_h.append(parameter[1].split('x')[1])
    # read ep and eta
    ep_eta = read_txt(os.path.join(args.txts_path, all_txts_ep_eta[i]))
    ep.append(ep_eta[0])
    eta.append(ep_eta[1])
    SPSNR_txt_path = parameter[0] + '_' + parameter[1] + '_'+'wspsnr_spsnr.txt'
    WSPSNR_SPSNR = read_txt(os.path.join(args.txts_path, SPSNR_txt_path))
    SPSNR.append(WSPSNR_SPSNR[2])
    WSPSNR.append(WSPSNR_SPSNR[0])
if os.path.exists(os.path.join(args.result_path, csv_name)):
    print('{} exists! delete and rewrite'.format(csv_name))
    os.remove(os.path.join(args.result_path, csv_name))

with open(os.path.join(args.result_path, csv_name),'a+', newline='', encoding='utf-8') as f:
    csv_write = csv.writer(f)
    csv_head = ["Qstep", "S_w", "S_h", "S-PSNR","WS-PSNR", "ep_trans_eff", "eta_coding_eff"]
    csv_write.writerow(csv_head)
    for i in range(len(qp)):
        data_row = [qp[i], s_w[i], s_h[i], SPSNR[i], WSPSNR[i], ep[i], eta[i]]
        csv_write.writerow(data_row)

print("{} done!".format(csv_name))


    
