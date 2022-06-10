import pandas as pd
import os

def train_test_split(csv_path,csv_name, result_path, random_state, train_frac):
    '''
    csv_path: path  of origin csv file
    result_path: where you want to save 
    train_frac: the fraction of train dataset, 0~0.1
    '''
    csv = pd.read_csv(os.path.join(csv_path, csv_name))
    train = csv.sample(frac=train_frac,random_state=random_state)
    test = csv[~csv.index.isin(train.index)]
    train.to_csv(os.path.join(result_path, csv_name[:-4] + '_train.csv'), index = False)
    test.to_csv(os.path.join(result_path, csv_name[:-4] + '_test.csv'), index = False)
    print("{} {} is splited!".format(csv_path, csv_name))

path = '/home/lhr1/database/GCN_data/all_csv/origin_csv'

csvs = os.listdir(path)
for csv in csvs:
    train_test_split(path, csv, '/home/lhr1//database/GCN_data/all_csv/train_test_csv', 0, 0.7)