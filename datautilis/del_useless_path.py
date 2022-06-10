import os 
import argparse
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str)
parser.add_argument('--log_path', type = str)
args = parser.parse_args()
model_path = args.model_path
log_path = args.log_path
folders = os.listdir(model_path)
log_folders = os.listdir(log_path)
for folder in folders:
    if folder in log_folders:
        if len(os.listdir(os.path.join(model_path, folder))) == 0:
            print('{} is useless!'.format(folder))
            os.rmdir(os.path.join(model_path, folder))
            # os.rmdir(os.path.join(log_path, folder), )
            shutil.rmtree(os.path.join(log_path, folder))
    else:
        os.rmdir(os.path.join(model_path, folder))
