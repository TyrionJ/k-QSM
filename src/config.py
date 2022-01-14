VERSION = 'v8_9_1'
DATA_VERSION = 'v8_9_1'

PATCH_SIZE = 64

VERSION_FOLDER = f'../data/{VERSION}'

predict_folder = f'{VERSION_FOLDER}/predict'
test_img_folder = f'{VERSION_FOLDER}/images'

data_folder = f'../data/{DATA_VERSION}'
train_file = f'{data_folder}/train.hdf5'
valid_file = f'{data_folder}/valid.hdf5'
stat_file = f'{data_folder}/stat.mat'

test_folder = f'{data_folder}/test'
test_prefix = f'{test_folder}/test_'

check_folder = f'../checkpoint/{VERSION}'
record_file = f'{check_folder}/record.txt'
best_model = f'{check_folder}/best.pkt'
last_model = f'{check_folder}/last.pkt'
spec_model = f'{check_folder}/last.pkt'

TDK_THRESHOLD = 0.1


import os
if not os.path.exists(VERSION_FOLDER):
    os.mkdir(VERSION_FOLDER)
if not os.path.exists(data_folder):
    os.mkdir(data_folder)
if not os.path.exists(check_folder):
    os.mkdir(check_folder)
if not os.path.exists(predict_folder):
    os.mkdir(predict_folder)
if not os.path.exists(test_img_folder):
    os.mkdir(test_img_folder)
if not os.path.exists(test_folder):
    os.mkdir(test_folder)
