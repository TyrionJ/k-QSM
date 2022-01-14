from typing import Callable

PATCH_SIZE = 64

DATA_FOLDER = f'../data'

predict_folder = f'{DATA_FOLDER}/predict'
test_img_folder = f'{DATA_FOLDER}/images'

train_file: Callable[[float], str] = lambda thr: f'{DATA_FOLDER}/train-thr{thr}.hdf5'
valid_file: Callable[[float], str] = lambda thr: f'{DATA_FOLDER}/valid-thr{thr}.hdf5'

check_folder = f'../checkpoint'
record_file: Callable[[float], str] = lambda thr: f'{check_folder}/{thr}/record.txt'
best_model: Callable[[float], str] = lambda thr: f'{check_folder}/{thr}/best.pkt'
last_model: Callable[[float], str] = lambda thr: f'{check_folder}/{thr}/last.pkt'

THR_01 = 0.1
THR_02 = 0.2


def make_fdr():
    import os
    if not os.path.exists(DATA_FOLDER):
        os.mkdir(DATA_FOLDER)
    if not os.path.exists(check_folder):
        os.mkdir(check_folder)
    if not os.path.exists(predict_folder):
        os.mkdir(predict_folder)
    if not os.path.exists(test_img_folder):
        os.mkdir(test_img_folder)
    if not os.path.exists(f'{check_folder}/{THR_01}'):
        os.mkdir(f'{check_folder}/{THR_01}')
    if not os.path.exists(f'{check_folder}/{THR_01}'):
        os.mkdir(f'{check_folder}/{THR_02}')


make_fdr()
