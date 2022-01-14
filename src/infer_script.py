import os.path
from typing import Callable

from scipy.io import loadmat, savemat
import numpy as np
import nibabel as nib
import torch
import argparse
import time
import matplotlib.pyplot as plt
from torch.fft import ifftn, ifftshift, fftn, fftshift

from utils.common import set_env
from utils.proj_utils import phase2in, load_model
from config import last_model, THR_01, THR_02

gyro = 2 * 3.14159265 * 42.58


def infer_Kirby_data(device, epoch, model, loss_func, thr):
    print(' Kirby_data')
    end_time = time.time()
    root_fdr = '../data/Kirby_data'
    sub = 'Sub001'
    B0, TE = 7, 0.23

    mat = loadmat(f'{root_fdr}/{sub}/cosmos.mat')
    susc = mat['susc']
    mask = torch.from_numpy(mat['mask'])

    oris = [i for i in os.listdir(f'{root_fdr}/{sub}') if 'ori' in i]

    for i, ori in enumerate(oris):
        print(f'  {ori}:', end='')
        phs = nib.load(f'{root_fdr}/{sub}/{ori}/{sub}_{ori}_phase.nii.gz')
        phs = np.array(phs.dataobj) / (TE * B0 * gyro)

        with open(f'{root_fdr}/{sub}/{ori}/{sub}_{ori}.txt') as f:
            s = f.read()
            ori_vec = np.array(s.split('\n'), dtype=np.float32)

        mat_fdr = f'../data/predict/Kirby'
        img_fdr = f'../data/images/Kirby'
        mat_file: Callable[[float], str] = lambda loss: f'{sub}_{ori}_t{thr}_e{epoch}_{loss}.mat'
        img_file: Callable[[float], str] = lambda loss: f'{sub}_{ori}_t{thr}_e{epoch}_{loss}.png'
        end_time = do_infer(phs, susc, mask, ori_vec, device, model,
                            loss_func, thr, mat_fdr, img_fdr, mat_file, img_file)

    return end_time


def do_infer(phs, susc, mask, ori_vec, device, model, loss_func, thr, mat_fdr, img_fdr, mat_file, img_file):
    x_input, y_label, mask = format_data(phs, susc, mask, ori_vec, device, thr)
    y_hat = model(x_input)
    end_time = time.time()

    t_loss, _, _ = loss_func(y_label, y_hat, x_input, thr)
    t_loss = round(t_loss.item(), 4)
    print(f' loss={t_loss}')

    y_hat = to_susc(y_hat, mask)

    k_tkd = torch.complex(x_input[0, 0], x_input[0, 1])
    tkd = ifftn(ifftshift(k_tkd))
    tkd = torch.sign(torch.real(tkd)) * torch.abs(tkd) * mask

    if not os.path.exists(mat_fdr):
        os.mkdir(mat_fdr)
    savemat(f'{mat_fdr}/{mat_file(t_loss)}', mdict={'kQSM': y_hat[0].detach().numpy(),
                                                    'label': susc, 'mask': mask.numpy(),
                                                    'TKD': tkd.detach().numpy()})
    if not os.path.exists(img_fdr):
        os.mkdir(img_fdr)
    print_img([tkd.detach().numpy(), y_hat[0].detach().numpy(), susc],
              ['TKD', 'k-QSM', 'Label'],
              f'{img_fdr}/{img_file(t_loss)}')

    return end_time


def to_susc(y_hat, mask):
    y_hat = torch.complex(y_hat[:, 0, :, :, :], y_hat[:, 1, :, :, :])
    y_hat = torch.fft.ifftn(torch.fft.ifftshift(y_hat))
    return torch.abs(y_hat) * torch.sign(torch.real(y_hat)) * mask


def print_img(data_arr: list, title_arr: list, to_file: str):
    row, col = 3, len(data_arr)
    x, y, z = np.array(data_arr[0].shape) // 2

    fig = plt.figure(1)
    for i in range(col):
        plt.subplot(row, col, 1 + i)
        plt.xticks([]), plt.yticks([]), plt.ylabel('Axial')
        plt.title(title_arr[i])
        plt.imshow(data_arr[i][:, :, z].T, vmax=0.1, vmin=-0.1, cmap='gray')

        plt.subplot(row, col, 1 + i + col)
        plt.xticks([]), plt.yticks([]), plt.ylabel('Coronal')
        plt.imshow(np.flipud(data_arr[i][:, y, :].T), vmax=0.1, vmin=-0.1, cmap='gray')

        plt.subplot(row, col, 1 + i + 2 * col)
        plt.xticks([]), plt.yticks([]), plt.ylabel('Sagittal')
        plt.imshow(np.flipud(data_arr[i][x, :, :].T), vmax=0.1, vmin=-0.1, cmap='gray')

    plt.savefig(to_file)
    plt.close(fig)


def format_data(X, Y, mask, ori_vec, device, thr):
    X = torch.from_numpy(np.expand_dims(X, [0, 1])).to(device)
    Y = torch.from_numpy(np.expand_dims(Y, [0, 1])).to(device)

    y_label = fftshift(fftn(Y))
    y_label = torch.cat([torch.real(y_label), torch.imag(y_label)], dim=1)

    mask = mask.to(device)
    ori_vec = torch.from_numpy(np.expand_dims(ori_vec, [0, 1])).to(device)

    return phase2in(X, ori_vec, thr), y_label, mask


def main(args):
    print('---------------------------------------------------')
    device = set_env(args.device)

    print('Loading model ...')
    epoch, model, loss_func, _, _ = load_model(device, args.model(args.threshold))
    print('[Done]Model loaded\n')

    print('Inferencing ...')
    start_time = time.time()
    model.eval()
    end_time = infer_Kirby_data(device, epoch, model, loss_func, args.threshold)
    print(f'Inferenced, duration: {round(end_time - start_time, 3)}s\n')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', type=str, default='cpu')
    parser.add_argument('-m', '--model', type=str, default=last_model)
    parser.add_argument('-t', '--threshold', type=float, default=THR_01, choices=[THR_01, THR_02])

    return parser.parse_args()


if __name__ == '__main__':
    main(get_args())
