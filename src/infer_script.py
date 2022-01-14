import os.path

from scipy.io import loadmat, savemat
import numpy as np
import nibabel as nib
import torch
import argparse
import time
import matplotlib.pyplot as plt
from torch.fft import ifftn, ifftshift, fftn, fftshift

from common import set_env
from v8_9_1.chk_utils import phase2in, load_model
from v8_9_1.config import TDK_THRESHOLD, last_model, VERSION, best_model

gyro = 2 * 3.14159265 * 42.58


def infer_Kirby_data(device, epoch, model, loss_func):
    print(' infer_Kirby_data')
    root_fdr = '../../../nifti_data'
    sub = 'Sub002'
    B0, TE = 7, 0.23

    mat = loadmat(f'{root_fdr}/{sub}/new_cosmos.mat')
    susc = mat['susc']
    mask = torch.from_numpy(mat['mask'])

    for ori_idx in range(1, 6):
        print(f'  ori {ori_idx}:', end='')
        phs = nib.load(f'{root_fdr}/{sub}/ori{ori_idx}/{sub}_ori{ori_idx}_phase.nii.gz')
        phs = np.array(phs.dataobj) / (TE * B0 * gyro)

        with open(f'../../../nifti_data/{sub}/ori{ori_idx}/{sub}_ori{ori_idx}.txt') as f:
            s = f.read()
            ori_vec = np.array(s.split('\n'), dtype=np.float32)

        mat_fdr = f'../data/{VERSION}/predict/Kirby'
        img_fdr = f'../data/{VERSION}/images/Kirby'
        mat_file = lambda loss: f'{sub}_ori{ori_idx}_e{epoch}_{loss}.mat'
        img_file = lambda loss: f'{sub}_ori{ori_idx}_e{epoch}_{loss}.png'
        end_time = do_infer(phs, susc, mask, ori_vec, device, model, loss_func, mat_fdr, img_fdr, mat_file, img_file)

    return end_time


def infer_Chan2016(device, epoch, model, loss_func):
    root_data = r'H:\Data\QSM\qsm_2016_challenge'
    phs_all = loadmat(fr'{root_data}\data_12orientations\phs_all.mat')
    R_tot = loadmat(fr'{root_data}\data_12orientations\R_tot.mat')
    chi_cosmos = loadmat(fr'{root_data}\data\chi_cosmos.mat')['chi_cosmos']
    mask = torch.from_numpy(loadmat(fr'{root_data}\data\msk.mat')['msk'])

    ori_idx = 0  # 0~11
    for ori_idx in range(12):
        print(f'  Infer Chan2016 ori: {ori_idx}')

        if ori_idx == -1:
            ori_vec = np.array([0, 0, 1])
            phs = np.average(phs_all['phs_all'], axis=3)
        else:
            ori_vec = R_tot['R_tot'][2, :, ori_idx]
            phs = phs_all['phs_all'][:, :, :, ori_idx]

        mat_fdr = f'../data/{VERSION}/predict/Chan2016'
        img_fdr = f'../data/{VERSION}/images/Chan2016'
        mat_file = lambda loss: f'ori{ori_idx}_e{epoch}_{loss}.mat'
        img_file = lambda loss: f'ori{ori_idx}_e{epoch}_{loss}.png'
        end_time = do_infer(phs, chi_cosmos, mask, ori_vec, device, model, loss_func, mat_fdr, img_fdr, mat_file, img_file)

    return end_time


def infer_MS(device, epoch, model, loss_func):
    mat = loadmat('../../../MoDL_data/MS_data/test_data.mat')
    X = mat['phi'][:, :, :]
    STAR = mat['star_qsm']
    mask = torch.from_numpy(mat['mask'])
    ori_vec = [0., 0., 1.]

    X = X / (1.5 * 1e-3 * 3 * gyro)

    mat_fdr = f'../data/{VERSION}/predict/MS'
    img_fdr = f'../data/{VERSION}/images/MS'
    mat_file = lambda loss: f'e{epoch}_{loss}.mat'
    img_file = lambda loss: f'e{epoch}_{loss}.png'
    end_time = do_infer(X, STAR, mask, ori_vec, device, model, loss_func, mat_fdr, img_fdr, mat_file, img_file)

    return end_time


def infer_Hemorrhage(device, epoch, model, loss_func):
    mat = loadmat('../../../MoDL_data/Hemorrhage/test_data.mat')
    X = mat['phi'][:, :, :]
    STAR = mat['star_qsm']
    mask = torch.from_numpy(mat['mask'])
    ori_vec = [0., 0., 1.]

    X = X / (1.5 * 1e-3 * 3 * gyro)

    mat_fdr = f'../data/{VERSION}/predict/Hemorrhage'
    img_fdr = f'../data/{VERSION}/images/Hemorrhage'
    mat_file = lambda loss: f'e{epoch}_{loss}.mat'
    img_file = lambda loss: f'e{epoch}_{loss}.png'
    end_time = do_infer(X, STAR, mask, ori_vec, device, model, loss_func, mat_fdr, img_fdr, mat_file, img_file)

    return end_time


def do_infer(phs, susc, mask, ori_vec, device, model, loss_func, mat_fdr, img_fdr, mat_file, img_file):
    x_input, y_label, mask = format_data(phs, susc, mask, ori_vec, device)
    y_hat = model(x_input)
    end_time = time.time()

    t_loss, _, _ = loss_func(y_label, y_hat, x_input, TDK_THRESHOLD)
    t_loss = round(t_loss.item(), 4)
    print(f'  loss={t_loss}')

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


def format_data(X, Y, mask, ori_vec, device):
    X = torch.from_numpy(np.expand_dims(X, [0, 1])).to(device)
    Y = torch.from_numpy(np.expand_dims(Y, [0, 1])).to(device)

    y_label = fftshift(fftn(Y))
    y_label = torch.cat([torch.real(y_label), torch.imag(y_label)], dim=1)

    mask = mask.to(device)
    ori_vec = torch.from_numpy(np.expand_dims(ori_vec, [0, 1])).to(device)

    return phase2in(X, ori_vec, TDK_THRESHOLD), y_label, mask


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', type=str, default='cpu')
    # parser.add_argument('-m', '--model', type=str, default=last_model)
    parser.add_argument('-m', '--model', type=str, default=best_model)

    return parser.parse_args()


def main(args):
    print('---------------------------------------------------')
    device = set_env(args.device)

    print('Loading model ...')
    epoch, model, loss_func, _, _ = load_model(device, args.model)
    print('[Done]Model loaded\n')

    print('Inferencing ...')
    start_time = time.time()
    model.eval()
    end_time = infer_Kirby_data(device, epoch, model, loss_func)
    # end_time = infer_Chan2016(device, epoch, model, loss_func)
    # end_time = infer_MS(device, epoch, model, loss_func)
    # end_time = infer_Hemorrhage(device, epoch, model, loss_func)
    print(f'Inferenced, duration: {round(end_time - start_time, 3)}s\n')


if __name__ == '__main__':
    main(get_args())
