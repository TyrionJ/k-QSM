import os
from typing import Callable

import h5py
import torch
from torch.fft import fftn, fftshift
from torch import from_numpy as to_torch
import numpy as np
from scipy.io import loadmat
from dipy.io.image import load_nifti_data

from utils.kernel import create_kernel
from config import train_file, valid_file, THR_01, THR_02

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

x_patch = [(0, 64), (32, 96), (64, 128), (96, 160), (128, 192), (160, 224)]
y_patch = [(0, 64), (32, 96), (64, 128), (96, 160), (128, 192), (160, 224)]
z1 = [(0, 64), (14, 78), (32, 96), (48, 112), (62, 126)]
z2 = [(0, 64), (14, 78), (32, 96), (46, 110)]

second_group = ['Sub005', 'Sub006', 'Sub008', 'Sub009']

test_group, valid_group = ['Sub002'], ['Sub008']

base_folder = '../data/Kirby_data'
deg_oris = [[15, 0, 0], [-15, 0, 0],
            [0, 15, 0], [0, -15, 0]]

gyro = 2 * 3.14159265 * 42.58
B0 = 7
TE = 0.23


def partition(judger: Callable, hdf5_file: str, thr: float):
    kernel_patches = []
    k_tdk_real_patches, k_tdk_imag_patches = [], []
    k_susc_real_patches, k_susc_imag_patches = [], []

    subjects = [i for i in os.listdir(base_folder) if os.path.isdir(f'{base_folder}/{i}')]
    dots = ''
    for i, sub in enumerate(subjects):

        sub_folder = f'{base_folder}/{sub}'
        z_patch = z2 if sub in second_group else z1

        mat = loadmat(f'{sub_folder}/cosmos.mat')
        mask = to_torch(mat['mask'])
        susc = to_torch(mat['susc'])
        k_susc = fftshift(fftn(susc))

        oris = [ori for ori in os.listdir(sub_folder) if 'ori' in ori]

        for j, ori in enumerate(oris):
            if not judger(sub):
                continue
            dots = dots + '.' if len(dots) < 3 else '.'
            print(f'\r  Processing {sub} [{i + 1}/{len(subjects)}] ori{j + 1} {dots}', end='')

            phase = load_nifti_data(f'{sub_folder}/{ori}/{sub}_{ori}_phase.nii.gz')
            phase = phase / (TE * B0 * gyro)

            with open(f'{sub_folder}/{ori}/{sub}_{ori}.txt') as f:
                ori_vec = np.array(f.read().split('\n'), dtype=np.float32)

            kernel = fftshift(create_kernel(phase.shape, ori_vec))

            kernel_bak = kernel.clone()
            kernel_bak[kernel.abs() < thr] = thr
            kernel_inv = torch.sign(kernel_bak) / kernel_bak.abs()

            k_phase = fftshift(fftn(to_torch(phase)))
            k_tkd = kernel_inv * k_phase

            for xx in x_patch:
                for yy in y_patch:
                    for zz in z_patch:
                        knl_patch = kernel[xx[0]:xx[1], yy[0]:yy[1], zz[0]:zz[1]]
                        k_tkd_patch = k_tkd[xx[0]:xx[1], yy[0]:yy[1], zz[0]:zz[1]]
                        k_susc_patch = k_susc[xx[0]:xx[1], yy[0]:yy[1], zz[0]:zz[1]]

                        kernel_patches.append(knl_patch.numpy())
                        k_tdk_real_patches.append(torch.real(k_tkd_patch).numpy())
                        k_tdk_imag_patches.append(torch.imag(k_tkd_patch).numpy())

                        k_susc_real_patches.append(torch.real(k_susc_patch).numpy())
                        k_susc_imag_patches.append(torch.imag(k_susc_patch).numpy())

        # data augmentation
        for j, deg_ori in enumerate(deg_oris):
            if not judger(sub):
                continue
            dots = dots + '.' if len(dots) < 3 else '.'
            print(f'\r  Processing {sub} [{i + 1}/{len(subjects)}] ori{j + 1} {dots}', end='')

            x_deg, y_deg, _ = deg_ori

            rad_X = -x_deg * np.pi / 180
            rad_Y = -y_deg * np.pi / 180
            ori_vec = [np.sin(rad_Y) * np.cos(rad_X), np.sin(rad_X), np.cos(rad_X) * np.cos(rad_Y)]

            kernel = fftshift(create_kernel(mask.shape, ori_vec))

            kernel_bk = kernel.clone()
            kernel_bk[kernel_bk.abs() < thr] = thr
            k_tkd = k_susc / kernel_bk
            kernel_bk[kernel_bk.abs() < thr*2] = thr*2
            k_tkd = k_tkd * kernel_bk

            for xx in x_patch:
                for yy in y_patch:
                    for zz in z_patch:
                        knl_patch = kernel[xx[0]:xx[1], yy[0]:yy[1], zz[0]:zz[1]]
                        k_tkd_patch = k_tkd[xx[0]:xx[1], yy[0]:yy[1], zz[0]:zz[1]]
                        k_susc_patch = k_susc[xx[0]:xx[1], yy[0]:yy[1], zz[0]:zz[1]]

                        kernel_patches.append(knl_patch.numpy())
                        k_tdk_real_patches.append(torch.real(k_tkd_patch).numpy())
                        k_tdk_imag_patches.append(torch.imag(k_tkd_patch).numpy())

                        k_susc_real_patches.append(torch.real(k_susc_patch).numpy())
                        k_susc_imag_patches.append(torch.imag(k_susc_patch).numpy())

    print('\r  [DONE]Processing                              ')
    f = h5py.File(hdf5_file, 'w')

    print('  Saving kernel_patches ...', end='')
    kernel_patches = np.array(kernel_patches)
    shape = kernel_patches.shape
    f.create_dataset('kernel', data=kernel_patches)
    del kernel_patches
    print('\r  [DONE]kernel_patches')

    print('  Saving k_tdk_real_patches ...', end='')
    k_tdk_real_patches = np.array(k_tdk_real_patches)
    f.create_dataset('r_tdk', data=k_tdk_real_patches)
    del k_tdk_real_patches
    print('\r  [DONE]k_tdk_real_patches')

    print('  Saving k_tdk_imag_patches ...', end='')
    k_tdk_imag_patches = np.array(k_tdk_imag_patches)
    f.create_dataset('i_tdk', data=k_tdk_imag_patches)
    del k_tdk_imag_patches
    print('\r  [DONE]k_tdk_imag_patches')

    print('  Saving k_susc_real_patches ...', end='')
    k_susc_real_patches = np.array(k_susc_real_patches)
    f.create_dataset('r_susc', data=k_susc_real_patches)
    del k_susc_real_patches
    print('\r  [DONE]k_susc_real_patches')

    print('  Saving k_susc_imag_patches ...', end='')
    k_susc_imag_patches = np.array(k_susc_imag_patches)
    f.create_dataset('i_susc', data=k_susc_imag_patches)
    del k_susc_imag_patches
    print('\r  [DONE]k_susc_imag_patches')

    f.close()
    print('  [All DONE]Data size:', shape)


def patch_train(thr: float):
    print('In patch_train')
    judger: Callable[[str], bool] = lambda sub: sub not in test_group and sub not in valid_group
    partition(judger, train_file(thr), thr)
    print('Done patch_train')


def patch_valid(thr: float):
    print('In patch_valid')
    judger: Callable[[str], bool] = lambda sub: sub in valid_group
    partition(judger, valid_file(thr), thr)
    print('Done patch_valid')


if __name__ == '__main__':
    for t in [THR_01, THR_02]:
        print(f'-----Processing thr={t}-----')
        patch_train(t)
        patch_valid(t)
        print('-----------------------------\n')
