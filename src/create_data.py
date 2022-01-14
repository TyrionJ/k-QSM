import os
import h5py
import torch
from torch.fft import fftn, ifftn, fftshift
from torch import from_numpy as to_torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
from scipy.io import savemat, loadmat
from dipy.io.image import load_nifti_data

from utils.cosmos import gen_cosmos
from utils.rotation import aug_susc
from utils.kernel import read_b0_vec, create_kernel
from v8_9_1.config import train_file, valid_file, PATCH_SIZE, test_prefix, TDK_THRESHOLD
from v8_9_1.net_dataset import QSMDataset

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

x_patch = [(0, 64), (32, 96), (64, 128), (96, 160), (128, 192), (160, 224)]
y_patch = [(0, 64), (32, 96), (64, 128), (96, 160), (128, 192), (160, 224)]
z1 = [(0, 64), (14, 78), (32, 96), (48, 112), (62, 126)]
z2 = [(0, 64), (14, 78), (32, 96), (46, 110)]

second_group = ['Sub005', 'Sub006', 'Sub008', 'Sub009']

test_group = ['Sub001_ori1', 'Sub002_ori2', 'Sub003_ori3', 'Sub002']
valid_group = ['Sub004_ori3', 'Sub008_ori2']

base_folder = '../../../nifti_data'
deg_oris = [[15, 0, 0], [-15, 0, 0],
            [0, 15, 0], [0, -15, 0]]

gyro = 2 * 3.14159265 * 42.58
B0 = 7
TE = 0.23


def partition(judger, hdf5_file):
    kernel_patches = []
    k_tdk_real_patches, k_tdk_imag_patches = [], []
    k_susc_real_patches, k_susc_imag_patches = [], []

    subjects = [i for i in os.listdir(base_folder) if os.path.isdir(f'{base_folder}/{i}')]
    dots = ''
    for i, sub in enumerate(subjects):

        sub_folder = f'{base_folder}/{sub}'
        z_patch = z2 if sub in second_group else z1

        mat = loadmat(f'{sub_folder}/new_cosmos.mat')
        mask = to_torch(mat['mask'])
        susc = to_torch(mat['susc'])
        k_susc = fftshift(fftn(susc))

        oris = [ori for ori in os.listdir(sub_folder) if 'ori' in ori]

        for j, ori in enumerate(oris):
            if not judger(sub, j + 1):
                continue
            dots = dots + '.' if len(dots) < 3 else '.'
            print(f'\r  Processing {sub} [{i + 1}/{len(subjects)}] ori{j + 1} {dots}', end='')

            phase = load_nifti_data(f'{sub_folder}/{ori}/{sub}_{ori}_phase.nii.gz')
            phase = phase / (TE * B0 * gyro)

            ori_vec, _, _ = read_b0_vec(f'{sub_folder}/{ori}/{sub}_{ori}.txt')

            kernel = fftshift(create_kernel(phase.shape, ori_vec))

            kernel_bak = kernel.clone()
            kernel_bak[kernel.abs() < TDK_THRESHOLD] = TDK_THRESHOLD
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
            if not judger(sub, j + 1):
                continue
            dots = dots + '.' if len(dots) < 3 else '.'
            print(f'\r  Processing {sub} [{i + 1}/{len(subjects)}] ori{j + 1} {dots}', end='')

            x_deg, y_deg, _ = deg_ori

            rad_X = -x_deg * np.pi / 180
            rad_Y = -y_deg * np.pi / 180
            ori_vec = [np.sin(rad_Y) * np.cos(rad_X), np.sin(rad_X), np.cos(rad_X) * np.cos(rad_Y)]

            kernel = fftshift(create_kernel(mask.shape, ori_vec))

            kernel_bk = kernel.clone()
            kernel_bk[kernel_bk.abs() < TDK_THRESHOLD] = TDK_THRESHOLD
            k_tkd = k_susc / kernel_bk
            kernel_bk[kernel_bk.abs() < TDK_THRESHOLD*2] = TDK_THRESHOLD*2
            k_tkd = k_tkd * kernel_bk

            for xx in x_patch:
                for yy in y_patch:
                    for zz in z_patch:
                        knl_patch = kernel[xx[0]:xx[1], yy[0]:yy[1], zz[0]:zz[1]]
                        k_tkd_patch = k_tkd[xx[0]:xx[1], yy[0]:yy[1], zz[0]:zz[1]]
                        k_susc_patch = k_susc[xx[0]:xx[1], yy[0]:yy[1], zz[0]:zz[1]]

                        k_mask = k_tkd_patch.clone()
                        k_mask[knl_patch.abs()<TDK_THRESHOLD] = 0

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


def patch_train():
    print('In patch_train')
    judger = lambda sub, i: f'{sub}_ori{i}' not in test_group and f'{sub}_ori{i}' not in valid_group and f'{sub}' not in test_group
    partition(judger, train_file)
    print('Done patch_train')


def patch_valid():
    print('In patch_valid')
    judger = lambda sub, i: f'{sub}_ori{i}' in valid_group
    partition(judger, valid_file)
    print('Done patch_valid')


def create_test():
    print('In inference_test ...')
    subjects = os.listdir(base_folder)

    t_id = 1
    dots = '.'
    for i, sub in enumerate(subjects):
        sub_folder = f'{base_folder}/{sub}'

        mat = loadmat(f'{sub_folder}/new_cosmos.mat')
        mask = to_torch(mat['mask'])
        susc = to_torch(mat['susc'])

        if sub in test_group:
            tissue_phs, kernel, k_phase, ori_vec = aug_susc(susc, mask, 0, 0)
            savemat(f'{test_prefix}{t_id}.mat',
                    mdict={'X': tissue_phs.numpy(),
                           'Y': susc.numpy(),
                           'mask': mask.numpy(),
                           'ORI': ori_vec})
            t_id += 1

        for j, deg_ori in enumerate(deg_oris):
            if f'{sub}_ori{j + 1}' not in test_group:
                continue
            dots = dots + '.' if len(dots) < 3 else '.'
            print(f'\r  Processing {sub} [{i + 1}/{len(subjects)}] ori{j + 1} {dots}', end='')

            x_deg, y_deg, _ = deg_ori
            tissue_phs, kernel, k_phase, ori_vec = aug_susc(susc, mask, x_deg, y_deg)

            savemat(f'{test_prefix}{t_id}.mat',
                    mdict={'X': tissue_phs.numpy(),
                           'Y': susc.numpy(),
                           'mask': mask.numpy(),
                           'ORI': ori_vec})
            t_id += 1
    print(f'\n[Done]inference_test: {t_id - 1}')


def test_processed():
    dataset = QSMDataset(mode='train')
    data_loader = DataLoader(dataset, batch_size=1)

    for i, item in enumerate(data_loader):
        X, Y = item
        kernel = X[:, 2, :, :, :]
        k_tdk = torch.complex(X[:, 0, :, :, :], X[:, 1, :, :, :])
        k_susc = torch.complex(Y[:, 0, :, :, :], Y[:, 1, :, :, :])

        k_tdk = torch.abs(k_tdk)
        k_susc = torch.abs(k_susc)

        _, _, y, z = np.array(k_tdk.shape) // 2

        fig = plt.figure(i)
        plt.subplot(2, 3, 1)
        plt.imshow(k_susc[0, :, :, z], cmap='gray', vmin=0, vmax=100)
        plt.title('susc')
        plt.subplot(2, 3, 2)
        plt.imshow(k_tdk[0, :, :, z], cmap='gray', vmin=0, vmax=100)
        plt.title("TDK")
        plt.subplot(2, 3, 3)
        plt.imshow(kernel[0, :, y, :], cmap='gray')
        plt.title("kernel")

        plt.show()
        plt.close(fig)


def phs2cosmos():
    print('In phs2cosmos')
    subs = os.listdir(base_folder)

    for sub in subs:
        print(f'\r  Processing {sub} ...', end='')
        cosmos_file = f'{base_folder}/{sub}/new_cosmos.mat'

        mask = load_nifti_data(f'{base_folder}/{sub}/cosmos/{sub}_mask.nii.gz')
        phase_arr, ori_vec_arr = [], []

        oris = [ori for ori in os.listdir(f'{base_folder}/{sub}') if 'ori' in ori]
        for ori in oris:
            phase = load_nifti_data(f'{base_folder}/{sub}/{ori}/{sub}_{ori}_phase.nii.gz')
            b0_vec, _, _ = read_b0_vec(f'{base_folder}/{sub}/{ori}/{sub}_{ori}.txt')

            phase = phase / (TE * B0 * gyro)
            phase_arr.append(phase)
            ori_vec_arr.append(b0_vec)

        cosmos = gen_cosmos(phase_arr, ori_vec_arr, mask)

        savemat(cosmos_file, mdict={'susc': cosmos.numpy(), 'mask': mask})
    print()


def show_new_cosmos():
    nii_folder = r'D:\Program\Python\IMIDP_Research\QSM\Dipole_Inversion\nifti_data'
    subs = os.listdir(nii_folder)
    for i, sub in enumerate(subs):
        mat = loadmat(rf'{nii_folder}\{sub}\new_cosmos.mat')
        susc = mat['susc']

        fig = plt.figure(i + 1)

        plt.subplot(1, 3, 1)
        plt.imshow(susc[:, :, 55].T, cmap='gray', vmin=-0.1, vmax=0.12)
        plt.subplot(1, 3, 2)
        plt.imshow(np.rot90(susc[:, 112, :]), cmap='gray', vmin=-0.1, vmax=0.12)
        plt.subplot(1, 3, 3)
        plt.imshow(np.rot90(susc[112, :, :]), cmap='gray', vmin=-0.1, vmax=0.12)
        plt.show()

        plt.close(fig)


if __name__ == '__main__':
    # phs2cosmos()
    # patch_valid()
    # patch_train()
    # test_processed()
    create_test()
    # show_new_cosmos()
