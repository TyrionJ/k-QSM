import torch
from torch.fft import fftshift, fftn, ifftn
import numpy as np


def create_kernel(shape, ori_vec=None, vox_sz=None):
    if vox_sz is None:
        vox_sz = [1., 1., 1.]
    if ori_vec is None:
        ori_vec = [0., 0., 1.]
    N = torch.tensor(shape, dtype=torch.int)
    kx, ky, kz = torch.meshgrid(torch.arange(-N[0]//2, N[0]//2),
                                torch.arange(-N[1]//2, N[1]//2),
                                torch.arange(-N[2]//2, N[2]//2))

    spatial = torch.tensor(vox_sz, dtype=torch.double)
    kx = (kx / kx.abs().max()) / spatial[0]
    ky = (ky / ky.abs().max()) / spatial[1]
    kz = (kz / kz.abs().max()) / spatial[2]
    k2 = kx ** 2 + ky ** 2 + kz ** 2 + 2.2204e-16

    tk = 1 / 3 - (kx * ori_vec[0] + ky * ori_vec[1] + kz * ori_vec[2]) ** 2 / k2
    tk = fftshift(tk)

    return tk


def susc2phase(susc, ori=None):
    if ori is None:
        ori = [0., 0., 1.]
    k_dipole = create_kernel(susc.shape, ori)
    k_cosmos = fftn(susc)

    return torch.real(ifftn(k_cosmos * k_dipole))


def gen_tkd_qsm(phase, mask=None, ori=None, thd=None, voxel_size=None):
    if voxel_size is None:
        voxel_size = [1., 1., 1.]
    if ori is None:
        ori = [0., 0., 1.]
    if thd is None:
        thd = 0.1
    kernel = create_kernel(phase.shape, ori, voxel_size)
    kernel[kernel.abs() < thd] = thd
    kernel_inv = torch.sign(kernel) / kernel.abs()
    del kernel

    k_phase = fftn(phase)
    k_tkd = k_phase * kernel_inv
    del k_phase, kernel_inv

    tkd = ifftn(k_tkd)
    tkd = torch.abs(tkd) * torch.sign(torch.real(tkd))
    del k_tkd

    if mask is not None:
        tkd *= mask

    return tkd


def read_b0_vec(ori_file):
    with open(ori_file) as f:
        vec = np.array(f.read().split('\n'), dtype=np.float32)
    rad_X = np.arcsin(vec[1])
    rad_Y = np.arcsin(vec[0] / np.cos(rad_X))

    return vec, rad_X, rad_Y


def test():
    from scipy.io import loadmat
    import matplotlib.pyplot as plt
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    person = 'dengzhiming'
    phase = loadmat(fr'D:\Data\QSM_STEPS_MAT\{person}\tissue_phase.mat')['tissue_phase']
    mask = loadmat(fr'D:\Data\QSM_STEPS_MAT\{person}\mask.mat')['mask']
    star_qsm = loadmat(fr'D:\Data\QSM_STEPS_MAT\{person}\STAR_QSM.mat')['QSM_IMG']
    header = loadmat(fr'D:\Data\QSM_STEPS_MAT\{person}\header.mat')
    B0_vector = header['B0_vector'][0]
    B0 = header['B0'][0][0]
    TEs = header['TEs'][0]
    vox = header['voxel_size'][0]
    kernel = create_kernel(mask.shape, B0_vector, vox)

    gamma = 42.58
    gyro = 2 * np.pi * gamma
    phase = phase / (TEs.mean() * 1e-3 * B0 * gyro)
    phase = torch.from_numpy(phase)
    k_phs = torch.fft.fftn(phase)

    kthre = 0.1
    kernel_inv = torch.zeros_like(kernel)
    good_k = abs(kernel) > kthre
    kernel_inv[good_k] = 1 / kernel[good_k]

    mask = torch.from_numpy(mask)
    chi_tkd = torch.real(torch.fft.ifftn(k_phs * kernel_inv)) * mask

    print(torch.min(chi_tkd), torch.max(chi_tkd))
    print(np.min(star_qsm), np.max(star_qsm))

    sli = mask.shape[2] // 2
    plt.subplot(1, 3, 1)
    plt.axis('off')
    plt.imshow(phase[:, :, sli], cmap='gray')
    plt.title('phase')

    plt.subplot(1, 3, 2)
    plt.axis('off')
    plt.imshow(star_qsm[:, :, sli], cmap='gray')
    plt.title('STAR-QSM')

    plt.subplot(1, 3, 3)
    plt.axis('off')
    plt.imshow(chi_tkd[:, :, sli], cmap='gray')
    plt.title('TKD')

    plt.show()

if __name__ == '__main__':
    test()