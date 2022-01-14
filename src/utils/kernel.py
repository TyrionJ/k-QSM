import torch
from torch.fft import fftshift


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
