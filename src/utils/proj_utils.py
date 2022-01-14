import os
import torch
from torch import optim
from torch.fft import fftshift, fftn
import time

from utils.kernel import create_kernel
from config import last_model, best_model, check_folder
from network.net_loss import NetLoss
from network.net_model import LFDQSM


def phase2in(phase, ori, thd):
    b, _, w, h, d = phase.shape
    inputs = torch.empty([b, 3, w, h, d], device=phase.device)

    for btn in range(b):
        kernel = fftshift(create_kernel([w, h, d], ori[btn][0]).to(device=phase.device))
        k_phase = fftshift(fftn(phase[btn][0]))

        kernel_bak = kernel.clone()
        kernel_bak[kernel.abs() < thd] = thd
        kernel_inv = torch.sign(kernel_bak) / kernel_bak.abs()

        k_tkd = kernel_inv * k_phase

        inputs[btn][0] = torch.real(k_tkd)
        inputs[btn][1] = torch.imag(k_tkd)
        inputs[btn][2] = torch.unsqueeze(kernel, 0)

    return inputs


def save_model(epoch, model, opt, thr, best_valid=None):
    state = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': opt.state_dict(),
        'best_valid': best_valid
    }
    torch.save(state, last_model(thr))

    if best_valid is not None:
        torch.save(state, best_model(thr))

    if (epoch+1) % 10 == 0:
        torch.save(state, f'{check_folder}/{thr}/m_{epoch}.pkt')


def load_model(device, model_file):
    print('Loading model ...')
    epoch, best_valid = 0, 999
    loss_func = NetLoss()

    model = LFDQSM()
    model.to(device=device)

    opt = optim.Adam(model.parameters(), lr=0.0001)

    if os.path.exists(model_file):
        state = torch.load(model_file, map_location=device)
        epoch = state['epoch'] + 1
        model.load_state_dict(state['model_state'])
        opt.load_state_dict(state['optimizer_state'])
        best_valid = state['best_valid'] if best_valid in state else best_valid

        print('  ' + model_file + ' loaded.')
        print('  epoch: ' + str(epoch))
    else:
        time.sleep(0.1)
        print('  Use initial model')

    return epoch, model, loss_func, opt, best_valid
