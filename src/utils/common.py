import os
import torch
from torch.utils.data import DataLoader


def set_env(device):
    torch.set_default_dtype(torch.float32)
    if device != 'cpu':
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = device
        return torch.device('cuda')
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        return torch.device('cpu')


def get_data_loader(args, train_set, valid_set):
    train_loader = DataLoader(
        dataset=train_set,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.batch_size * len(args.device.split(','))
    )

    valid_loader = DataLoader(
        dataset=valid_set,
        shuffle=True,
        batch_size=1,
        num_workers=1
    )

    return train_loader, valid_loader
