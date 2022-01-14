from abc import ABC
import torch
import h5py
import numpy as np
from torch.utils.data import Dataset

from v8_9_1.config import train_file, valid_file


class QSMDataset(Dataset, ABC):
    def __init__(self, mode: str = 'train'):
        super(QSMDataset, self).__init__()

        self.mode = mode
        self.kernel = None
        self.r_input, self.i_input = None, None
        self.r_label, self.i_label = None, None

        self.dataset_len = 0
        self.data_file = train_file if mode == 'train' else valid_file

        with h5py.File(self.data_file, 'r') as f:
            self.dataset_len = len(f['kernel'])

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        if self.r_input is None:
            self.kernel = h5py.File(self.data_file, "r")['kernel']
            self.r_input = h5py.File(self.data_file, "r")['r_tdk']
            self.i_input = h5py.File(self.data_file, "r")['i_tdk']
            self.r_label = h5py.File(self.data_file, "r")['r_susc']
            self.i_label = h5py.File(self.data_file, "r")['i_susc']

        kernel = np.expand_dims(self.kernel[index], 0)
        r_input = np.expand_dims(self.r_input[index], 0)
        i_input = np.expand_dims(self.i_input[index], 0)
        r_label = np.expand_dims(self.r_label[index], 0)
        i_label = np.expand_dims(self.i_label[index], 0)

        x = np.concatenate([r_input, i_input, kernel])
        y = np.concatenate([r_label, i_label])

        return [torch.from_numpy(x.astype(np.float32)),
                torch.from_numpy(y.astype(np.float32))]
