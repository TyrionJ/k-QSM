import torch
import torch.nn as nn


class NetLoss(nn.Module):
    def __init__(self):
        super(NetLoss, self).__init__()
        self.MSELoss = nn.MSELoss()
        self.L1Loss = nn.L1Loss()

    def forward(self, y_label, y_hat, x, thr):
        mse_loss = self.MSELoss(y_label, y_hat)

        thr_mask = x[:, 2, :, :, :].abs() > thr
        thr_mask = torch.unsqueeze(thr_mask, 1)
        thr_mask = thr_mask.expand_as(y_hat)

        l1_loss = self.L1Loss(x[:, 0:2, :, :, :][thr_mask], y_hat[thr_mask])
        t_loss = mse_loss + l1_loss

        return t_loss, mse_loss, l1_loss
