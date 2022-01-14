import torch
from tqdm import tqdm
import time
import argparse

from config import record_file, last_model, THR_01, THR_02
from network.net_dataset import QSMDataset
from utils.proj_utils import save_model, load_model
from utils.common import set_env, get_data_loader


def train(epoch, data_loader, model, loss_func, opt, device, epoch_num, thr):
    model.train()
    avg_loss = 0

    with tqdm(total=len(data_loader), desc=f'Epoch {epoch + 1}/{epoch_num}', unit='it') as pbar:
        for N, item in enumerate(data_loader):
            opt.zero_grad()

            x, y = item
            x = x.to(device, dtype=torch.float)
            y_hat = model(x)

            y = y.to(device, dtype=torch.float)
            t_loss, mse_loss, l1_loss = loss_func(y, y_hat, x, thr)
            avg_loss = (avg_loss * N + t_loss.item()) / (N + 1)

            t_loss.backward()
            opt.step()

            pbar.set_postfix(**{'1)avg_loss': avg_loss,
                                '2)bat_loss': t_loss.item(),
                                '3)mse_loss': mse_loss.item(),
                                '4)L1_loss': l1_loss.item()})
            pbar.update()

    return round(avg_loss, 6)


def validation(data_loader, model, loss_func, device, thr):
    time.sleep(0.1)
    print(f' Validating [Size={len(data_loader.dataset)}]...', end='')
    avg_loss = 0

    model.eval()
    for N, item in enumerate(data_loader):
        x, y = item
        x = x.to(device, dtype=torch.float)
        y_hat = model(x)

        y = y.to(device, dtype=torch.float)
        t_loss, mse_loss, l1_loss = loss_func(y, y_hat, x, thr)

        avg_loss = (avg_loss * N + t_loss.item()) / (N + 1)

    avg_loss = round(avg_loss, 6)
    print(f'\r Validating [Size={len(data_loader.dataset)}, loss={avg_loss}]...')
    time.sleep(0.1)

    return avg_loss


def main(args):
    device = set_env(args.device)
    thr = args.threshold

    train_set = QSMDataset(thr, mode='train')
    valid_set = QSMDataset(thr, mode='valid')
    train_loader, valid_loader = get_data_loader(args, train_set, valid_set)

    start_epoch, model, loss_func, opt, best_valid = load_model(device, last_model(thr))
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=50, gamma=0.9)

    print('\nTraining model ...')
    time.sleep(0.1)
    for epoch in range(start_epoch, args.epoch):
        start_time = time.time()
        train_loss = train(epoch, train_loader, model,
                           loss_func, opt,
                           device, args.epoch, thr)
        duration = round(time.time() - start_time, 6)

        if (epoch+1) % 5 == 0:
            val_loss = validation(valid_loader, model, loss_func, device, thr)
        else:
            val_loss = None
        sched.step()

        with open(record_file(thr), 'a') as f:
            f.write(f'[{epoch + 1}/{args.epoch}]: {duration}\t'
                    f'{train_loss}\t'
                    f'{val_loss}\n')

        _best_valid = None if val_loss is None or val_loss >= best_valid else val_loss
        save_model(epoch, model, opt, thr, _best_valid)


def proj_param():
    parser = argparse.ArgumentParser(description='k-QSM: MRI DIPOLE INVERSION')
    parser.add_argument('-d', '--device', default='cpu', type=str)
    parser.add_argument('-b', '--batch_size', default=1, type=int)
    parser.add_argument('-e', '--epoch', default=500, type=int)
    parser.add_argument('-t', '--threshold', default=0.1, type=float, choices=[THR_01, THR_02])

    return parser.parse_args()


if __name__ == '__main__':
    main(proj_param())
