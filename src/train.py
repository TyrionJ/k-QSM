import torch
from tqdm import tqdm
import time
import warnings

from v8_9_1.config import record_file, TDK_THRESHOLD
from v8_9_1.net_dataset import QSMDataset
from v8_9_1.chk_utils import save_model, load_model
from common import proj_param, set_env, get_data_loader

warnings.filterwarnings('ignore')


def train(epoch, data_loader, model, loss_func, opt, device, epoch_num):
    model.train()
    avg_loss = 0

    with tqdm(total=len(data_loader), desc=f'Epoch {epoch + 1}/{epoch_num}', unit='it') as pbar:
        for N, item in enumerate(data_loader):
            opt.zero_grad()

            x, y = item
            x = x.to(device, dtype=torch.float)
            y_hat = model(x)

            y = y.to(device, dtype=torch.float)
            t_loss, mse_loss, l1_loss = loss_func(y, y_hat, x, TDK_THRESHOLD)
            avg_loss = (avg_loss * N + t_loss.item()) / (N + 1)

            t_loss.backward()
            opt.step()

            pbar.set_postfix(**{'1.avg_loss': avg_loss,
                                '2.bat_loss': t_loss.item(),
                                '3.mse_loss': mse_loss.item(),
                                '4.L1_loss': l1_loss.item()})
            pbar.update()

    return round(avg_loss, 6)


def validation(data_loader, model, loss_func, device):
    print('\nValidating ...')
    print('Validation Size:', len(data_loader.dataset))
    avg_loss = 0

    model.eval()
    for N, item in enumerate(data_loader):
        x, y = item
        x = x.to(device, dtype=torch.float)
        y_hat = model(x)

        y = y.to(device, dtype=torch.float)
        t_loss, mse_loss, l1_loss = loss_func(y, y_hat, x, TDK_THRESHOLD)

        avg_loss = (avg_loss * N + t_loss.item()) / (N + 1)

    avg_loss = round(avg_loss, 6)
    print(f'####[Validation Loss]: {avg_loss}\n')

    return avg_loss


def main(args):
    device = set_env(args.device)

    train_set = QSMDataset()
    valid_set = QSMDataset(mode='valid')
    train_loader, valid_loader = get_data_loader(args, train_set, valid_set)

    start_epoch, model, loss_func, opt, best_valid = load_model(device)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=50, gamma=0.9)

    for epoch in range(start_epoch, args.epoch):
        start_time = time.time()
        train_loss = train(epoch, train_loader, model,
                           loss_func, opt,
                           device, args.epoch)
        duration = round(time.time() - start_time, 6)

        if (epoch+1) % 5 == 0 or True:
            val_loss = validation(valid_loader, model, loss_func, device)
        else:
            val_loss = -999
        sched.step()

        with open(record_file, 'a') as f:
            f.write(f'[{epoch + 1}/{args.epoch}]: {duration}\t'
                    f'{train_loss}\t'
                    f'{val_loss}\n')

        _best_valid = val_loss if val_loss < best_valid else None
        save_model(epoch, model, opt, _best_valid)


if __name__ == '__main__':
    main(proj_param())
