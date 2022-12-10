import random
import os
import json
import numpy as np
# from sklearn.metrics import precision_recall_fscore_support, classification_report
from tqdm import tqdm, trange

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch import nn, optim
from utils import *
from networks import RNNPredictor, RNNPredictorSpecAug, ChenPredictor
import argparse
from datetime import datetime
import re

import wandb

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser()
parser.add_argument('--source_id', type=int, default=3)
parser.add_argument('--obs_ord', type=int, default=1)
parser.add_argument('--hidden_size', type=int, default=32)
parser.add_argument('--scale', type=float, default=Delta_i)
parser.add_argument('--num_layers', type=int, default=1)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--data_path', type=str, default='./traces/')
parser.add_argument('--seq_len', type=int, default=256)
parser.add_argument('--loss', type=str, default='mse')
parser.add_argument('--backbone', type=str, default='lstm')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--model', type=str, default='Chen')
parser.add_argument('--do_parse_only', type=bool, default=False)
parser.add_argument('--wandb', type=bool, default=True)
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

GLOBAL_STEPS = 0

def one_epoch(
    model: nn.Module,
    data: np.ndarray,
    mode: str,
    device: torch.device = DEVICE,
    epoch_id: int = None,
    optimizer: any = optim.SGD,
    args: argparse.Namespace = None,
    writer: SummaryWriter = None,
):
    loss_cls = nn.MSELoss if args.loss == 'mse' else None
    def run(mode, device=device, loss_cls=loss_cls, writer=writer):
        criterion = loss_cls()
        losses = []
        fp_cnt, td_tot = 0, 0
        length = 0
        for r in trange(1, data.shape[0], desc='Epoch '+str(epoch_id)+' '+mode):
            # inputs, targets = batch[0].to(device), batch[1].to(device)
            l = max(0, r-args.seq_len)
            source = torch.from_numpy(data[l:r])
            if r < args.seq_len: # left paddings
                left_zeros = torch.zeros(args.seq_len-source.shape[0], source.shape[1])
                source = torch.cat((left_zeros, source), dim=0)
            source = source.unsqueeze(0).float().to(device)
            target = torch.tensor([data[r][0]]).reshape(1, 1).float().to(device)
            length += source.shape[0]
            # forward
            output = model(source)
            ### pred = torch.argmax(output, dim=1) ?????? DAMN!
            pred = output
            # criteria
            loss = criterion(pred, target)
            detection_time = torch.maximum(
                input=pred-target,
                other=torch.zeros_like(pred)
            )
            if args.wandb:
                wandb.log({
                    'step loss': loss.item(),
                    'step detection time (scaled)': detection_time,
                    'step fp': float((pred < target).item())
                })
            else:  # debug mode
                print(pred, target)
            fp_cnt += float((pred < target).item())
            td_tot += detection_time.item()
            losses.append(loss.item())

            if writer:
                global GLOBAL_STEPS
                writer.add_scalar(f'step_loss/{mode}', loss.item(), GLOBAL_STEPS)
                GLOBAL_STEPS += 1

            if mode == 'train':
                opt.zero_grad()
                loss.backward()
                opt.step()
                
        return {'loss': np.mean(losses), 'fpr': fp_cnt / length, 'td_avg': td_tot / length}

    if mode == 'train':
        opt = optimizer(model.parameters(), lr=args.lr)
        model = model.train()
        return run(mode, device)
    else:
        model = model.eval()
        with torch.no_grad():
            return run(mode, device)


def train(
    model: nn.Module,
    train_data: np.ndarray,
    test_data: np.ndarray,
    args: argparse.Namespace,
    device: torch.device = DEVICE,
    writer: SummaryWriter = None,
    evaluate_first: bool = True
):
    model.to(device)
    if evaluate_first:
        train_ret = {'loss': None, 'fpr': None, 'td_avg': None}
        val_ret = one_epoch(model, test_data, 'val', device, epoch_id=0, args=args, writer=writer)

        if args.wandb:
            wandb.log(log_epoch(train_ret, val_ret))

    if args.model == 'Chen':  # not trainable
        return

    for epoch in range(1, args.epochs+1):
        train_ret = one_epoch(model, train_data, 'train', device, epoch_id=epoch, args=args, writer=writer)

        print('Epoch {}: '.format(epoch))
        print(f"train loss: {train_ret['loss']}")
        print(f"train fpr: {train_ret['fpr']}")
        print(f"train td_avg: {train_ret['td_avg']}")

        val_ret = one_epoch(model, test_data, 'val', device, epoch_id=epoch, args=args, writer=writer)
        
        print(f"eval loss: {val_ret['loss']}")
        print(f"eval fpr: {val_ret['fpr']}")
        print(f"eval td_avg: {val_ret['td_avg']}")

        if args.wandb:
            wandb.log(log_epoch(train_ret, val_ret))
        
        if writer:
            writer.add_scalar('loss/train', train_ret['loss'], epoch)
            writer.add_scalar('fpr/train', train_ret['fpr'], epoch)
            writer.add_scalar('td_avg/train', train_ret['td_avg'], epoch)
            writer.add_scalar('loss/val', val_ret['loss'], epoch)
            writer.add_scalar('fpr/val', val_ret['fpr'], epoch)
            writer.add_scalar('td_avg/val', val_ret['td_avg'], epoch)
    
    model.cpu()
            
"""
def evaluate(model, device, data, comment='eval', epoch_id=None):
    model = model.to(device)
    val_ret = one_epoch(model, data, 'eval', device, epoch_id)

    return val_ret

"""
if __name__ == '__main__':
    hyper_dict = vars(args)
    print(hyper_dict)
    # exit()

    train_data, test_data = get_data(
        trace_path='./traces/trace.log',
        source_id=args.source_id,
        obs_ord=args.obs_ord,
        scale=args.scale,
    )
    if args.do_parse_only:
        exit()
    
    if args.wandb:
        wandb.init(project='dlfd', entity='kgv007', config=hyper_dict)
    rnn_cls = nn.LSTM if args.backbone == 'lstm' else nn.RNN

    # model_cls = RNNPredictor if not args.spec_aug else RNNPredictorSpecAug
    if args.model == 'Chen':
        model_cls = ChenPredictor
    elif args.model == 'RNNBase':
        model_cls = RNNPredictor
    elif args.model == 'RNNSpecAug':
        model_cls = RNNPredictorSpecAug
    else:
        print(f'model type {args.model} unsupported...')
        exit()

    model = model_cls(
        obs_ord=args.obs_ord,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        rnn=rnn_cls,
        seq_len=args.seq_len,
        scale=args.scale,
    )

    cur_time = re.sub(r'\W+', '', str(datetime.now()))
    tb_dir = os.path.join('./runs/', cur_time)
    os.makedirs(tb_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=tb_dir)
    for k, v in hyper_dict.items():
        writer.add_text(k, str(v))
    # writer.add_hparams(vars(args), metric_dict={'loss': np.inf})

    train(
        model=model,
        train_data=train_data,
        test_data=test_data,
        args=args,
        writer=writer,
    )
