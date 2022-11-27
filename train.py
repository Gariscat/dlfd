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
from networks import RNNPredictor
import argparse

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

EPOCHS = 5
LR = 1e-3
BATCH_SIZE = 32
DATA_PATH = './traces/'
SEQ_LEN = 256


def one_epoch(
    model: nn.Module,
    data: np.ndarray,
    mode: str,
    device: torch.device = DEVICE,
    epoch_id: int = None,
    loss_cls: any = nn.MSELoss,
    optimizer: any = optim.SGD,
    seq_len: int = SEQ_LEN,
):
    def run(mode, device=device, loss_cls=loss_cls):
        criterion = loss_cls()
        losses = []
        fp_cnt, td_tot = 0, 0
        length = 0
        for r in trange(1, data.shape[0], desc='Epoch '+str(epoch_id+1)+' '+mode):
            # inputs, targets = batch[0].to(device), batch[1].to(device)
            l = max(0, r-seq_len)
            source = torch.from_numpy(data[l:r])
            left_zeros = torch.zeros(seq_len-source.shape[0], source.shape[1])
            source = torch.cat((left_zeros, source), dim=0).unsqueeze(0).float().to(device)
            target = torch.tensor([data[r][0]]).reshape(1, 1).float().to(device)

            length += source.shape[0]

            output = model(source)
            pred = torch.argmax(output, dim=1)
            
            loss = criterion(output, target)
            fp_cnt += torch.sum(pred < target).item()
            detection_time = torch.maximum(
                input=pred-target,
                other=torch.zeros_like(pred)
            )
            td_tot += torch.sum(detection_time).item()
            losses.append(loss.item())

            if mode == 'train':
                opt.zero_grad()
                loss.backward()
                opt.step()
                
        return {'loss': np.mean(losses), 'fpr': fp_cnt / length, 'td_avg': td_tot / length}

    if mode == 'train':
        opt = optimizer(model.parameters(), lr=LR)
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
    epochs: int = EPOCHS,
    device: torch.device = DEVICE,
    writer: SummaryWriter = None,
    evaluate_first: bool = True
):
    model.to(device)
    epoch = -1
    if evaluate_first:
        one_epoch(model, test_data, 'val', device, epoch)
        
    for epoch in range(epochs):
        train_ret = one_epoch(model, train_data, 'train', device, epoch)

        print('Epoch {}: '.format(epoch+1))
        print(f"train loss: {train_ret['loss']}")
        print(f"train fpr: {train_ret['fpr']}")
        print(f"train td_avg: {train_ret['td_avg']}")

        val_ret = one_epoch(model, test_data, 'val', device, epoch)
        
        print(f"eval loss: {val_ret['loss']}")
        print(f"eval fpr: {val_ret['fpr']}")
        print(f"eval td_avg: {val_ret['td_avg']}")
        
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
    train_data, test_data = get_data(
        trace_path='./traces/trace.log',
        source_id=3,
        obs_ord=3,
        scale=1e8,
    )
    model = RNNPredictor(rnn=nn.LSTM)
    train(
        model=model,
        train_data=train_data,
        test_data=test_data,
        evaluate_first=True
    )