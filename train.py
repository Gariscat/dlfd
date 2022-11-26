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


def one_epoch(
    model: nn.Module,
    loader: DataLoader,
    mode: str,
    device: torch.device = DEVICE,
    epoch_id: int = None,
    loss_cls: any = nn.MSELoss,
    optimizer: any = optim.SGD,
):
    def run(mode, loss_cls=loss_cls, device=device):
        criterion = loss_cls()
        losses = []
        fp_cnt, td_tot = 0, 0
        length = 0
        for batch in tqdm(loader, desc='Epoch '+str(epoch_id+1)+' '+mode):
            inputs, targets = batch[0].to(device), batch[1].to(device)
            length += inputs.shape[0]

            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            
            loss = criterion(outputs, targets)
            fp_cnt += torch.sum(preds < targets).item()
            detection_time = torch.maximum(
                input=preds-targets,
                other=torch.zeros_like(preds)
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
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = EPOCHS,
    device: torch.device = DEVICE,
    writer: SummaryWriter = None,
    evaluate_first: bool = True
):
    model.to(device)
    epoch = -1
    if evaluate_first:
        evaluate(model, device, val_loader, 'val', epoch)
        
    for epoch in range(epochs):
        train_ret = one_epoch(model, train_loader, 'train', device, epoch)

        print('Epoch {}: '.format(epoch+1))
        print(f"train loss: {train_ret['loss']}")
        print(f"train fpr: {train_ret['fpr']}")
        print(f"train td_avg: {train_ret['td_avg']}")

        val_ret = evaluate(model, device, val_loader, 'val', epoch)
        
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
            

def evaluate(model, device=DEVICE, loader=None, comment='eval', epoch_id=None):
    if loader is None:
        return None
    
    model = model.to(device)
    val_ret = one_epoch(model, loader, 'eval', device, epoch_id)

    return val_ret


if __name__ == '__main__':
    train_set, val_set = get_data(
        trace_path='./traces/trace.log',
        source_id=3,
        obs_ord=3,
        scale=1e8,
        seq_len=1024
    )
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)
    
    model = RNNPredictor(rnn=nn.GRU)
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        evaluate_first=True
    )