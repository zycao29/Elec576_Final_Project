import numpy as np
import os
import sys
import argparse
from sklearn.utils import shuffle
import torch
import torch.nn as nn
from models.training import *
from models.model import *
from utils.dataloader import MyDataset

from torch import optim
from torch.utils.data import  Dataset, DataLoader, TensorDataset
##

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,     default=10000)
    parser.add_argument('--data_dir', type=str, default="./data")
    parser.add_argument('--train_data_file', type=str, default="mitbih_train.csv")
    parser.add_argument('--test_data_file', type=str, default="mitbih_test.csv")
    parser.add_argument('--model_para', type=str, default="./experiments")
    parser.add_argument('--num_epoches',type=int,     default=10)
    parser.add_argument('--lr',type=float,            default=0.001)
    parser.add_argument('--decay_rate',type=float,    default=0.1)
    parser.add_argument('--step_size',type=int,       default=10)
    parser.add_argument('--batch_size',type=int,      default=100)
    parser.add_argument('--cuda_device', type=str,    default="cuda:0")
    parser.add_argument('--model', type=str, default="mlp")
    parser.add_argument('--input_length',type=int,    default=187)
    parser.add_argument('--fine_tune',default = False, action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    dataloaders={
        'train':DataLoader(MyDataset(os.path.join(args.data_dir, args.train_data_file), is_training=True, args=args), batch_size=args.batch_size, shuffle=True ),
        'test': DataLoader(MyDataset(os.path.join(args.data_dir, args.test_data_file), is_training=False, args=args), batch_size=args.batch_size, shuffle=False)
    }
   
    if args.model == "mlp":
        model = model_mlp(args.input_length)
    elif args.model == "cnn":
        model = model_conv(args.input_length, 1)
    else:
        print("Undefined")

    device = torch.device(args.cuda_device if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    params_to_update = model.parameters()
    optimizer = optim.Adam(params_to_update, lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.decay_rate)
    lr_scheduler.step();   
    criterion = nn.BCEWithLogitsLoss()
    if args.fine_tune == False:
        train_model(model, dataloaders, criterion, optimizer, lr_scheduler, device, args);
    else:
        train_model_fine_tune(model, dataloaders, criterion, optimizer, lr_scheduler, device, args);

if __name__ == "__main__":
    main()