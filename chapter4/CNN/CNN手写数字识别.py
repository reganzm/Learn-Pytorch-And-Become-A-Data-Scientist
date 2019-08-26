import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
import torch.multiprocessing as mp
import argparse
import os
from train import train,test
from model import CNN_Net

# 解析传入的参数
parser = argparse.ArgumentParser(description='PyTorch MNIST')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--num-processes', type=int, default=6, metavar='N',
                    help='how many training processes to use (default: 1)')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='enables CUDA training')
if __name__=="__main__":
    #解析参数
    args = parser.parse_args()
    #判断是否使用GPU
    use_cuda = args.cuda and torch.cuda.is_available()
    #运行时设备
    device = torch.device("cuda" if use_cuda else "cpu")
    #使用固定缓冲区
    dataloader_kwargs = {'pin_memory': True} if use_cuda else {}
    #多进程训练，windows使用spawn方式
    mp.set_start_method('spawn')
    #模型拷贝到设备
    model = CNN_Net().to(device)
    #多进程共享模型参数
    model.share_memory()

    processes = []
    for rank in range(args.num_processes):
        p = mp.Process(target=train, args=(rank,args, model,device, dataloader_kwargs))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    #测试模型
    test(args, model, device, dataloader_kwargs)
