import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import argparse
import os

def train(rank, args, model, device, dataloader_kwargs):
    #手动设置随机种子
    torch.manual_seed(args.seed + rank)
    #加载训练数据
    train_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=True, download=True,transform=transforms.Compose([
        transforms.ToTensor(),
                        transforms.Normalize((0.,), (1.,))
                        ])),batch_size=args.batch_size, shuffle=True, num_workers=1,**dataloader_kwargs)
    #使用随机梯度下降进行优化
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    #开始训练，训练epoches次
    for epoch in range(1, args.epochs + 1):
        train_epoch(epoch, args, model, device, train_loader, optimizer)

def test(args, model, device, dataloader_kwargs):
    #设置随机种子
    torch.manual_seed(args.seed)
    #加载测试数据
    test_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
            transforms.Normalize((0.,), (1.,))
            ])),batch_size=args.batch_size, shuffle=True, num_workers=1,**dataloader_kwargs)
    #运行测试
    test_epoch(model, device, test_loader)


def train_epoch(epoch, args, model, device, data_loader, optimizer):
    #模型转换为训练模式
    model.train()
    pid = os.getpid()
    for batch_idx, (data, target) in enumerate(data_loader):
        #优化器梯度置0
        optimizer.zero_grad()
        #输入特征预测值
        output = model(data.to(device))
        #预测值与标准值计算损失
        loss = F.nll_loss(output, target.to(device))
        #计算梯度
        loss.backward()
        #更新梯度
        optimizer.step()
        #每10步打印一下日志
        if batch_idx % 10 == 0:
            print('{}\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(pid, epoch, batch_idx * len(data), len(data_loader.dataset),
                                                                               100. * batch_idx / len(data_loader), loss.item()))


def test_epoch(model, device, data_loader):
    #模型转换为测试模式
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            output = model(data.to(device))
            #将每个批次的损失加起来
            test_loss += F.nll_loss(output, target.to(device), reduction='sum').item()
            #得到概率最大的索引,
            pred = output.max(1)[1]
            #预测的索引和目标索引相同，认为预测正确
            correct += pred.eq(target.to(device)).sum().item()

    test_loss /= len(data_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(data_loader.dataset),
                                                                                 100. * correct / len(data_loader.dataset)))
