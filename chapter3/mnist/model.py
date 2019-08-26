import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet,self).__init__()
        self.layer1 = nn.Linear(784,50)
        self.layer2 = nn.Linear(50,10)
    def forward(self,x):
        #输入层到隐藏层，使用tanh激活函数
        x = self.layer1(x.reshape(-1,784))
        x = torch.tanh(x)
        #隐藏层到输出层，使用relu激活函数
        x = self.layer2(x)
        x = F.relu(x)
        #log(softmax)操作，使用NLLLoss损失函数
        x = F.log_softmax(x,dim=1)
        return x