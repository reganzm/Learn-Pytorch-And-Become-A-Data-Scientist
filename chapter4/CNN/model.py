import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_Net(nn.Module):
    def __init__(self):
        super(CNN_Net,self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=7)
        self.conv2 = nn.Conv2d(8, 32, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(288, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 288)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(x, dim=1)
    
    
class CNN_Dilation_Net(nn.Module):
    def __init__(self):
        super(CNN_Dilation_Net,self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=7,dilation=2)
        self.conv2 = nn.Conv2d(8, 32, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(128, 50)
        self.fc2 = nn.Linear(50, 10)
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 128)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(x, dim=1)
