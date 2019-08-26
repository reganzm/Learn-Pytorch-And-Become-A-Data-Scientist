import torch
import torch.nn as nn
import torch.nn.functional as F
class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,batch_size):
        super(RNNCell, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden
    
    def initHidden(self):
        return torch.zeros(self.batch_size, self.hidden_size)
    
#隐藏层神经元个数
n_hidden = 10
#分类类别数
target_size = 2
#输入长度，以字符英文字符为例，英文字符个数为26
input_size=26
#batch_size
batch_size=64
#实例化RNN
rnnCell = RNNCell(input_size, n_hidden, target_size,batch_size)
#初始化隐藏状态，初始为0
hidden = rnnCell.initHidden()
#构造输入,随机生成0到10，形状为[64,26]
input = torch.randint(0,10,(batch_size,input_size)).float()
print(input)
print(input.shape)
output, next_hidden = rnnCell(input, hidden)#得到[20*2,20*10]
print(output.data.size(),next_hidden.data.size())

