import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

bins = 50           # RNN时间步长
input_dim = 1       # RNN输入尺寸
lr = 0.01         # 初始学习率
epochs = 2000      # 轮数
hidden_size=32      # 隐藏层神经元个数
num_layers = 2      # 神经元层数
nonlinearity="relu" #只支持relu和tanh

class RNNDemo(nn.Module):
    def __init__(self,input_dim,hidden_size,num_layers,nonlinearity):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.nonlinearity = nonlinearity
        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity=nonlinearity
        )
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, x, h):
        r_out, h_state = self.rnn(x,h)
        outs = [] 
        for time_step in range(r_out.size(1)):
            outs.append(self.out(r_out[:, time_step, :]))
        return torch.stack(outs, dim=1), h_state



rnnDemo = RNNDemo(input_dim,hidden_size,num_layers,nonlinearity).cuda()
optimizer = torch.optim.Adam(rnnDemo.parameters(), lr=lr)
loss_func = nn.MSELoss()

h_state = None
for step in range(epochs):
    start, end = step * np.pi, (step + 1) * np.pi  # 时间跨度
    # 使用Sin函数预测Cos函数
    steps = np.linspace(start, end, bins, dtype=np.float32, endpoint=False)
    x_np = np.sin(steps)
    y_np = np.cos(steps)
    x = torch.from_numpy(x_np).unsqueeze(1).unsqueeze(2).cuda()#【100，1,1】尺寸大小为(time_step, batch, input_size)
    y = torch.from_numpy(y_np).unsqueeze(1).unsqueeze(2).cuda()#【100，1,1】
    prediction, h_state = rnnDemo(x, h_state)  # RNN输出（预测结果，隐藏状态）
    #将每一次输出的中间状态传递下去(不带梯度)
    h_state = h_state.detach()  
    loss = loss_func(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if(step%100==0):
        print("loss:{:.8f}".format(loss))
plt.scatter(steps,y_np,marker="^")
plt.scatter(steps, prediction.cpu().data.numpy().flatten(),marker=".")
plt.show()