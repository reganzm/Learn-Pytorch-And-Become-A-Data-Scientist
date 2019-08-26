import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

bins = 100           # RNN时间步长
input_dim = 1       # RNN输入尺寸
lr = 0.01         # 初始学习率
epochs = 2000      # 轮数
hidden_size=32      # 隐藏层神经元个数
num_layers = 2      # 神经元层数

class LstmDemo(nn.Module):
    def __init__(self,input_dim,hidden_size,num_layers):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers
        )
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, x, h_0_c_0):
        r_out, h_state = self.lstm(x,h_0_c_0)
        outs = [] 
        #r_out:(h_n, c_n)
        for time_step in range(r_out.size(1)):
            outs.append(self.out(r_out[:, time_step, :]))
        return torch.stack(outs, dim=1), h_state



lstmDemo = LstmDemo(input_dim,hidden_size,num_layers).cuda()
optimizer = torch.optim.Adam(lstmDemo.parameters(), lr=lr)
loss_func = nn.MSELoss()

h_c_state = (torch.zeros(num_layers,1,hidden_size).cuda(),torch.zeros(num_layers,1,hidden_size).cuda())
for step in range(epochs):
    start, end = step * np.pi, (step + 1) * np.pi  # 时间跨度
    # 使用Sin函数预测Cos函数
    steps = np.linspace(start, end, bins, dtype=np.float32, endpoint=False)
    x_np = np.sin(steps)
    y_np = np.cos(steps)
    x = torch.from_numpy(x_np).unsqueeze(1).unsqueeze(2).cuda()#【100，1,1】尺寸大小为(time_step, batch, input_size)
    y = torch.from_numpy(y_np).unsqueeze(1).unsqueeze(2).cuda()#【100，1,1】
    prediction, h_state = lstmDemo(x,h_c_state)  # RNN输出（预测结果，隐藏状态）
    #将每一次输出的中间状态传递下去(不带梯度)
    h_c_state = (h_state[0].detach(),h_state[1].detach())
    loss = loss_func(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if(step%100==0):
        print("loss:{:.8f}".format(loss))
plt.scatter(steps,y_np,marker="^")
plt.scatter(steps, prediction.cpu().data.numpy().flatten(),marker=".")
plt.show()