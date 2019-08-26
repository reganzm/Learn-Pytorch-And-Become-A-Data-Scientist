import numpy as np
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
import torch
import random 
from train import train

class LinearRegression(torch.nn.Module):
    def __init__(self):
        super(LinearRegression,self).__init__()
        #输入输出都是一维
        self.linear = torch.nn.Linear(1,1)
    def forward(self,x):
        return self.linear(x)


if __name__ == "__main__":
    mp.set_start_method('spawn')
    x = np.arange(20)
    y = np.array([5*x[i]+random.randint(1,20) for i in range(len(x))])
    x_train = torch.from_numpy(x).float()
    y_train = torch.from_numpy(y).float()   
    
    #新建模型，误差函数，优化器
    model = LinearRegression()
    model.share_memory()
    
    processes = []
    for rank in range(10):
        p = mp.Process(target=train, args=(x_train,y_train,model))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()    
    
    #预测一波    
    input_data = x_train.unsqueeze(1)
    predict = model(input_data)
    plt.xlabel("X")
    plt.ylabel("Y")    
    plt.plot(x_train.data.numpy(),predict.squeeze(1).data.numpy(),"r")
    plt.scatter(x_train,y_train)
    
    plt.show()
    
   


