import os
import torch
import torch.optim as optim

def train(x_train,y_train,model):
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(),0.001)        
    #开始训练
    num_epochs = 100000
    for i in range(num_epochs):
        input_data = x_train.unsqueeze(1)
        target = y_train.unsqueeze(1)
        out = model(input_data)
        loss = criterion(out,target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("PID:{},Epoch:[{}/{}],loss:[{:.4f}]".format(os.getpid(),i+1,num_epochs,loss.item()))