import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
import os
def foo(q):  #传递队列对象给函数
    pid = os.getpid()
    q.put('my pid is:{}'.format(pid))
    print(pid)

if __name__ == '__main__':
    #设置启动进程方式,windows下默认为spawn,linux下为fork
    mp.set_start_method('spawn') 
     #创建队列对象
    q = mp.Queue() 
    ps = []
    #创建10个进程，传递运行函数和参数
    [ps.append(mp.Process(target=foo, args=(q,))) for i in range(10)]
    #启动进程
    [p.start() for p in ps]
    #join方法让主线程阻塞，等待子线程执行完成再执行
    [p.join() for p in ps]
    #获取队列数据
    data = q.get()
    while(data):
        print(data)
        data = q.get()