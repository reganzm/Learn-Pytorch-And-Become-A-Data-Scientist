import torch

class Config:
    '''
    Chatbot模型参数
    '''
    corpus_data_path = './QNS_corpus.pth' #已处理的对话数据
    use_QA_first = True #是否载入知识库
    max_input_length = 30 #输入的最大句子长度
    max_generate_length = 30 #生成的最大句子长度
    prefix = 'checkpoints/chatbot'  #模型断点路径前缀
    model_ckpt  = 'checkpoints/chatbot_0630_1610' #加载模型路径
    '''
    训练超参数
    '''
    batch_size = 128
    shuffle = True #dataloader是否打乱数据
    num_workers = 0 #dataloader多进程提取数据
    bidirectional = True #Encoder-RNN是否双向
    hidden_size = 128
    embedding_dim = 300
    method = 'dot' #attention method
    dropout = 0.0 #是否使用dropout
    clip = 50.0 #梯度裁剪阈值
    num_layers = 2 #Encoder-RNN层数
    learning_rate = 1e-3
    #teacher_forcing比例
    teacher_forcing_ratio = 1.0 
    decoder_learning_ratio = 1.0
    drop_last = True
    '''
    训练周期信息
    '''
    epoch = 200
    print_every = 1 #每多少步打印一次
    save_every = 10 #没迭代多少Epoch保存一次模型
    '''
    GPU#是否使用gpu
    '''
    use_gpu = torch.cuda.is_available() 
    #使用GPU或CPU
    device = torch.device("cuda" if use_gpu else "cpu")
    #是否使用固定缓冲区
    pin_memory = True if(use_gpu) else False

if __name__=="__main__":
    print(Config.pin_memory)
