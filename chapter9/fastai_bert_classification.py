import numpy as np
import pandas as pd
from pathlib import Path
from typing import *
import torch
import torch.optim as optim
from fastai import *
from fastai.vision import *
from fastai.text import *
from fastai.callbacks import *
import torchsnooper




class Config(dict):
    """
    定义Config类，便于参数配置与更改
    继承自dict字典
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)
    def set(self, key, val):
        self[key] = val
        setattr(self, key, val)
        
config = Config(
    testing = False,
    bert_model_name="bert-base-chinese", 
    #Chinese Simplified and Traditional, 12-layer, 768-hidden, 12-heads, 110M parameters
    max_lr=3e-5,#学习率
    epochs=5,
    use_fp16=False, #fastai里可以方便地调整精度，加快训练速度：learner.to_fp16()
    bs=8,#batch size
    max_seq_len=128, #选取合适的seq_length，较大的值可能导致训练极慢报错等
)

from pytorch_pretrained_bert import BertTokenizer
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

#使用Bert分词器分词的适配器
class FastAiBertTokenizerAdapter(BaseTokenizer):
    """包装BertTokenizer为FastAI中的BaseTokenizer"""
    def __init__(self, tokenizer: BertTokenizer, max_seq_len: int=128, **kwargs):
        self._pretrained_tokenizer = tokenizer
        self.max_seq_len = max_seq_len
    def __call__(self, *args, **kwargs):
        return self
    def tokenizer(self, t:str) -> List[str]:
        """限制最大序列长度，使用Bert中的分词器将传入的序列进行分词，并在首位分别加上[CLS][SEP]标记"""
        return ["[CLS]"] + self._pretrained_tokenizer.tokenize(t)[:self.max_seq_len - 2] + ["[SEP]"]

#创建FastAI分词器实例，由分词器和规则组成，默认为SpacyTokenizer，SpacyTokenizer只支持英文，因此无法用于处理中文
fastai_tokenizer = Tokenizer(
    tok_func=FastAiBertTokenizerAdapter(bert_tokenizer, max_seq_len=config.max_seq_len), 
    pre_rules=[], 
    post_rules=[]
)

#设置FastAI Vocab
fastai_vocab = Vocab(list(bert_tokenizer.vocab.keys()))

import pandas as pd
df_train = pd.read_csv("./datas/train.csv",encoding='utf-8')
df_test = pd.read_csv("./datas/test.csv",encoding='utf-8')
#标签涵义：1代表正向评论，0代表负向评论
label_denotation = {1:'pos',0:'neg'}
df_train['label'] = df_train["label"].map(lambda x:0 if x=='neg' else 1)
from sklearn.model_selection import train_test_split

df_train, df_val = train_test_split(df_train,test_size=0.2,random_state=10)

df_test['label'] = df_test["label"].map(lambda x:0 if x=='neg' else 1)


def get_databunch():
    #建立TextDataBunch  
    databunch = TextClasDataBunch.from_df(".", df_train, df_val,df_test,
                      tokenizer=fastai_tokenizer,
                      vocab=fastai_vocab,
                      include_bos=False,
                      include_eos=False,
                      text_cols="sentence",
                      label_cols='label',
                      bs=config.bs,
                      collate_fn=partial(pad_collate, pad_first=False, pad_idx=0),
                      pin_memory=True,
                      num_workers = 1,
                      device=torch.device("cpu")
                 )   
    return databunch

def get_model():
    #model
    from pytorch_pretrained_bert.modeling import BertConfig, BertForSequenceClassification
    bert_model = BertForSequenceClassification.from_pretrained(config.bert_model_name, num_labels=2).cpu()
    return bert_model


def get_loss_fun():
    #损失函数：二分类问题选用CrossEntrypyLoss作为损失函数
    loss_func = nn.CrossEntropyLoss()
    return loss_func
    
def get_metrics():
    return [accuracy,AUROC(),error_rate]

def get_learner():
    databunch = get_databunch()
    bert_model = get_model()
    loss_func = get_loss_fun()
    #建立Learner(数据,预训练模型,损失函数)
    learner = Learner(databunch, bert_model,loss_func=loss_func,metrics=get_metrics())
    return learner

def train():
    learner = get_learner()
    #尝试寻找合适的最大学习率，这里使用了BERT原论文推荐的学习率3e-5作为默认值
    #learner.lr_find()
    #learner.recorder.plot(skip_end=20)
    #开始训练
    learner.fit_one_cycle(config.epochs, max_lr=config.max_lr)
    #模型保存
    learner.save('./fastai_bert_chinese_classification') 

def predict_learner():
    learner = get_learner()
    learner.load("./fastai_bert_chinese_classification")
    return learner
    

if __name__ == "__main__":
    #建立Learner(数据,预训练模型,损失函数)
    learner = get_learner()
    learner.load("./fastai_bert_chinese_classification")    
    #用样例测试下
    result = learner.predict("房间稍小，交通不便，专车往返酒店与浦东机场，车程10分钟，但是经常满员，不得不站在车里")
    print("predict result:{}".format(result))
    #在整个测试集上进行测试
    #tf_test_sentences = df_test["sentence"].values
    #df_test_labels = df_test["label"].values
    #import numpy as np
    #predict_labels = []
    #for sentence in tf_test_sentences:
    #    result = learner.predict(sentence)
    #    label = result[1].item()
    #    predict_labels.append(label)
    #correct = np.sum(df_test_labels==np.array(predict_labels))
    #acc = correct / df_test_labels.size
    #print("accuracy:{}".format(acc))
    #accuracy:0.926
    
