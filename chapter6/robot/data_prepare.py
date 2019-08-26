import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
import jieba
import re

#未录入字符替换符
unknown = '</UNK>' 
#句子结束符
eos = '</EOS>' 
#句子开始符
sos = '</SOS>' 
#句子填充符
padding = '</PAD>' 
#字典最大长度
max_voc_length = 50000 
#加入字典的词的词频最小值
min_freq = 1
#最大句子长度
max_sentence_length = 50 
#已处理的对话数据集保存路径
save_path = 'QNS_corpus.pth' 
#中文英文处理正则
reg = re.compile("[^\u4e00-\u9fa5^a-z^A-Z^0-9]")

def datapreparation():
    '''处理对话数据集'''
    data = []
    lines = np.load("./datas/CN-corpus.npy")
    for line in lines:
        values = line.split('|')
        sentences = []
        for value in values:
            sentence = jieba.lcut(reg.sub("",value))
            #每句话的结束，添加</EOS>标记
            sentence = sentence[:max_sentence_length] + [eos]
            sentences.append(sentence)
        data.append(sentences)
    '''生成字典和句子索引'''
    words_dict = {} #统计单词的词频
    def update(word):
        words_dict[word] = words_dict.get(word, 0) + 1
    #更新词典
    {update(word) for sentences in data for sentence in sentences for word in sentence}
    #按词频从高到低排序
    word_nums_list = sorted([(num, word) for word, num in words_dict.items()], reverse=True)
    #词典最大长度: max_voc_length 最小单词词频: min_freq
    words = [word[1] for word in word_nums_list[:max_voc_length] if word[0] >= min_freq]
    words = [unknown, padding, sos] + words
    word2ix = {word: ix for ix, word in enumerate(words)}
    ix2word = {ix: word for word, ix in word2ix.items()}
    #使用构建的词典对原对话语料进行编码
    ix_corpus = [[[word2ix.get(word, word2ix.get(unknown)) for word in sentence] for sentence in item] for item in data]
    clean_data = {
        'corpus': ix_corpus, 
        'word2ix': word2ix,
        'ix2word': ix2word,
        'unknown' : '</UNK>',
        'eos' : '</EOS>',
        'sos' : '</SOS>',
        'padding': '</PAD>',
    }
    torch.save(clean_data, save_path)
    return words_dict
if __name__ == "__main__":
    datapreparation()
