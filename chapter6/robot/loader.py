import torch
import torch.nn as nn
import itertools
from torch.utils.data import DataLoader,Dataset
from torch.nn import functional as F


def create_collate_fn(padding, eos):
    def collate_fn(corpus_item):
        #按照inputQue的长度进行排序,是调用pad_packed_sequence方法的要求
        corpus_item.sort(key=lambda p: len(p[0]), reverse=True) 
        inputs, targets, indexes = zip(*corpus_item)
        input_lengths = torch.tensor([len(line) for line in inputs])
        inputs = zeroPadding(inputs, padding)
        #词嵌入需要使用Long类型的Tensor
        inputs = torch.LongTensor(inputs)
        max_target_length = max([len(line) for line in targets])
        targets = zeroPadding(targets, padding)
        mask = binaryMatrix(targets, padding)
        mask = torch.ByteTensor(mask)
        targets = torch.LongTensor(targets)
        return inputs, targets, mask, input_lengths, max_target_length, indexes
    return collate_fn

def zeroPadding(datas, fillvalue):
    return list(itertools.zip_longest(*datas, fillvalue=fillvalue))

def binaryMatrix(datas, padding):
    m = []
    for i, seq in enumerate(datas):
        m.append([])
        for token in seq:
            if token == padding:
                m[i].append(0)
            else:
                m[i].append(1)
    return m


class CorpusDataset(Dataset):
    def __init__(self, opts):
        self.opts = opts
        self.datas = torch.load(opts.corpus_data_path)
        self.word2ix = self.datas['word2ix']
        self.ix2word = self.datas['ix2word']
        self.corpus = self.datas['corpus']
        self.padding = self.word2ix.get(self.datas.get('padding'))
        self.eos = self.word2ix.get(self.datas.get('eos'))
        self.sos = self.word2ix.get(self.datas.get('sos'))
        self.unknown = self.word2ix.get(self.datas.get('unknown'))
    def __getitem__(self, index):
        #问
        inputQue = self.corpus[index][0]
        #答
        targetAns = self.corpus[index][1]
        return inputQue,targetAns, index
    def __len__(self):
        return len(self.corpus)

def get_loader(opts):
    dataset = CorpusDataset(opts)
    dataloader = DataLoader( dataset,
                             batch_size=opts.batch_size,
                             shuffle=opts.shuffle,
                             num_workers=opts.num_workers, 
                             drop_last=opts.drop_last,
                             collate_fn=create_collate_fn(dataset.padding, dataset.eos),
                             pin_memory=opts.pin_memory)
    return dataloader,dataset


#dataloader,dataset = get_loader(opts)

#for i in dataloader:
#    print(i)




