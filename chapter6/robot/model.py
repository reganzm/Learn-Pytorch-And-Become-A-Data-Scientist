import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence

class EncoderRNN(nn.Module):
    def __init__(self, opts, voc_length):
        super(EncoderRNN, self).__init__()
        self.num_layers = opts.num_layers
        self.hidden_size = opts.hidden_size
        self.embedding = nn.Embedding(voc_length, opts.embedding_dim)
        #双向GRU作为Encoder
        self.gru = nn.GRU(opts.embedding_dim, self.hidden_size, self.num_layers,dropout= opts.dropout, bidirectional=opts.bidirectional)
    def forward(self, input_seq, input_lengths, hidden=None):
        """
        input_seq:[max_seq_length,batch_size]
        input_lengths:the lengths in batchs
        """
        embedded = self.embedding(input_seq) 
        #packed data shape:[all_words_size_in_batch,embedding_size]
        packed = pack_padded_sequence(embedded, input_lengths)
        #outputs data shape:[all_words_size_in_batch,num_layer*hidden_size] 
        #hidden shape:[num_layer*bidirection,batch_size,hidden_size]
        outputs, hidden = self.gru(packed, hidden)
        #outputs shape:[max_seq_length,batch_size,num_layer*hidden_size]
        outputs, _ = pad_packed_sequence(outputs)
        #将双向的outputs求和
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
        return outputs, hidden


class Attention(torch.nn.Module):
    def __init__(self, attn_method, hidden_size):
        super(Attention, self).__init__()
        #Attention的方式：dot和general
        self.method = attn_method 
        self.hidden_size = hidden_size
        if self.method not in ['dot', 'general']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        if self.method == 'general':
            self.attn = torch.nn.Linear(self.hidden_size, self.hidden_size)
    #dot方式
    def dot_score(self, hidden, encoder_outputs):
        """
        hidden shape:[1,batch_size,hidden_size]
        encoder_outputs shape:[max_seq_length,batch_size,hidden_size]
        result shape:[max_seq_length,batch_size]
        """
        return torch.sum(hidden * encoder_outputs, dim=2)
    #general方式
    def general_score(self, hidden, encoder_outputs):
        energy = self.attn(encoder_outputs)
        return torch.sum(hidden * energy, dim=2)
    #前向传播
    def forward(self, hidden, encoder_outputs):
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)
        attn_energies = attn_energies.t()#[batch_size,max_seq_length]
        return F.softmax(attn_energies, dim=1).unsqueeze(1)#[batch_size,1,max_seq_length]

class AttentionDecoderRNN(nn.Module):
    def __init__(self, opts, voc_length):
        super(AttentionDecoderRNN, self).__init__()
        self.attn_method = opts.method
        self.hidden_size = opts.hidden_size
        self.output_size = voc_length
        self.num_layers = opts.num_layers
        self.dropout = opts.dropout
        self.embedding = nn.Embedding(voc_length, opts.embedding_dim)
        self.embedding_dropout = nn.Dropout(self.dropout)
        self.gru=nn.GRU(opts.embedding_dim,self.hidden_size,self.num_layers, dropout = self.dropout )
        self.concat = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.attention = Attention(self.attn_method, self.hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        """
        input_step shape:[1,batch_size]
        embedded shape:[1,batch_size,embedding_size]
        
        """
        embedded = self.embedding(input_step) 
        embedded = self.embedding_dropout(embedded)
        #rnn_output shape:[1,batch_size,hidden_size]
        #hideen shape:[num_layer*bidirection,batch_size,hidden_size]
        rnn_output, hidden = self.gru(embedded, last_hidden)
        #注意力权重#[batch_size,1,max_seq_length]
        attn_weights = self.attention(rnn_output, encoder_outputs)
        #由注意力权重通过bmm批量矩阵相乘计算出此时rnn_output对应的注意力Context
        #context shape:[batch_size,1,hidden_size]
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        #上下文和rnn_out拼接
        concat_input = torch.cat((rnn_output, context), 1)
        #使用tanh非线性函数将值范围变成[-1,1]
        concat_output = torch.tanh(self.concat(concat_input))  
        output = self.out(concat_output)
        #Softmax函数计算output的得分值
        output = F.softmax(output, dim=1)
        return output, hidden
