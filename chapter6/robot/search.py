import torch
import torch.nn as nn



class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, sos, eos, input_seq, input_length, max_length, device):
        # Encoder的Forward计算 
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # 把Encoder最后时刻的隐状态作为Decoder的初始值
        decoder_hidden = encoder_hidden[:self.decoder.num_layers]
        # Decoder的初始输入是SOS
        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * sos
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device)
        # 搜索，直到遇到EOS结束符或者达到最大解码长度
        for _ in range(max_length):
            # 单步执行Decoder Forward
            #decodr shape:(batch,vob_size)=(1,vob_size)
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            # 返回概率最大的词的概率和得分
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # 用unsqueeze增加batch维度。
            if decoder_input.item() == eos:
                break
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # 返回所有的词和得分。
        return all_tokens, all_scores