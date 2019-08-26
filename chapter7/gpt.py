### First, tokenize the input
#############################
import torch
tokenizer = torch.hub.load('huggingface/pytorch-pretrained-BERT', 'openAIGPTTokenizer', 'openai-gpt')

text = "I Love "
tokenized_text = tokenizer.tokenize(text)
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
tokens_tensor = torch.tensor([indexed_tokens])


#使用openAIGPTModel模型计算隐状态
model = torch.hub.load('huggingface/pytorch-pretrained-BERT', 'openAIGPTModel', 'openai-gpt')
#转换为测试模式
model.eval()
#计算隐状态
with torch.no_grad():
	hidden_states = model(tokens_tensor)


#使用openAIGPTLMHeadModel模型对下一个词进行预测
lm_model = torch.hub.load('huggingface/pytorch-pretrained-BERT', 'openAIGPTLMHeadModel', 'openai-gpt')
lm_model.eval()
#得到预测值
with torch.no_grad():
	predictions = lm_model(tokens_tensor)
#取出最可能的词
predicted_index = torch.argmax(predictions[0][0, -1, :]).item()
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
