
import torch
#下载BERT分词器
tokenizer = torch.hub.load('huggingface/pytorch-pretrained-BERT', 'bertTokenizer', 'bert-base-chinese', do_basic_tokenize=False)
#构建输入
text = "[CLS]北京天安门。[SEP]四川成都。[SEP]"
tokenized_text = tokenizer.tokenize(text)
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

### Get the hidden states computed by `bertModel`
# Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
segments_ids = [0, 0, 0, 0, 0, 0,0,0,1, 1, 1, 1, 1, 1]

# Convert inputs to PyTorch tensors
segments_tensors = torch.tensor([segments_ids])
tokens_tensor = torch.tensor([indexed_tokens])

model = torch.hub.load('huggingface/pytorch-pretrained-BERT', 'bertModel', 'bert-base-chinese')
model.eval()

with torch.no_grad():
    encoded_layers, _ = model(tokens_tensor, segments_tensors)


    ### Predict masked tokens using `bertForMaskedLM`
    # Mask a token that we will try to predict back with `BertForMaskedLM`
    masked_index = 12
    tokenized_text[masked_index] = '[MASK]'
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])
    
    maskedLM_model = torch.hub.load('huggingface/pytorch-pretrained-BERT', 'bertForMaskedLM', 'bert-base-chinese')
    maskedLM_model.eval()
    
    with torch.no_grad():
        predictions = maskedLM_model(tokens_tensor, segments_tensors)
    
    # Get the predicted token
    predicted_index = torch.argmax(predictions[0][0, masked_index]).item()
    predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
    #assert predicted_token == 'Jim'

### Classify next sentence using ``bertForNextSentencePrediction``
# Going back to our initial input
#text = "[CLS]四川的省会是? [SEP]成都[SEP]"
text = "[CLS]四川的省会是? [SEP]滚蛋[SEP]"
tokenized_text = tokenizer.tokenize(text)
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
tokens_tensor = torch.tensor([indexed_tokens])
segments_ids = [0, 0, 0, 0, 0, 0,0,0,0,1, 1, 1]
# Convert inputs to PyTorch tensors
segments_tensors = torch.tensor([segments_ids])
nextSent_model = torch.hub.load('huggingface/pytorch-pretrained-BERT', 'bertForNextSentencePrediction', 'bert-base-chinese')
nextSent_model.eval()

# Predict the next sentence classification logits
with torch.no_grad():
    next_sent_classif_logits = nextSent_model(tokens_tensor, segments_tensors)

### Fine-tune BERT using `bertForPreTraining`
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])

forPretraining_model = torch.hub.load('huggingface/pytorch-pretrained-BERT', 'bertForPreTraining', 'bert-base-cased')
masked_lm_logits_scores, seq_relationship_logits = forPretraining_model(tokens_tensor, segments_tensors)

### Fine-tune BERT using `bertForPreTraining`
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])
#large case
forPretraining_model = torch.hub.load('huggingface/pytorch-pretrained-BERT', 'bertForPreTraining', 'bert-large-cased')
masked_lm_logits_scores, seq_relationship_logits = forPretraining_model(tokens_tensor, segments_tensors)