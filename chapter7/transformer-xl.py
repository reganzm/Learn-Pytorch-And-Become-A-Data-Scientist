### First, tokenize the input
#############################
import torch
tokenizer = torch.hub.load('huggingface/pytorch-pretrained-BERT', 'transformerXLTokenizer', 'transfo-xl-wt103')

#  Prepare tokenized input
text_1 = "Who was Jim Henson ?"
text_2 = "Jim Henson was a puppeteer"
tokenized_text_1 = tokenizer.tokenize(text_1)
tokenized_text_2 = tokenizer.tokenize(text_2)
indexed_tokens_1 = tokenizer.convert_tokens_to_ids(tokenized_text_1)
indexed_tokens_2 = tokenizer.convert_tokens_to_ids(tokenized_text_2)
tokens_tensor_1 = torch.tensor([indexed_tokens_1])
tokens_tensor_2 = torch.tensor([indexed_tokens_2])

### Get the hidden states computed by `transformerXLModel`
##########################################################
model = torch.hub.load('huggingface/pytorch-pretrained-BERT', 'transformerXLModel', 'transfo-xl-wt103')
model.eval()

# Predict hidden states features for each layer
# past can be used to reuse precomputed hidden state in a subsequent predictions
with torch.no_grad():
    hidden_states_1, mems_1 = model(tokens_tensor_1)
    hidden_states_2, mems_2 = model(tokens_tensor_2, mems=mems_1)

### Predict the next token using `transformerXLLMHeadModel`
###########################################################
lm_model = torch.hub.load('huggingface/pytorch-pretrained-BERT', 'transformerXLLMHeadModel', 'transfo-xl-wt103')
lm_model.eval()

# Predict hidden states features for each layer
with torch.no_grad():
    predictions_1, mems_1 = lm_model(tokens_tensor_1)
    predictions_2, mems_2 = lm_model(tokens_tensor_2, mems=mems_1)

# Get the predicted last token
predicted_index = torch.argmax(predictions_2[0, -1, :]).item()
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
assert predicted_token == 'who'