from transformers import BertTokenizer, BertModel
import torch

# Reference : https://huggingface.co/google-bert/bert-base-uncased
# initialization
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased").to(device)

text = "Hello world"
token_ids = tokenizer(text, return_tensors="pt").to(device) # turn string into tensor
outputs = model(**token_ids) 
print(token_ids['input_ids'])  # tensor([[101, 7592, 2088, 102]])