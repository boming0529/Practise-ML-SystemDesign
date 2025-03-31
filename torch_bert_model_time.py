import time
from transformers import BertTokenizer, BertModel
import torch

# initialization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# text = "a person walk with his dog in montreal"
text = "a person walk with his dog in montreal " * 500  # long
# text = ["a person walk with his dog in montreal" * 10] * 32  # batch size = 32

def check_char_mps(text, max_length=512):
    tokens = tokenizer.tokenize(text)
    chunks = [tokens[i:i + (max_length - 2)] for i in range(0, len(tokens), max_length - 2)]  # -2 reserve [CLS] and [SEP] two words
    inputs = [tokenizer.encode_plus(chunk, return_tensors="pt", max_length=max_length, truncation=True).to('mps') for chunk in chunks]
    return inputs

def check_char_cpu(text, max_length=512):
    tokens = tokenizer.tokenize(text)
    chunks = [tokens[i:i + (max_length - 2)] for i in range(0, len(tokens), max_length - 2)]  # -2 reserve [CLS] and [SEP] two words
    inputs = [tokenizer.encode_plus(chunk, return_tensors="pt", max_length=max_length, truncation=True).to('cpu') for chunk in chunks]
    return inputs

chunks_cpu = check_char_cpu(text)
chunks_mps = check_char_mps(text)

# CPU
model_cpu = BertModel.from_pretrained("bert-base-uncased").to("cpu")
# token_ids_cpu = tokenizer(text, return_tensors="pt").to("cpu")
token_ids_cpu = tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to("cpu") # truncation
start = time.time()
# model_cpu(**token_ids_cpu) # no using chunk function
outputs = [model_cpu(**chunk) for chunk in chunks_cpu]
print("CPU time:", time.time() - start)

# MPS
model_mps = BertModel.from_pretrained("bert-base-uncased").to('mps')
# token_ids_mps = tokenizer(text, return_tensors="pt").to('mps')
token_ids_mps = tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to('mps') # truncation
start = time.time()
# model_mps(**token_ids_mps) # no using chunk function
outputs = [model_mps(**chunk) for chunk in chunks_mps]
print("MPS time:", time.time() - start)


