import string
import unicodedata
from nltk.stem import WordNetLemmatizer
# from nltk.stem.porter import PorterStemmer # if need using PorterStemmer
from transformers import BertTokenizer
from nltk import pos_tag, word_tokenize
import torch

# case1.
raw_text = 'A person walking with his dog in Montréal !'
# case2 the same semantic.
# raw_text = 'a person walks with his dog, in Montréal.'

## text normalization
# lowercasing
text = raw_text.lower()
print('After lowercasing:', text)

# removing punctuation
text = text.translate(str.maketrans('','', string.punctuation))
print('After removing punctuation:', text)

# Trimming Whitespace
text = " ".join(text.split())
print('After trimming whitespace:', text)

# NFKD (Removing Accents/Diacritics) or Unicode normalization
text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
print('After NFKD normalization:', text)

# Lemmatization
def get_wordnet_pos(tag):
    if tag.startswith('V'):    # verb
        return 'v'
    elif tag.startswith('N'):  # n
        return 'n'
    elif tag.startswith('J'):  # adj
        return 'a'
    elif tag.startswith('R'):  # adv
        return 'r'
    else:
        return 'n'
    
tokens = word_tokenize(text)
tagged = pos_tag(tokens)

lemmatizer = WordNetLemmatizer()
# text = " ".join(lemmatizer.lemmatize(word) for word in text.split())
text = " ".join(lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in tagged)
print('After lemmatization:', text)

# Stemming
# stemmer = PorterStemmer()
# text = " ".join(stemmer.stem(word) for word in text.split())
# print(text)

## tokenization
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
tokens = tokenizer.tokenize(text)
print(tokens)
print('tokenization:', text)

##  token convert ids 
ids = tokenizer.convert_tokens_to_ids(tokens)
print('token ids:', ids)

## or using transformers BertTokenizer tokenization to ids 
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
token_ids = tokenizer(text, return_tensors="pt").to(device)
print(token_ids.input_ids)