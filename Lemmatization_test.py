from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize

# ensure download
# resources = ['wordnet', 'averaged_perceptron_tagger_eng', 'punkt_tab']
# for resource in resources:
#     try:
#         nltk.data.find(f'corpora/{resource}' if resource == 'wordnet' else f'taggers/{resource}' if resource == 'averaged_perceptron_tagger' else f'tokenizers/{resource}')
#     except LookupError:
#         nltk.download(resource)

lemmatizer = WordNetLemmatizer()
text = "walking better"
text = " ".join(lemmatizer.lemmatize(word) for word in text.split())
print('After lemmatization:', text)

# default pos
print(lemmatizer.lemmatize("walking"))  # output: "walking"
print(lemmatizer.lemmatize("better"))   # output: "better"

# Specify pos
print(lemmatizer.lemmatize("walking", pos="v"))  # output: "walk"（verb）
print(lemmatizer.lemmatize("better", pos="a"))   # output: "good"（adj）



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
text = " ".join(lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in tagged)
print('After lemmatization:', text)