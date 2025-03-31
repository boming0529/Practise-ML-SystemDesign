import nltk
## base
nltk.download("wordnet")

## Auto POS tagging, ex. verb, adj, n
# nltk < 3.8
# nltk.download('averaged_perceptron_tagger')
# nltk.download('punkt')     

# nltk > 3.8
nltk.download('punkt_tab')   
nltk.download('averaged_perceptron_tagger_eng')
# nltk.download('averaged_perceptron_tagger_ger') # for German if need