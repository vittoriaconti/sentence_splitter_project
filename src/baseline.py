import nltk

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

def split_with_nltk(text, language="italian"):
    sentences = nltk.sent_tokenize(text, language=language)
    return sentences