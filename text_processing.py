import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords


def puncation(text):
    tokens = word_tokenize(text)
    words = [word for word in tokens if word.isalpha()]
    words=" ".join(words)
    return words

def stopwordsremoval(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [word for word in tokens if not word in stop_words]
    words=" ".join(words)
    return words


def pstem(text): 
    ps = PorterStemmer()
    tokens = word_tokenize(text)
    words = []
    for w in tokens:
        a = ps.stem(w)
        words.append(a)
    words = " ".join(words)
    return words


w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()
def lemt(text):
      a=[lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]
      a=" ".join(a)
      return a

