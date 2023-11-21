#Text processing technique: GloVe vectorizer
pip install spacy
import spacy
python -m spacy download en_core_web_md


""""Decision tree and Word2Vec""""
#Didnt work. It printed a lot of lines, but none of them were the MAE

"""Code for Word2Vec"""
"""Adjustments"""
from gensim.models import Word2Vec
from sklearn.base import BaseEstimator, TransformerMixin

class Word2VecVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, vector_size=100, window=5, min_count=1, workers=4):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.model = None

    def fit(self, X, y=None):
        # Assuming X is a pandas Series containing text data
        sentences = [text.split() for text in X]
        self.model = Word2Vec(sentences, vector_size=self.vector_size, window=self.window, min_count=self.min_count, workers=self.workers)
        return self

    def transform(self, X):
        # Assuming X is a pandas Series containing text data
        return [self.vectorize(text) for text in X]

    def vectorize(self, text):
        # Vectorize a single text
        words = text.split()
        vectors = [self.model.wv[word] if word in self.model.wv else [0] * self.vector_size for word in words]
        return sum(vectors) / len(vectors)


""""Decision tree and Word2Vec""""
"""#Didnt work. It printed a lot of lines, but none of them were the MAE"""

# Custom class to vectorize text using spaCy and GloVe embeddings
class GloveVectorizer:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_md', disable=['parser', 'tagger', 'ner'])

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [self.vectorize(text) for text in X]

    def vectorize(self, text):
        doc = self.nlp(text)
        return doc.vector