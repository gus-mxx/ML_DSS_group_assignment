import pandas as pd
import logging
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

#New text processing tecniques that I applied
from sklearn.feature_extraction.text import TfidfVectorizer

#Text processing technique: GloVe vectorizer
pip install spacy
import spacy
python -m spacy download en_core_web_md

#Decision tree regressor
from sklearn.tree import DecisionTreeRegressor 

import time

start_time = time.time() 

def main():
    logging.getLogger().setLevel(logging.INFO)
    logging.info("Loading training/test data")
    train = pd.DataFrame.from_records(json.load(open('train.json'))).fillna("")
    test = pd.DataFrame.from_records(json.load(open('test.json'))).fillna("")

    logging.info("Splitting validation")
    train, val = train_test_split(train, stratify=train['year'], random_state=123)
    #Adjusted the features by creating a for loop that runs through multiple atributes.
    feature_names=["title","abstract", "publisher", "author"] 

    transformers=[(feature, GloveVectorizer(), feature)for feature in feature_names]

    featurizer = ColumnTransformer(
        transformers=transformers,
        remainder='drop')
    dummy = make_pipeline(featurizer, DummyRegressor(strategy='mean'))
    #ridge = make_pipeline(featurizer, Ridge())
    tree = make_pipeline(featurizer, DecisionTreeRegressor()) #add decision tree model

    logging.info("Fitting models")
    dummy.fit(train.drop('year', axis=1), train['year'].values)
    #ridge.fit(train.drop('year', axis=1), train['year'].values)
    tree.fit(train.drop('year', axis=1), train['year'].values) #decision tree

    logging.info("Evaluating on validation data")
    err = mean_absolute_error(val['year'].values, dummy.predict(val.drop('year', axis=1)))
    logging.info(f"Mean baseline MAE: {err}")
    #err = mean_absolute_error(val['year'].values, ridge.predict(val.drop('year', axis=1)))
    #logging.info(f"Ridge regress MAE: {err}")
    err = mean_absolute_error(val['year'].values, tree.predict(val.drop('year', axis=1)))
    logging.info(f"Tree regress MAE: {err}")

    logging.info(f"Predicting on test")
    #pred = ridge.predict(test)
    pred = tree.predict(test)
    test['year'] = pred
    logging.info("Writing prediction file")
    test.to_json("predicted.json", orient='records', indent=2)
    

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Execution Time: {elapsed_time} seconds")

main()

""""Predictions with CountVectorizer"""

#Prediction with feature= "title"
#INFO:root:Mean baseline MAE: 7.8054390754858805
#INFO:root:Ridge regress MAE: 5.8123259857253915

#Prediction with feature = "abstract"
#INFO:root:Mean baseline MAE: 7.8054390754858805
#INFO:root:Ridge regress MAE: 6.371295315649925

#Prediction with feature = "publisher"
#INFO:root:Mean baseline MAE: 7.8054390754858805
#INFO:root:Ridge regress MAE: 5.443128700037363

""""Predictions with TfidfVectorizer"""

#Prediction with feature= "title"
#INFO:root:Mean baseline MAE: 7.8054390754858805
#INFO:root:Ridge regress MAE: 5.387367490324335

#Prediction with feature = "publisher"
#INFO:root:Mean baseline MAE: 7.8054390754858805
#INFO:root:Ridge regress MAE: 5.444409508095158

#Prediction with features = "abstract", title and publisher and a decision tree
#INFO:root:Mean baseline MAE: 7.8054390754858805
#INFO:root:decision tree MAE: 4.031015478965557 !!!!!!!!!!!!!!!!

""""Decision tree and TfidfVectorizer""""
#Prediction with features = "abstract", title and publisher
# INFO:root:Mean baseline MAE: 7.8054390754858805
# INFO:root:Ridge regress MAE: 4.097044383573657

#Less features result in a higher MAE

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