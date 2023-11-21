import pandas as pd
import logging #for logging info during excecution. Logging is keeping a log of events that occur in the system, such as errors or info on current operations
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer #converting a collection of text documents to a matrix of token counts
from sklearn.compose import ColumnTransformer #applying transformers to columns of an array or Dataframe. Transformers are crucial for data preprocessing. Can perform many tasks such as imputation (filling in missing values), normalization/scaling, encoding, feature extraction, text processing, custom transformations.
from sklearn.pipeline import make_pipeline #constructing a pipeline from a list of (name, transform) tuples
from sklearn.dummy import DummyRegressor #imported from "sklearn.compose"
from sklearn.linear_model import Ridge #imported for the Ridge regression model (linear regression whose coefficients arent estimated by ordinary least squares, but by an estimater, called ridge estimator)
from sklearn.metrics import mean_absolute_error

def main():
    logging.getLogger().setLevel(logging.INFO)
    
    logging.info("Loading training/test data")
    train = pd.DataFrame.from_records(json.load(open('train.json'))).fillna("") #missing values are filled with empty strings
    test = pd.DataFrame.from_records(json.load(open('test.json'))).fillna("")
    
    logging.info("Splitting validation")
    train, val = train_test_split(train, stratify=train['year'], random_state=123)
    featurizer = ColumnTransformer(
        transformers=[("title", CountVectorizer(), "title")],
        remainder='drop') #transform the data, applying specific transformations to specified columns while dropping the remaining columns
                            #the parameter is a list of tuples. Countvectorizer() is a feature extraction technique in NLP, which transforms the "title" column into a numerical representation. Remainder='drop'
                            # that parameter specifies that columns that are not specified in the transformer should be dropped/removed
    dummy = make_pipeline(featurizer, DummyRegressor(strategy='mean')) 
    ridge = make_pipeline(featurizer, Ridge()) # a pipeline is a way to streamline a lot of the routine processes as it bundles together a sequence 
                                                #of data processing steps and a model. Featurizer transforms the data using the specified transformations
                                                #dummy predicts the mean value of the target variable. Ridge regression is more complex as it learns the relationships
                                                #between features in the data. Dummy regressor is basic model, doesnt learn from data but makes predictions based on
                                                #a predefined strategy (in this case predicting the mean)
    
    logging.info("Fitting models") #fit models to the training data
    dummy.fit(train.drop('year', axis=1), train['year'].values) #train['year], provides the target variable values
    ridge.fit(train.drop('year', axis=1), train['year'].values)
    
    logging.info("Evaluating on validation data")
    err = mean_absolute_error(val['year'].values, dummy.predict(val.drop('year', axis=1)))
    
    logging.info(f"Mean baseline MAE: {err}")
    err = mean_absolute_error(val['year'].values, ridge.predict(val.drop('year', axis=1)))
    logging.info(f"Ridge regress MAE: {err}")
    
    logging.info(f"Predicting on test") #predicting the year with the trained ridge model
    pred = ridge.predict(test)
    test['year'] = pred
    
    logging.info("Writing prediction file")
    test.to_json("predicted.json", orient='records', indent=2)
    
main()

