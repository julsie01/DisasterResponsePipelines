import datetime
import logging
import re
import sqlite3
import sys
from time import time

import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     train_test_split)
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sqlalchemy import create_engine

import model_evaluator
import text_process.transform_text as transform_text

def load_data_from_sql_lite(db_name, table_name):
    """ Input: db_name - name of the sql lite  database to read from
        Output: df: Pandas DataFrame
    """ 
    engine = create_engine("sqlite:///" + db_name)
    df = pd.read_sql_table(table_name, engine)
    return df

def split_out_target_variable(df, target_columns, text_column):
    """ Input: dataframe, name of target column, name of text column
        Output: X : Features columns, Y : Target Column
    """ 
    X = df[text_column].values
    y = df[target_columns].values
    return X, y

def load_data(db_name):

    """Input: database name
    Output: X features columns, Y: Target Column, target_columns: List of Target Columns

    Function to load data from the named database and return features and targets
    """

    TEXT_COLUMN = 'message'

    #load the data from sql lite database
    messages = load_data_from_sql_lite(db_name, 'Messages')
    categories = load_data_from_sql_lite(db_name, 'Categories')
    
    #get target columns
    target_columns = categories['0'].values
    #split out target variables
    X, y = split_out_target_variable(messages, target_columns, TEXT_COLUMN)
    return X, y, target_columns


def build_randomforest_pipeline():
    """
    Create a multioutput classifer with a Random Forest Classifier
    """
    classifier = MultiOutputClassifier(RandomForestClassifier(n_estimators=10, random_state=1))
    return classifier

#could change this function to take a model that gets wrappedin a multioutput classifier
def build_xgboost_pipeline():
    """
    Create a multioutput classifer with an XGBoost Classifier
    """
    #default number of estimators is 100
    classifier = MultiOutputClassifier(xgb.XGBClassifier(max_depth=5, reg_alpha=0.01, scale_pos_weight = 1))
    return classifier   

def build_classifier_pipeline(selected_classifier):
    """Input: a classifier
        Output: classifier packaged as a multioutput classifer for the pipeline

        
     """

    classifier = MultiOutputClassifier(selected_classifier)
    return classifier

def build_text_features_pipeline():
    """
    Build a text features pipeline by combining count vectorizere and tfidf transformer
    """
    features_pipeline = Pipeline([
                ('vect', CountVectorizer(tokenizer=transform_text.tokenize, ngram_range=(1,2))),
                ('tfidf', TfidfTransformer(use_idf=False))
                 ])
      
        
    return features_pipeline




def build_model():
    """
    Build a model pipeline by combining a text features pipeline with a classifier pipeline
    """
    featurespipeline =build_text_features_pipeline()
    #modelpipeline = build_classifier_pipeline(RandomForestClassifier(n_estimators=10, random_state=1))
    modelpipeline = build_xgboost_pipeline()

    pipeline = Pipeline([
            
           ('textpipeline', featurespipeline ),
      
        ('clf', modelpipeline)
    ])

    return pipeline

def perform_grid_search(pipeline, parameters, X_train, y_train):  

    """
    Input: pipeline: sklearn pipeline , parameters : grid of parameters for the pipeline
    Output: The best estimator returned by the grid search
    """
    
    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    print(parameters)
    print("available parameters")
    
    grid_search = GridSearchCV(pipeline, param_grid=parameters, verbose=5, cv=5)
    
    t0 = time()
    grid_search.fit(X_train, y_train)
    
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    
    best_parameters = grid_search.best_estimator_.get_params()
    #get the best estimator
    best_estimator = grid_search.best_estimator_
    
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
    return best_estimator

def perform_random_grid_search(pipeline, parameters, X_train, y_train):  
    """
    Input: pipeline: sklearn pipeline , parameters : grid of parameters for the pipeline
    Output: The best estimator returned by the random search
    """

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    print(parameters)
    print("available parameters")
    
    n_iter_search = 20
    grid_search = RandomizedSearchCV(pipeline, param_grid=parameters, n_iter=n_iter_search, cv=5, verbose=5)
    
    t0 = time()
    grid_search.fit(X_train, y_train)
    
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    
    best_parameters = grid_search.best_estimator_.get_params()
    #get the best estimator
    best_estimator = grid_search.best_estimator_
    
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
    return best_estimator



def evaluate_model(model, X_test, Y_test, category_names):
    preds = model.predict(X_test)
    #requirement is to print out the results
    evaluator = model_evaluator.ModelEvaluation()
    evaluator.display_results(Y_test, preds, category_names)

def tune_model(pipeline, x_train, y_train, grid_search):

    """
    Input: pipeline: Sklearn pipeline
    X_train : training set features, 
    y_train : training set outcome variable to be predicted,
    grid_search: boolean flag, if on run grid search otherwise run random search
    Output: the model with the best score from the grid search
    """
    #small subset of parameters for testing code as grid search is long running
    mini_parameters = {
        'textpipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'textpipeline__tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [50, 100, 200]
    }
    parameters = {
        'textpipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'textpipeline__vect__max_df': (0.5, 0.75, 1.0),
        'textpipeline__vect__max_features': (None, 5000, 10000),
        'textpipeline__tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [50, 100, 200],
         'clf__estimator__min_samples_split': [2, 3, 4],
        'clf__estimator__max_depth' : [5,7,10]
       
    }

    if (grid_search):
        best_estimator = perform_grid_search(pipeline, parameters, x_train, y_train)
    else:
        best_estimator = perform_random_grid_search(pipeline, parameters, x_train, y_train)
    
    return best_estimator


def save_model(model, model_filepath):
    """
    Save Model : Stores the model on the file system
    Input: model: model to save, model_filepath: the path to save the model too
    """

    joblib_file = model_filepath   
    joblib.dump(model, joblib_file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

        print('Tuning Model')
        best_model = tune_model(model, X_train, Y_train, False)

        save_model(best_model, 'best_estimator.pkl')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
