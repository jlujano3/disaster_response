import sys

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# import libraries
import pandas as pd
import numpy as np
import sqlite3
import re
import string

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import FunctionTransformer

def load_data(database_filepath):
    print(database_filepath)
    df = pd.read_sql('select * from message2', con = sqlite3.connect(database_filepath))

    x_cols = ['message', 'genre']
    dnu_cols = ['id', 'original']

    X = df[x_cols]
    #y = df[list(set(df.columns) - set(x_cols + dnu_cols))[:2]]
    y = df[['related', 'aid_related', 'weather_related', 'direct_report', 'request']]
    
    for col in y.columns:
        if len(y[col].unique()) != 2:
            y.drop([col], axis = 1, inplace = True)
    
    dum = pd.get_dummies(X['genre'], prefix = 'genre')
    
    X.drop(['genre'], axis = 1, inplace = True)
    
    X = pd.concat([X, dum], axis = 1)
    
    return X, y

def tokenize(text):
    #remove punctuation
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    
    #remove None
    
    #remove stop words
    stop_word = set(stopwords.words('english'))
    
    #tokenize words
    text = regex.sub('', text)
    tokens = word_tokenize(text)
    
    #lemmatize words
    lemmatizer = WordNetLemmatizer()

    #lower case words
    clean_tokens = []
    for tok in tokens:
        #remove stop words
        if tok.lower() in stop_word:
            continue
            
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def get_msg_col(df):
    return df['message']

def get_gen_col(df):
    return df[[col for col in df.columns if 'genre' in col]]

def build_model():
    pipeline = Pipeline([
        ('features', FeatureUnion([(
                'msg_pl', Pipeline([
                    ('msg_col', FunctionTransformer(get_msg_col, validate=False)),
                    ('vect', CountVectorizer(tokenizer=tokenize, max_features = 10000)),
                    ('tfidf', TfidfTransformer())
                ])),
                ('gen_pl', Pipeline([
                    ('gen_col', FunctionTransformer(get_gen_col, validate=False))
                ]))])),
        ('clf', MultiOutputClassifier(GradientBoostingClassifier(random_state = 1)))
    ])
    
    parameters = {
        'clf__estimator__min_samples_leaf': (20, 100), 
        'clf__estimator__n_estimators': (50, 100),
        'clf__estimator__max_depth': (3, 4)
    }

    cv = GridSearchCV(pipeline, param_grid = parameters, verbose = 2, return_train_score = True, n_jobs = -1)
    return cv

def evaluate_model(model, X_test, y_test):
    print('creating prediction')
    y_pred = model.predict(X_test)
    print('creating probability prediction')
    pred_prob = model.predict_proba(X_test)
    print('displaying results')
    
    def display_results(cv, y_pred, y_test, y_pred_prob):
        print(cv.best_params_)
        for idx, col in enumerate(y_test.columns):
            print(col)
            print('prediction probability mean, std = {0:.4f} {1:.4f}'.format(y_pred_prob[idx][:, 1].mean(), y_pred_prob[idx][:, 1].std()))
            print('prediction mean = {0:.4f}'.format(y_pred[:, idx].mean()))
            print('actual mean = {0:.4f}'.format(y_test[col].mean()))
            print('precision = {0:.4f}'.format(precision_score(y_test[col], y_pred[:, idx])))
            print('recall = {0:.4f}'.format(recall_score(y_test[col], y_pred[:, idx])))
            print('f1 = {0:.4f}'.format(f1_score(y_test[col], y_pred[:, idx])))
            print('auc = {0:.4f}'.format(roc_auc_score(y_test[col], y_pred_prob[idx][:, 1])))
            print()
    
    display_results(model, y_pred, y_test, pred_prob)
    
    cols = y_test.columns
    
    def model_exp(model, cols):
        def find_val(vocab_dict, ret_val):
            for key, val in vocab_dict.items():
                if val == ret_val:
                    return key

        best_vocab = model.best_estimator_.steps[0][1].transformer_list[0][1].steps[1][1].vocabulary_

        for idx, clf in enumerate(model.best_estimator_.steps[1][1].estimators_):
            print(cols[idx])
            feat_imp = clf.feature_importances_

            max_list = [0, 0, 0]

            for val in feat_imp:
                if val > min(max_list):
                    max_list[max_list.index(min(max_list))] = val

            max_idx = [np.where(feat_imp == feat)[0][0] for feat in max_list]

            for counter, idx in enumerate(max_idx):
                print(max_list[counter], idx, find_val(best_vocab, idx))

            print()

    model_exp(model, cols)
    
def save_model(model, model_filepath):
    pd.to_pickle(model, model_filepath)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test)

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')

if __name__ == '__main__':
    main()