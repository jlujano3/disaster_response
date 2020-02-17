import json
import plotly
import pandas as pd

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
import sqlite3

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

app = Flask(__name__)

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

# load data
df = pd.read_sql('select * from message2', con = sqlite3.connect('/home/workspace/disaster_response.db'))

# load model
model = joblib.load("../models/classifier.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)

# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 
    
    input_data = pd.DataFrame({'message': [query],
                               'genre_direct': [1],
                               'genre_news': [0],
                               'genre_social': [0]})

    x_cols = ['message', 'genre']
    dnu_cols = ['id', 'original']
    col_labs = ['related', 'aid_related', 'weather_related', 'direct_report', 'request']
    
    # use model to predict classification for query
    print(query)
    print(col_labs)
    preds = model.predict(input_data)[0]
    print(preds)
    
    classification_labels = preds
    classification_results = dict(zip(col_labs, classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )

def main():
    app.run(host='0.0.0.0', port=3001, debug=True)

if __name__ == '__main__':
    main()