import json
import plotly
import nltk
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens

# create class of Starting Verb extractor 
class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    Starting Verb Extractor class
    This class extract the starting verb of a sentence,
    creating a new feature for the ML classifier
    """
    def startingverb(self, text):
        sentences = sent_tokenize(text)
        for sentence in sentences: 
            pos_tag = nltk.pos_tag(tokenize(sentence))
            if len(pos_tag)>0: 
                first_word, first_tag = pos_tag[0]
                if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                    return True 
        return False   
    def fit(self, X, y= None):
        return self
    def transform(self, X): 
        X_tagged = pd.Series(X).apply(self.startingverb)
        return pd.DataFrame(X_tagged)
   
# load data
engine = create_engine('sqlite:///' + './data/disaster_response.db')
df = pd.read_sql_table('df', engine)

# load model
model = joblib.load("./models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    category_counts = df.drop(['id','message','original','genre'], axis=1).sum()
    category_names = list(category_counts.index)

    word_series = pd.Series(' '.join(df['message']).lower().split())
    top_words = word_series[~word_series.isin(stopwords.words("english"))].value_counts()[:5]
    top_words_names = list(top_words.index)


    # create visuals
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
        },
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=top_words_names,
                    y=top_words
                )
            ],

            'layout': {
                'title': 'Most Frequent Words',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Words"
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

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

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
