import sys
from sqlalchemy import create_engine
import pandas as pd 
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report 
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer 
from sklearn.pipeline import Pipeline, FeatureUnion 
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer 
from nltk.stem.porter import PorterStemmer 
import re 
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.multioutput import MultiOutputClassifier 
from sklearn.model_selection import train_test_split, GridSearchCV
import pickle
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])

def load_data(database_filepath):
    """
    argument: database file path 
    output: features and labels data 
    
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('df', engine) 
    X = df['message'] 
    y = df.drop(['message','id', 'genre', 'original'], axis = 1)
    category_names = y.columns.tolist()
    return X, y, category_names

def tokenize(text):
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
   
    text = re.sub(r"[^A-Za-z0-9]", " ", text.lower().strip())
    tokens = word_tokenize(text)
    stop_words = stopwords.words("english")
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [WordNetLemmatizer().lemmatize(tok) for tok in tokens]
    return tokens 
    
    
def build_model():
    pipeline = Pipeline([('features', FeatureUnion([
                                          ('text_pipeline', Pipeline([
                                                 ('vect', CountVectorizer(tokenizer = tokenize)), 
                                                 ('Tfidf', TfidfTransformer())
                                                                                ])), 
                                           ('startingverb', StartingVerbExtractor())  ])) , 
                 ('clf', MultiOutputClassifier(estimator = RandomForestClassifier(random_state =42)) )  
    ])
    parameters = { "features__text_pipeline__vect__ngram_range" : [(1,2), (1,1)] , "clf__estimator__max_depth":[10, 15, None] , 
              "clf__estimator__min_samples_leaf" :[1,2,3], "clf__estimator__class_weight" :['balanced', None] }
    cv = GridSearchCV(pipeline, parameters, cv = 5)

    return cv
 
    
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
     
   
def evaluate_model(model, X_test, y_test, category_names):
    y_predict = model.predict(X_test) 
    y_predict_df = pd.DataFrame(y_predict, columns = category_names)
    for category in category_names: 
        print(classification_report(y_test[category], y_predict_df[category]))
        
def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test , category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()