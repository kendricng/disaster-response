from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine

import nltk
import pandas as pd
import pickle
import re
import sys

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def load_data(database_filepath):
    """
    Load data into DataFrames.

    Input:
        database_filepath - suggest using './data/database_name'

    Return:
        X              - DataFrame with message as predictive variable
        Y              - DataFrame with all message types
        category_names - column names of Y
    """
    # load database onto dataframe
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    table_name = re.sub(
        r'(.db)$', '', database_filepath.split('/')[-1]
    )
    df = (
        pd
        .read_sql_table(table_name, engine)
        .drop('id', axis=1)
    )
        
    # set model variables
    X = df['message']
    Y = df.drop(['message', 'genre'], axis=1)
    category_names = Y.columns.values
    
    return X, Y, category_names


def tokenize(text):
    """
    Tokenize text into a list of words.

    Input:
        text - a 'message' value in the DataFrame

    Return:
        words - lemmatize message and add them as elements
                in a list
    """
    # step 1: tokenize text
    words = word_tokenize(
        re.sub(r'[^a-zA-Z0-9]', ' ', text.lower())
        .strip()
    )
    # step 2: remove stop words
    words = [ w for w in words if w not in stopwords.words('english') ]
    
    # step 3: lemmatize text
    words = [ WordNetLemmatizer().lemmatize(w) for w in words ]
    
    # step 3: get word stems only
    words = [ PorterStemmer().stem(w) for w in words ]

    return words


def build_model():
    """
    Build a machine learning pipeline with GridSearch
    hyperparamenters, which have been limited to 2 to
    run on a CPU for a reasonable amount of time.

    Expect around 2-3 hours of training time.

    Input:
        None.

    Return:
        cv - a machine learning pipeline with GridSearch
             hyperparameters included
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'clf__estimator__min_samples_split': [2, 3, 4]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluate model and print out classification metrics"""
    # predict model
    Y_pred = model.predict(X_test)
    
    # print classification metrics
    for idx, val in enumerate(category_names):
        print('Category {}: {}'.format(idx, val))
        print(classification_report(Y_test.values[:, idx], Y_pred[:, idx]))
    
    pass


def save_model(model, model_filepath):
    """Save model into a pickle file"""
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)
    
    pass


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

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
