from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sqlalchemy import create_engine
from plotly.graph_objs import Bar

import nltk
import re
import pandas as pd

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def load_data():
    """Load data into dataframe"""
    # load database onto dataframe
    engine = create_engine('sqlite:///./data/DisasterResponse.db')
    df = (
        pd
        .read_sql_table('DisasterResponse', engine)
        .drop('id', axis=1)
    )
    
    return df

def tokenize_graph(text):
    """Tokenize function specific to graph 3"""
    words = word_tokenize(
        re.sub(r'[^a-zA-Z0-9]', ' ', text.lower())
        .strip()
    )
    words = [ w for w in words if w not in stopwords.words('english') ]
    words = [ WordNetLemmatizer().lemmatize(w) for w in words ]

    return " ".join(words)

def clean_data_one(df):
    """Clean data for graph one"""
    categories = df.drop(columns=['message', 'genre'])
    sum_per_category = categories.sum()
    percent_category = (
        (sum_per_category / sum_per_category.sum())
        .sort_values(ascending=False)
        .reset_index()
    )
    percent_category.columns = ['category', 'percent_share']
    
    return percent_category

def clean_data_two(df):
    """Clean data for graph two"""
    genre_counts = df.groupby('genre').count()['message']
    
    return genre_counts

def clean_data_three(df, col_name, max_words=50):
    """Returns a list of top most frequently used words"""
    # join all messages into one string
    y = " ".join(
        df
        .query('{} == 1'.format(col_name))
        .message
        .apply(tokenize_graph)
        .values
    )
    
    # hard code stop words for this exercise
    stop_words = [
        'also', 'area', 'bit', 'co', 'help', 'http', 'https', 'http co',
        'include', 'ly', 'many', 'need', 'one', 'people', 'please', 
        'rt', 'said', 'still', 'thank', 'thanks', 'two', 'well', '000',
        '7', 'would', 'like', '8', 'u', '5', '0', '2', '6', '1', '3', '4',
        '10', '12', '9', '20'
    ]
    stop_words.append(col_name)
    
    # calculate word frequency
    total = y.split()
    unique = set(y.split()) ^ set(stop_words)
    counts = dict()
    for word in unique:
        counts[word] = total.count(word)
    counts = {k: v for k, v in sorted(
        counts.items(), key=lambda item: item[1], reverse=True
    )}
    
    # store most frequent words and their count
    top_words = [k for k, v in counts.items()]
    top_n_words = top_words[:max_words]
    
    top_values = [v for k, v in counts.items()]
    top_n_values = top_values[:max_words]
    
    return top_n_words, top_n_values

def return_figures():
    """Create plotly graphs"""
    # load relevant data
    df = load_data()
    percent_category = clean_data_one(df)
    genre_counts = clean_data_two(df)
    words, count = clean_data_three(df, 'security', 20)
    
    # graph one: percentage of categories represented in dataset
    graph_one = []
    graph_one.append(
        Bar(
            x = percent_category.category.tolist()[:10],
            y = percent_category.percent_share.tolist()[:10]
        )
    )

    layout_one = dict(
        title = 'Share of Messages per Category (Top 10)',
        xaxis = dict(title = 'Category'),
        yaxis = dict(title = 'Percent Share')
    )
    
    # graph two: message count per genre
    graph_two = []
    graph_two.append(
        Bar(
            x = list(genre_counts.index),
            y = genre_counts
        )
    )

    layout_two = dict(
        title = 'Distribution of Message Genres',
        xaxis = dict(title = 'Genre'),
        yaxis = dict(title = 'Message Count')
    )

    # graph three: word frequency for the `security` category
    graph_three = []
    graph_three.append(
        Bar(
            x = words,
            y = count
        )
    )

    layout_three = dict(
        title = 'Top 20 Used Words for Security Related Messages',
        xaxis = dict(title = 'Word Used'),
        yaxis = dict(title = 'Count')
    )
    
    # append all graphs to the figures list
    figures = []
    figures.append(dict(data=graph_one, layout=layout_one))
    figures.append(dict(data=graph_two, layout=layout_two))
    figures.append(dict(data=graph_three, layout=layout_three))
    
    return figures