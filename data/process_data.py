from sqlalchemy import create_engine

import numpy as np
import pandas as pd
import re
import sys


def load_data(messages_filepath, categories_filepath):
    """
    Combines the messages and categories CSV files into one dataframe
    
    Input:
        messages_filepath - path where messages CSV file is stored
        categories_filepath - path where categories CSV is stored
    
    Returns:
        df - combined dataframe
    
    """
    df = (
        pd
        .read_csv(messages_filepath)
        .drop(['original'], axis=1)
        .merge(
            pd.read_csv(categories_filepath), how='outer', on='id'
        )
    )
    
    return df


def clean_data(df):
    """
    Two main cleaning tasks:
        1. Get dummies for entries in the categories column; and
        2. Remove duplicate rows
    
    Input:
        df - combined dataframe
    
    Return:
        df_clean - cleaned dataframe
    
    """
    
    # step 1: get dummies for categories
    # set category names
    categories = df.categories.str.split(';', expand=True)
    categories.columns = [ 
        re.sub('(-[0-1])*', '', category) \
        for category in categories.iloc[0].values 
    ]

    # get dummies for each category
    for column in categories:
        categories.loc[:, column] = (
            categories[column]
            .astype(str)
            .str
            .extract(r'([0-1])$', expand=False)
            .fillna(0)
            .astype(int)
        )
    
    # remove duplicate categories row
    df = (
        df
        .drop(
            'categories', 
            axis=1
        )
        .merge(
            categories, 
            how='outer', 
            left_index=True, 
            right_index=True
        )
    )
    
    # step 2: remove duplicates
    # count unique ids
    ids = np.append(df['id'].values, df['id'].unique())
    ids_unique, count = np.unique(ids, return_counts=True)

    # identify duplicate entries
    ids_duplicate = []
    for i in range(len(ids_unique)):
        if count[i] > 2:
            ids_duplicate.append(ids_unique[i])

    # drop duplicate entries
    df_clean = df.drop(df[df['id'].isin(ids_duplicate)].index)
    
    return df_clean


def save_data(df, database_filepath):
    engine_name = re.sub(
        r'(.db)$', '', database_filepath.split('/')[-1]
    )
    engine = create_engine(f'sqlite:///{engine_name}')
    df.to_sql(database_filepath, engine, index=False)
    
    pass


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()