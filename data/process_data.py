import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    messages.head()
    
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    categories.head()
    
    # merge datasets
    df = messages.merge(categories, how = 'left', on = 'id')
    return df

def clean_data(df):
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat = ';', expand = True)

    # select the first row of the categories dataframe
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    # rename the columns of `categories`
    categories.rename(columns = dict(categories.iloc[0].str.split(pat = '-').apply(lambda row: row[0])), inplace = True)

    # set each value to be the last character of the string
    # convert column from string to numeric
    for col in categories.columns:
        categories[col] = categories[col].apply(lambda row: int(row[-1]))
        
    df.drop(['categories'], inplace = True, axis = 1)

    df = pd.concat([df, categories], axis = 1)

    # check number of duplicates
    print('{0} rows duplicated'.format(df.duplicated().sum()))

    # drop duplicates
    df = df.loc[~df.duplicated()]

    # check number of duplicates
    print('{0} rows duplicated'.format(df.duplicated().sum()))

    non_target = ['id', 'genre', 'message', 'original']
    for col in df.columns:
        if col in non_target:
            continue
        else:
            if len(df[col].unique()) > 2:
                df = df.loc[df[col].isin([0, 1])]
                print('{0} snipped'.format(col))
    return df

def save_data(df, database_filename):
    engine = create_engine(database_filename)
    df.to_sql('message3', engine, index=False)

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