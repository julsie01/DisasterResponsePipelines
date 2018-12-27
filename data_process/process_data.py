import sys
# import libraries
import pandas as pd
import numpy as np 

import warnings
import sqlite3
from sqlalchemy import create_engine
from sqlalchemy.sql import text


def load_data(messages_filepath, categories_filepath):
    """
    Input: messages_filepath - path to messages file
    Input: categories_filepath - path to categories file
    Output: Dataframe containing the two datasets merged
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    return messages, categories

def remove_duplicates(df, column):
    """
    Helper function to remove duplicate records from dataframe
    Keeps the first record found in the dataframe for the given column
    Input: df: Pandas dataframe, column: name of the column containing duplicates
    to be removed. 
    """
    df_unique = df.drop_duplicates(subset =column, 
                     keep = 'first', inplace = False) 

    return df_unique

def clean_columns(df):
        
        """
        Helper function to clean the columns by taking only the last character 
        which is a 1 or 0 and converting to a numeric value
        Input: Pandas Dataframe
        Output: Pandas Dataframe
        """
        
         #Iterate through the category columns in df to keep only the last character of each string (the 1 or 0). 
        # For example, related-0 becomes 0, related-1 becomes 1. Convert the string to a numeric value.
    for column in df:
        #do we need to trim white space
        # set each value to be the last character of the string
        df[column] = (df[column]).astype(str)
        df[column] = df[column].str[-1:]
        # convert column from string to numeric
        df[column] = (df[column]).astype(int)
    return df

def split_column_into_seperate_columns(df, column_to_split, delimiter):
      """
      Helper function to take a column containing delimted fields and split
      into different rows
      Input: df: Pandas Dataframe
      column_to_split: name of column to extract data from
      delimiter: character used to split data into seperate records
      Output: Dataframe with column extracted into many
      """
      # Split categories into separate category columns
      # create a dataframe of the 36 individual category columns
      split_column = df[column_to_split].str.split(pat=delimiter,expand=True)
      return split_column

def clean_data(df):
     #this method is specific to this dataset, however it uses a collection
     # of utility functions that could be applied to any dataset  
    
    #split category columns into seperate columns
    categories = split_column_into_seperate_columns(df, 'categories', ';')

    #categories look like related-1
    #in order to split out the category name from the number

    # select the first row of the categories dataframe
    row =  categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    
    row = row.astype(str)
    row = row.apply(lambda x: x[:-2])
    #store the category column names
    category_colnames = row
    # rename the columns of `categories`
    categories.columns = category_colnames
    #convert category values to 0 or 1
    
    for column in categories:
        
         # set each value to be the last character of the string
        categories[column] = (categories[column]).astype(str)
        categories[column] = categories[column].str[-1:]
        # convert column from string to numeric
        categories[column] = (categories[column]).astype(int)

    #replace categories column in df with new categories column

    #Concatenate df and categories data frames.
    df = pd.concat([df, categories], axis=1)
    #Drop the categories column from the df dataframe since it is no longer needed.
    df.drop('categories', axis=1, inplace=True)

     #remove duplicate messages
    df_unique = remove_duplicates(df, 'message')  
    df = pd.concat([df, categories], axis=1)

    return df, category_colnames

def save_data(df, database_filename, table_name):
    """
    Save the dataframe to an sqllite database table
    Data is appended to the named database table if it already exists
    Input: df : pandas dataframe, 
    database_filename: name of database file to save too
    table_name : name of table to save dataframe too
    Output: None
    """
    engine = create_engine("sqlite:///" + database_filename)
    #this has been set to replace to enable repeated running of script
    #with the same dataset. this would need to be updated to 'append' to
    # use with regular updates of data 
    df.to_sql(table_name, engine, index=False, if_exists='replace') 

def clean_database(database_filename):
    """
    Helper function to clean sql lite database between running of script
    Added to allow rerunning of script
    Do not call in production environment
    Input: name of database file
    """
    engine = create_engine("sqlite:///" + database_filename)
    sql = text('DROP TABLE IF EXISTS Categories;')
    result = engine.execute(sql)
    sql = text('DROP TABLE IF EXISTS Messages;')
    result = engine.execute(sql)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))

        messages, categories = load_data(messages_filepath, categories_filepath)

        print('Data loaded into dataframe with columns:')
        print(messages.columns)

        print ('Merge data sets')
        df = pd.merge(messages, categories, on='id')

        print('Cleaning data...')
        df, category_colnames = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath, 'Messages')

        #question how to get the categories dataset
        print('Saving category names')
        save_data(category_colnames, database_filepath, 'Categories')
        
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