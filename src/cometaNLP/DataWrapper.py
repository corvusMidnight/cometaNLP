## Script ##
import emoji
import lib2to3.pgen2
from lib2to3.pgen2 import token
import pandas as pd
import textstat
import os
from os import stat
import operator
from operator import index
import re
import regex
import pandas as pd
import collections
from collections import Counter
import string
from textstat.textstat import *
from nltk import word_tokenize
from nltk.corpus import stopwords
import nltk
import spacy
from spacy import displacy
from nltk.tokenize import TreebankWordTokenizer as twt
from nltk.tag import pos_tag
import matplotlib.pyplot as plt
from cometaNLP import TextAnalyzer


# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#SUBCLASS | SUBCLASS |SUBCLASS | SUBCLASS |SUBCLASS | SUBCLASS |SUBCLASS | SUBCLASS |SUBCLASS | SUBCLASS |SUBCLASS | SUBCLASS |SUBCLASS | SUBCLASS |SUBCLASS | SUBCLASS |
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------  

class DataWrapper(TextAnalyzer):
    """A subclass of the TextAnlyzer class. It is meant to analyze text data inside a dataframe"
    
    """
    def __init__(self, language: str, **args):
        super().__init__(language)
        
    def data_wrapper(self, file_type: str, file_path: str, data_frame = False):

        """A class method that applies a series of tranformations to csv and tsv files
        and returns a dictionary

        data_wrapper() reads .csv and .tsv file, applies the functions and methods within the
        the TextAnlyzer module to the column contaning the comments/text in the dataframe
        in meaningful order, and finally converts it into a dictionary by index according to the following format:
        {index -> {column -> value}}. The dictionary contains relevant information
        for each comment in the dataset.
        To make the function as comprehensive as possible, the user is asked to enter
        whether the csv/tsv file has a header or not and the index of the column
        on which they wish to apply the transformations. The column should be the one containing the comments\text
        data that the user whishes to analyze.

        Args:
            self: reference to the current instance of the class
            file_type (str): A string (csv/tsv)
            file_path (str): A string containing a file path to a csv/tsv file
            data_frame (bool): If set to true, the function returns additionally pandas DataFrame object
                               rather than a dictionary

        Returns:
            output (dict): A nested dictionary {index -> {column -> value}} containing
                           relevant data and metadata for each comment in the input dataframe
        """

        other_features_names = ["FKRA", "FRE","num_syllables", "avg_syl_per_word", "num_chars", "num_chars_total", \
                                "num_terms", "num_words", "num_unique_words"]
        
        df = self.load(file_type, file_path)
        
        #Identifying the column containing the hate comments
        c_column = int(input(f'What is the index of the column containing the comments? Remember: Index starts from 0 in Python'))
        comments = df.iloc[:, c_column]


        #Applying the functions to extract data and metadata
        df['n_hashtags'] =  df.iloc[:, c_column].apply(TextAnalyzer.count_hashtags)
        df['n_urls'] =  df.iloc[:, c_column].apply(TextAnalyzer.count_url)
        df['n_user_tags'] =  df.iloc[:, c_column].apply(TextAnalyzer.count_user_tags)

        df['clean_comments'] = df.iloc[:, c_column].apply(TextAnalyzer.preprocessor)
        df['clean_comments'] = df['clean_comments'].apply(TextAnalyzer.punctuation_removal)

        df['n_emojis'] = df.iloc[:, c_column].apply(TextAnalyzer.count_emoji)
        df['clean_comments'] = df['clean_comments'].apply(TextAnalyzer.demojizer)
        
        #Tokenizer choice depending on language
        if self.language == 'italian':
            df['tokenized_comments'] =  df['clean_comments'].apply(self.italian_tokenizer)
        else:
            df['tokenized_comments'] =  df['clean_comments'].apply(self.tokenizer)


        df['length'] =  df['tokenized_comments'].apply(TextAnalyzer.comment_length)
        df['TTR'] =  df['tokenized_comments'].apply(TextAnalyzer.type_token_ratio)
        df['CFR'] =  df['tokenized_comments'].apply(self.content_function_ratio)


        df['stop_words_removed'] =  df['tokenized_comments'].apply(self.stop_words_removal)
        df['lemmatized_comments'] =  df['stop_words_removed'].apply(self.lemmatizer)

        df['POS_comments'] =  df['lemmatized_comments'].apply(self.pos)


        output = df.to_dict('index')

        if data_frame == True:
            return output, df
        else:
            return output
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def data_wrapper_summary(self, file_type, file_path, visualize = True) -> tuple:

        """A class method that returns the relevant data comparison based on grouping 
        comparison (e.g., X v. Y) rather than for each comment individually.

        get_summary() is built upon the data_wrapper() method. If visualize is set to True, it also shows a simple visualization of all the
        summarized data. It compares average number of emojis, hashtags,
        urls, user tags, length, type-token ratio, content-function ratio for
        two classes of comments.
        Args:
            self: reference to the current instance of the class
            file_type (str): A string (csv/tsv)
            file_path (str): A string containing a file path to a csv/tsv file

        Returns:
            tuple: a tuple of values.
        """
        output, df = self.data_wrapper(file_type, file_path, data_frame=True)
        l_column = int(input(f'''Enter the index of the categorical by which you want the data to be grouped by.
                                 Remember: Index starts from 0 in Python'''))
        
        mean_emojis = df.groupby(df.iloc[:, l_column])['n_emojis'].mean()
        mean_hash = df.groupby(df.iloc[:, l_column])['n_hashtags'].mean()
        mean_urls = df.groupby(df.iloc[:, l_column])['n_urls'].mean()
        mean_user_tags = df.groupby(df.iloc[:, l_column])['n_user_tags'].mean()
        mean_length = df.groupby(df.iloc[:, l_column])['length'].mean()
        mean_TTR = df.groupby(df.iloc[:, l_column])['TTR'].mean()
        mean_CFR = df.groupby(df.iloc[:, l_column])['CFR'].mean()
        
        if visualize:
            mean_emojis.plot(kind='bar')
            plt.title('Mean number of emojis')
            plt.show(),

            mean_hash.plot(kind='bar')
            plt.title('Mean number of emojis')
            plt.show(),

            mean_urls.plot(kind='bar')
            plt.title('Mean number of urls')
            plt.show(), 

            mean_user_tags.plot(kind='bar')
            plt.title('Mean number of user tags')
            plt.show(),

            mean_length.plot(kind='bar')
            plt.title('Mean length of the comments')
            plt.show(),

            mean_TTR.plot(kind='bar')
            plt.title('Mean TTR')
            plt.show(),

            mean_CFR.plot(kind='bar')
            plt.title('Mean CFR')
            plt.show()

        return mean_emojis, mean_hash, mean_urls, mean_user_tags, mean_length, mean_TTR, mean_CFR