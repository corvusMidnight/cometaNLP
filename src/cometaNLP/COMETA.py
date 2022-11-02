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


class cometa:
    '''
    A NLP tool to extract and analyze comments from .tsv and .csv files.
    It supports Italian, Dutch, and English comments.
    
    cometaNLP proviedes two main methods:
    - get_summary
    - get_dictionary
    
    It also contains a series of static methods for independent text analysis.
    It is also possible to analyze individual string of texts using the 
    subclass TextAnalyzer.

    '''
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# STATIC METHODS | STATIC METHODS | STATIC METHODS | STATIC METHODS | STATIC METHODS |STATIC METHODS |STATIC METHODS |STATIC METHODS |STATIC METHODS | STATIC METHODS |
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def load(file_type: str, file_path: str) -> object:

        """A method that reads .csv and .tsv files.
        
       
        Args:
            file_type (str): The type of file containing the data (csv/tsv)
            file_path (str): The path to the file where the file is stored.
          
        Returns:
            object: A pandas DataFrame object
            
        
        """

        head = input('Does the dataframe have an header? Input "no"/"yes" to set Header = None/True"')

        if str(file_type) == "csv" and head == 'yes':
            df = pd.read_csv(file_path)
        if str(file_type) == "tsv" and head == 'yes':
            df = pd.read_csv(file_path, sep='\t')
        if str(file_type) == "csv" and head == 'no':
            df = pd.read_csv(file_path, header=None)
        if str(file_type) == "tsv" and head == 'no':
            df = pd.read_csv(file_path, sep='\t', header=None)
        elif str(file_type) != "csv" and str(file_type) != "tsv":
            return "Sorry, for now we only support .csv and .tsv files"

        #Removing all the Unnamed columns
        try:
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')] #Remove Unnamed columns
        except AttributeError:
            pass


        return df


#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def count_hashtags(text: str) -> int:

        '''
        A function to be run on comments. It returns the number of hashtags
        
        Parameters
        ----------
        text: str
            Any string
        
        Ret
        Necessary packages/modules:
        - re
                            '''
        
        count = 0
        hs = re.findall("#[A-Za-z0-9_]+", text)
        for h in hs:
                count +=1
        
        return count
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def count_url(text: str) -> int:

        '''A function to be run on comments and returns the number of urls'''
        
        '''Necessary packages/modules:
        - re
                            '''

        count = 0
        urls = re.findall("http\S+", text) + re.findall("www.\S+", text) + re.findall("URL", text) + re.findall("url", text)
        for url in urls:
                count +=1
        
        return count

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def count_user_tags(text: str) -> int:
        
        '''A function to be run on comments and returns the number of user mentions'''

        '''Necessary packages/modules:
        - re
                            '''

        count = 0
        tags = re.findall("@[A-Za-z0-9_]+", text)
        for user in tags:
                count +=1
        
        return count

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def preprocessor(text: str) -> str:

        '''A function to be run on the tweets through apply to clean them.
        The function should be run on the original tweet column to obtain
        a new "clean_tweet" column and to prepare the original "tweet" column
        to calculate the AFINN/VADER scores'''
    
        '''Necessary packages/modules:
        - re
                            '''
        #df = self.load()
        #text_column = input(f'What is the name of the column containing the comments? Enter one of the values you see below {df.sample(n=2)}')                    



        #Noise removal based on the explain weights function of the baseline logistic regression
        txt = text.replace('url', '')
        txt = txt.replace('URL', '')
    
        #Hashtag remover
        txt = re.sub("#[A-Za-z0-9_]+","", txt)
    
        #Genral user tag remover (accounting also for potential differently anonymized data) remover
        txt = re.sub("@[A-Za-z0-9_]+","", txt)
    
        #Genral url remover (same as above)
        txt = re.sub(r"http\S+", "", txt)
        txt = re.sub(r"www.\S+", "", txt)
    
        # remove numbers
        txt = re.sub(r'\d+', '', txt)

        # Also, removes leading and trailing whitespaces
        txt = re.sub('\s+', ' ', txt).strip()

        return txt
    
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def punctuation_removal(text:str) -> str:

        # remove punctuations and convert characters to lower case
        txt = "".join([char for char in text if char not in '!"#$%&\()*+,-./:;<=>?@[\\]^_`{|}~']) 
    
    
        return txt


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def count_emoji(text):

        '''A function to be applied to the cleaned comments. It returns the emoji
        count for each comment'''

        '''Necessary packages/modules:
        - emoji
                            '''
        
        emoji_counter = emoji.emoji_count(text)
        return emoji_counter
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def demojizer(txt):

        '''A function to be applied to the cleaned comments. It returns a demojized
        version of the same comments'''

        '''Necessary packages/modules:
        - emoji
                            '''
        txt = emoji.demojize(txt)
        
        return txt

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def comment_length(l: list) -> int:
        
        '''A function to be applied to the tokenized comments. It returns the length in tokens.
        '''

        count = len(l)
        
        return count

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def word_counts(l: list) -> dict:

        '''A function to accumulate counts in a dictionary for each 
        word that appears.
        '''
        counts = Counter()
        
        for token in l:
            counts[token] += 1
        
        return counts
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def type_token_ratio(text: str) -> float:

        '''A function to calculate the type-token ratio on the words in a string. The type-token
        ratio is defined as the number of unique word types divided by the number
        of total words.
        '''
        
        counts = cometa.word_counts(text)

        type_count = len(counts.keys())
        token_count = sum(counts.values())

        return type_count / token_count
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def visualize_pos(tokens: list) -> object:
        pos_tags = ["PRON", "VERB", "NOUN", "ADJ", "ADP",
                    "ADV", "CONJ", "DET", "NUM", "PRT"]
        
        text = " ".join(tokens)
        tags = nltk.pos_tag(tokens, tagset = "universal")

        # Get start and end index (span) for each token
        span_generator = twt().span_tokenize(text)
        spans = [span for span in span_generator]

        # Create dictionary with start index, end index, 
        # pos_tag for each token
        ents = []
        for tag, span in zip(tags, spans):
            if tag[1] in pos_tags:
                ents.append({"start" : span[0], 
                            "end" : span[1], 
                            "label" : tag[1] })

        doc = {"text" : text, "ents" : ents}

        colors = {"PRON": "blueviolet",
                "VERB": "lightpink",
                "NOUN": "turquoise",
                "ADJ" : "lime",
                "ADP" : "khaki",
                "ADV" : "orange",
                "CONJ" : "cornflowerblue",
                "DET" : "forestgreen",
                "NUM" : "salmon",
                "PRT" : "yellow"}
        
        options = {"ents" : pos_tags, "colors" : colors}
        
        displacy.render(doc, 
                        style = "ent", 
                        options = options, 
                        manual = True,
                    )



#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#INIT | INIT | INIT | INIT | INIT | INIT | INIT | INIT | INIT | INIT | INIT | INIT | INIT | INIT | INIT | INIT | INIT | INIT | INIT | INIT | INIT | INIT | INIT | INIT | INIT | 
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def __init__(self, language = 'italian'):

        self.language = language.lower()

        # Assig to self object
        if self.language == 'italian':
            self.nlp = spacy.load('it_core_news_sm')
        if self.language == 'dutch':
            self.nlp = spacy.load('nl_core_news_sm')
        if self.language == 'english':
            self.nlp = spacy.load('en_core_web_sm')
        
        if self.language not in ['italian', 'english', 'dutch']:
            raise AttributeError("Invalid language input: select one from ['italian', 'english', 'dutch']. Input language is always lowercased by default.")
        
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#CLASS METHODS | CLASS METHODS | CLASS METHODS | CLASS METHODS | CLASS METHODS | CLASS METHODS | CLASS METHODS | CLASS METHODS | CLASS METHODS | CLASS METHODS | CLASS METHODS | 
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def tokenizer(self, text: str) -> list:

        '''A method to be run on the comments through apply to tokenize them.
        The function is meant to be run on the "clean_tweet" column after the other
        preproccesing steps listed above to obtained a new "tokenized_comments"
        column'''
    
        '''Necessary packages/modules:
        - nltk
        - word_tokenize
                        '''
        txt = word_tokenize(text, language=self.language)
        txt = [token for token in txt if token]
        

        return txt
        
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def italian_tokenizer(self, text: str) -> list:

        '''A method to be run on the comments through apply to tokenize them.
        The function is ideantical to the tokenizer above. However, it is meant to be
        used for Italian data: nltk tokenizer does not split on the "'" correctly for Italian. '''
    
        '''Necessary packages/modules:
        - nltk
        - word_tokenize
                        '''
        txt = word_tokenize(text, language=self.language)
        txt = [token for token in txt if token]
        txt = [token for word in txt for token in word.split("'")]

        return txt

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def stop_words_removal(self, tokens: list) -> list:

        '''A method to be run on tokenized comments. It returns the comments striped
        off the stopwords'''
        
        '''Necessary packages/modules:
        - nltk
                            '''
        
        #Stopwords removal
        stop = set(stopwords.words(self.language))
        tokens = [i for i in tokens if i.lower() not in stop]
        
        return tokens
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def content_function_ratio(self, tokens: list) -> float:

        '''A function to calculate the content-function words ratio on the words in a list of tokens.
        of total words.
        '''
        
        stop = set(stopwords.words(self.language))
        
        content = {}
        function = {}

        for word in tokens:
            if word.lower() in stop:
                function[word] =+ 1
            if word.lower() not in stop:
                content[word] =+ 1

        content_count = sum(content.values())
        function_count = sum(function.values())

        if function_count!=0:
        
            return content_count / function_count

        else:
            return content_count
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def lemmatizer(self, tokens: list) -> list:

        '''A method to be run on tokenized comments. It returns the lemmatized comments'''
        
        '''Necessary packages/modules:
        - Spacy
                            '''
        lemmas = []
        text = ' '.join(tokens)


        document = self.nlp(text)
        for token in document:
            lemmas.append(token.lemma_)

        return lemmas

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def pos(self, tokens: list) -> list:

        '''>>>A method to be run on tokenized and lemmatized comments. It returns
        the POS tagging
        
        >>>Necessary packages/modules:
        - NLTK
                          '''
        pos = [item[1] for item in  pos_tag(tokens, tagset='universal') ]
       


        return pos

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    def get_dictionary(self, file_type: str, file_path: str, data_frame = False):

        '''A method that reads .csv and .tsv file, applies the functions and methods described above
        in meaningful order adding columns to the dataframe,
        and finally converts it into a dictionary by index according to the following format:
        {index -> {column -> value}}. The dictionary contains relevant information
        for each comment in the dataset'''

        other_features_names = ["FKRA", "FRE","num_syllables", "avg_syl_per_word", "num_chars", "num_chars_total", \
                                "num_terms", "num_words", "num_unique_words"]
        
        df = self.load(file_type, file_path)
        
        #Identifying the column containing the hate comments
        c_column = int(input(f'What is the index of the column containing the comments? Remember: Index starts from 0 in Python'))
        comments = df.iloc[:, c_column]


        #Applying the functions to extract data and metadata
        df['n_hashtags'] =  df.iloc[:, c_column].apply(cometa.count_hashtags)
        df['n_urls'] =  df.iloc[:, c_column].apply(cometa.count_url)
        df['n_user_tags'] =  df.iloc[:, c_column].apply(cometa.count_user_tags)

        df['clean_comments'] = df.iloc[:, c_column].apply(cometa.preprocessor)
        df['clean_comments'] = df['clean_comments'].apply(cometa.punctuation_removal)

        df['n_emojis'] = df.iloc[:, c_column].apply(cometa.count_emoji)
        df['clean_comments'] = df['clean_comments'].apply(cometa.demojizer)
        
        #Tokenizer choice depending on language
        if self.language == 'italian':
            df['tokenized_comments'] =  df['clean_comments'].apply(self.italian_tokenizer)
        else:
            df['tokenized_comments'] =  df['clean_comments'].apply(self.tokenizer)


        df['length'] =  df['tokenized_comments'].apply(cometa.comment_length)
        df['TTR'] =  df['tokenized_comments'].apply(cometa.type_token_ratio)
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
    def get_summary(self, file_type, file_path, visualize = True) -> tuple:

        '''>>>A class method that returns the relevant data grouping and
        comparing X v. Y
        rather than for each comment individually.

        >>>If visualize is set to True, it also shows a simple visualization of all the
        summarized data'''

        
        output, df = self.get_dictionary(file_type, file_path, data_frame=True)
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



#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#SUBCLASS | TEXT_MANAGER |SUBCLASS | TEXT_MANAGER |SUBCLASS | TEXT_MANAGER |SUBCLASS | TEXT_MANAGER |SUBCLASS | TEXT_MANAGER |SUBCLASS | TEXT_MANAGER |SUBCLASS | TEXT_MANAGER |SUBCLASS | TEXT_MANAGER |
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class TextAnalyser(cometa):

    def __init__(self, language: str, **args):
        super().__init__(language)
        
        
        
    def get_data(self, input_string: str, remove_punctuation = True, remove_emojis = True, remove_stopwords = True, TTR = True, CFR = True, lemmatize = True, pos = True, visualize = True) -> tuple:
        
        '''A function that operates on the text level to extract data and metadata from a comment. It applies a set of tranfromartion to it based on
        boolean arguments and returns a dictionary contatining the relevant information. If visualization is set to true, it 
        
        '''
        self.input_string = input_string
        c_string = cometa.preprocessor(self.input_string)

        #Basic data
        data = {'n_hashtags': cometa.count_hashtags(self.input_string),
        'n_urls': cometa.count_url(self.input_string),
        'n_user_tags': cometa.count_hashtags(self.input_string),
        'n_emojis': cometa.count_emoji(self.input_string),
        'clean_comment': c_string}

        if remove_punctuation:
            c_string = cometa.punctuation_removal(c_string)
        
        if remove_emojis:
            c_string = cometa.demojizer(c_string)

        #Tokenization is always required before stopwords removal/lemmatization
        if self.language == 'italian':
            tokens = cometa.italian_tokenizer(self, c_string)

        else:
            tokens = cometa.tokenizer(self, c_string)

    
        if remove_stopwords:
            tokens = cometa.stop_words_removal(self, tokens)
        
        if TTR:
            data['TTR'] = cometa.type_token_ratio(tokens)
        
        if CFR:
            data['CFR'] = cometa.content_function_ratio(self, tokens)

        if lemmatize:
            tokens = cometa.lemmatizer(self, tokens)
        
        if pos:
            tags = cometa.pos(self, tokens)
            data['pos'] = tags

        data['tokens'] = tokens

        if visualize:
            visualization  = cometa.visualize_pos(tokens=tokens)
        
        return data
