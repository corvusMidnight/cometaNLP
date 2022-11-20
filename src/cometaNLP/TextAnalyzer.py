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

class TextAnalyzer:
    '''
    A NLP tool to analyze comments in string format.
    It supports Italian, Dutch, and English comments.
    
    It also contains a series of static methods for independent text analysis.
    It is also possible to extract and analyze entire dataframes of string using the 
    subclass DataWrapper.
    The class has one main behavior:
    It operates on the text level to extract data and metadata from a comment. It applies a set of tranfromartion to it based on
        boolean arguments and returns a dictionary contatining the relevant information. If visualization is set to true, it 
        
        It applies a set of transformation on the string level.
        The user can choose which transformations
        to apply and which to not. If visualize is set to True, it also shows a simple visualization of the pos-tags.
        
        Args:
            self: reference to the current instance of the class
            input_string: Any string
            remove_punctuation (bool): Optionally choose whether to keep (False) or remove (True) punctuation from the input text
            remove_emojis (bool): Optionally choose whether to keep (False) or remove (True) emojis from the input text
            remove_stopwords (bool): Optionally choose whether to keep (False) or remove (True) stopwords from the input text
            TTR (bool): Optionally choose whether to calculate (True) or not (False) the input text TTR
            CFR (bool): Optionally choose whether to calculate (True) or not (False) the input text CFR
            lemmatize (bool): Optionally choose whether to lemmatize (True) or not (False) the input text
            pos (bool): Optionally choose whether to pos-tag (True) or not (False) the input text
            visualize (bool): If set to true, visualizes the pos-tags using nltk color schemes.
        Returns:
            dictionary: a dictionary containing the relevant data and metadata for the input text
        """

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
            df (object): A pandas DataFrame object
            
        
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

        """ A function to be run on comments. It returns the number of hashtags.
        
        Args:
            text (str): Any string
        
        Returns:
            count (int): A count of the number of hashtags contained in the string

        """
        
        count = 0
        hs = re.findall("#[A-Za-z0-9_]+", text)
        for h in hs:
                count +=1
        
        return count
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def count_url(text: str) -> int:

        """A function to be run on comments. It returns the number of urls.

        Args:
            text (str): Any string

        Returns:
            count (int): A count of the number of urls

        """

        count = 0
        urls = re.findall("http\S+", text) + re.findall("www.\S+", text) + re.findall("URL", text) + re.findall("url", text)
        for url in urls:
                count +=1
        
        return count

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def count_user_tags(text: str) -> int:
        
        """A function to be run on comments. It returns the number of user tags.

        Args:
            text (str): Any string

        Returns:
            count (int): A count of the number of user tags

        """

        count = 0
        tags = re.findall("@[A-Za-z0-9_]+", text)
        for user in tags:
                count +=1
        
        return count

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def preprocessor(text: str) -> str:

        """A function to be run on the comments through apply to clean them.
        
        The function applies a series of transformation to the comments. Hashtags,
        urls, and user tags are removed.Digits and leading/trailing spaces are also removed.

        Args:
            text (str): Any string
        
        Returns:
            txt (str): The input text without hashtags, urls, etc.
        
        
        """


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
        
        """A function to be run on comments. It returns comments without punctuation.
        
        Args:
            text (str): Any string

        Returns:
            txt (str): The input text without punctuation

        """
        # remove punctuations and convert characters to lower case
        txt = "".join([char for char in text if char not in '!"#$%&\()*+,-./:;<=>?@[\\]^_`{|}~']) 
    
    
        return txt


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def count_emoji(text: str) -> int:

        
        """A function to be run on comments. It returns the number of emojis.

        Args:
            text (str): Any string

        Returns:
            count (int): A count of the number of emojis

        """
        
        emoji_counter = emoji.emoji_count(text)
        return emoji_counter
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def demojizer(text: str) -> str:

        """A function to be run on comments. It returns the number of urls.

        Args:
            text (str): Any string

        Returns:
            txt (str): The input text without emojis

        """
        txt = emoji.demojize(text)
        
        return txt

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def comment_length(l: list) -> int:
        
        
        """A function to be run on comments. It returns the length of the comments.

        Args:
            text (str): Any string

        Returns:
            count (int): The comment length

        """

        count = len(l)
        
        return count

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def word_counts(l: list) -> dict:

        
        """A function to be run on comments. It returns a dictionary containing the word counts.

        Args:
            l (list): Any list of strings

        Returns:
            counts (dict): A word-counts dictionary

        """
        counts = Counter()
        
        for token in l:
            counts[token] += 1
        
        return counts
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def type_token_ratio(text: str) -> float:

        """"A function to calculate type-token ratio.
        
        A function to calculate the type-token ratio on the words in a string. The type-token
        ratio is defined as the number of unique word types divided by the number
        of total words. ATTENTION: requires the TextAnlyzer.word_counts() to run.

        Args:
            text (str): Any string
        
        Returns:
            
            float: A float expressing the comments TTR

        """
        
        counts = TextAnalyzer.word_counts(text)

        type_count = len(counts.keys())
        token_count = sum(counts.values())

        return type_count / token_count
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def visualize_pos(tokens: list) -> object:

        """A function used to visualize POS-tagged comments.

        visualize_pos() is a void function. If 'visualize = True',
        a displacy visualization will appear on top of the TextAnalyzer
        get_data() output.

        Args:
            tokens (list): Any list of strings
        
        Returns:
            NonValue-Returning
        
        """
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

        if self.language == 'italian' or self.language == 'ita' or self.language == 'it':
            try:
                self.nlp = spacy.load('it_core_news_sm')
            except:
                x = input('Model "it_core_news_sm" is not installed yet, do you want to install it? Y | N')
                if x == 'Y':
                    print('Downloading model...')
                    os.system('python -m spacy download it_core_news_sm')
                    print('Downloaded!')
                    self.nlp = spacy.load('it_core_news_sm')

        if self.language == 'dutch' or self.language == 'nld' or self.language == 'nl':
            try:
                self.nlp = spacy.load('nl_core_news_sm')
            except:
                x = input('Model "nl_core_news_sm" is not installed yet, do you want to install it? Y | N')
                if x == 'Y':
                    print('Downloading model...')
                    os.system('python -m spacy download nl_core_news_sm')
                    print('Downloaded!')
                    self.nlp = spacy.load('nl_core_news_sm')

        if self.language == 'english' or self.language == 'eng' or self.language == 'en':
            try:
                self.nlp = spacy.load('en_core_web_sm')
            except:
                x = input('Model "en_core_web_sm" is not installed yet, do you want to install it? Y | N')
                if x == 'Y':
                    print('Downloading model...')
                    os.system('python -m spacy download en_core_web_sm')
                    print('Downloaded!')
                    self.nlp = spacy.load('en_core_web_sm')

        if self.language not in ['italian', 'english', 'dutch']:
            raise AttributeError("Invalid language input: select one from ['italian', 'english', 'dutch']. Input language is always lowercased by default.")
        
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#CLASS METHODS | CLASS METHODS | CLASS METHODS | CLASS METHODS | CLASS METHODS | CLASS METHODS | CLASS METHODS | CLASS METHODS | CLASS METHODS | CLASS METHODS | CLASS METHODS | 
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def tokenizer(self, text: str) -> list:

        """A class method to tokenize text. It returns the tokenized comments.
        
        A class method to be run on the comments through apply to tokenize them.
        The function is applied on the "clean_comments" column after the other
        preproccesing steps listed above to obtained a new "tokenized_comments"
        column.

        Args:
            self: reference to the current instance of the class
            text (str): Any string

        Returns:
            txt (list): A list of strings. The tokenized input text.
        
        """
        
        try:
            txt = word_tokenize(text, language=self.language)
            txt = [token for token in txt if token]
        
        except LookupError:
            
            x = input('Model "punkt" is not installed yet, do you want to install it? Y | N')
            if x.lower() == 'y':
                    print('Downloading model...')
                    nltk.download('punkt')
                    print('Downloaded!')
                    txt = word_tokenize(text, language=self.language)
                    txt = [token for token in txt if token]
            
            else:
                    txt = ModuleNotFoundError
                    print('Please download "punkt" to used cometaNLP tokenizer')


        return txt
        
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def italian_tokenizer(self, text: str) -> list:

        """A class method to tokenize Italian text. It returns the tokenized text.
        
        A class method to be run on the comments through apply to tokenize them.
        The function is ideantical to the tokenizer above. However, it is meant to be
        used for Italian data: nltk tokenizer does not split on the "'" correctly for Italian.
    
        Args:
            self: reference to the current instance of the class
            text (str): Any string

        Returns:
            txt (list): A list of strings. The tokenized input text.
        
        """
        try:

            txt = word_tokenize(text, language=self.language)
            txt = [token for token in txt if token]
            txt = [token for word in txt for token in word.split("'")]

        except LookupError:
            
            x = input('Model "punkt" is not installed yet, do you want to install it? Y | N')
            if x.lower() == 'y':

                    print('Downloading model...')
                    nltk.download('punkt')
                    print('Downloaded!')
                    txt = word_tokenize(text, language=self.language)
                    txt = [token for token in txt if token]
                    txt = [token for word in txt for token in word.split("'")]
            
            else:
                    txt = ModuleNotFoundError
                    print('Please download "punkt" to used cometaNLP tokenizer')

        return txt

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def stop_words_removal(self, tokens: list) -> list:

        """A class method to be run on tokenized comments. It returns the comments striped off the stopwords.

        Args:
            self: reference to the current instance of the class
            tokens (list): Any list of strings

        Returns:
            tokens (list): The input tokenized text stripped off the stop words
        
        """
        
        #Stopwords removal
        stop = set(stopwords.words(self.language))
        tokens = [i for i in tokens if i.lower() not in stop]
        
        return tokens
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def content_function_ratio(self, tokens: list) -> float:

        """A class method to be run on tokenized comments. It returns the content-function words ratio.

        If the number of function words is equal to 0, the returned digit expresses the number of content words in the text

        Args:
            self: reference to the current instance of the class
            tokens (list): Any list of strings

        Returns:
            float: The content-function words ratio
        
        """

        
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

        """A class method to be run on tokenized text. It returns the lemmatized comments'''
         
        Args:
            self: reference to the current instance of the class
            tokens (list): Any list of strings

        Returns:
            lemmas (list): The lemmas of the input tokenized text
        
        """

        lemmas = []
        text = ' '.join(tokens)


        document = self.nlp(text)
        for token in document:
            lemmas.append(token.lemma_)

        return lemmas

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def pos(self, tokens: list) -> list:

        """A class method to be run on tokenized and/or lemmatized comments. It returns
        the POS tagging.

        Args:
            self: reference to the current instance of the class
            tokens (list): Any list of strings

        Returns:
            pos (list): The pos tags of the input tokenized text
        
        """
        pos = [item[1] for item in  pos_tag(tokens, tagset='universal') ]
       

        return pos

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

 
    def __call__(self, input_string: str, remove_punctuation = True, remove_emojis = True, remove_stopwords = True, TTR = True, CFR = True, lemmatize = True, pos = True, visualize = True) -> dict:
        
        """A class that operates on the text level to extract data and metadata from a comment. It applies a set of tranfromartion to it based on
        boolean arguments and returns a dictionary contatining the relevant information. If visualization is set to true, it 
        
        It applies a set of transformation on the string level.
        The user can choose which transformations
        to apply and which to not. If visualize is set to True, it also shows a simple visualization of the pos-tags.
        
        Args:
            self: reference to the current instance of the class
            input_string: Any string
            remove_punctuation (bool): Optionally choose whether to keep (False) or remove (True) punctuation from the input text
            remove_emojis (bool): Optionally choose whether to keep (False) or remove (True) emojis from the input text
            remove_stopwords (bool): Optionally choose whether to keep (False) or remove (True) stopwords from the input text
            TTR (bool): Optionally choose whether to calculate (True) or not (False) the input text TTR
            CFR (bool): Optionally choose whether to calculate (True) or not (False) the input text CFR
            lemmatize (bool): Optionally choose whether to lemmatize (True) or not (False) the input text
            pos (bool): Optionally choose whether to pos-tag (True) or not (False) the input text
            visualize (bool): If set to true, visualizes the pos-tags using nltk color schemes.
        Returns:
            dictionary: a dictionary containing the relevant data and metadata for the input text
        """
        
        self.input_string = input_string
        c_string = TextAnalyzer.preprocessor(self.input_string)

        #Basic data
        data = {'n_hashtags': TextAnalyzer.count_hashtags(self.input_string),
        'n_urls': TextAnalyzer.count_url(self.input_string),
        'n_user_tags': TextAnalyzer.count_hashtags(self.input_string),
        'n_emojis': TextAnalyzer.count_emoji(self.input_string),
        'clean_comment': c_string}

        if remove_punctuation:
            c_string = TextAnalyzer.punctuation_removal(c_string)
        
        if remove_emojis:
            c_string = TextAnalyzer.demojizer(c_string)

        #Tokenization is always required before stopwords removal/lemmatization
        if self.language == 'italian':
            tokens = TextAnalyzer.italian_tokenizer(self, c_string)

        else:
            tokens = TextAnalyzer.tokenizer(self, c_string)

    
        if remove_stopwords:
            tokens = TextAnalyzer.stop_words_removal(self, tokens)
        
        if TTR:
            data['TTR'] = TextAnalyzer.type_token_ratio(tokens)
        
        if CFR:
            data['CFR'] = TextAnalyzer.content_function_ratio(self, tokens)

        if lemmatize:
            tokens = TextAnalyzer.lemmatizer(self, tokens)
        
        if pos:
            tags = TextAnalyzer.pos(self, tokens)
            data['pos'] = tags

        data['tokens'] = tokens

        if visualize:
            visualization  = TextAnalyzer.visualize_pos(tokens=tokens)
        
        return data
