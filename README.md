# cometaNLP
[![PyPI license](https://img.shields.io/pypi/l/ansicolortags.svg)](https://pypi.python.org/pypi/ansicolortags/)
![PyPI - Downloads](https://img.shields.io/pypi/dm/cometaNLP)
[![Python 3.7](https://img.shields.io/badge/python->=3.7-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)


A NLP Comments Extraction and Text Analysis tool for Italian, Dutch, and English social media comments.

# Installation

### Windows
`py -m pip install cometaNLP`

### Unix\MacOS
`python3 -m pip install cometaNLP`

### .ipynb
`!pip install cometaNLP`

# Supported languages
 ![](https://img.shields.io/badge/languages-italian|dutch|english-red)

# Functions

cometa is composed by one main module named `COMETA`. The module has two main class methods:

**(1)** `get_dictionary`

        """A class method that applies a series of tranformations to csv and tsv files
        and returns a dictionary
        
        get_dictionary() reads .csv and .tsv file, applies the functions and methods within the
        the COMETA module to the column contaning the comments/text in the dataframe
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

**(2)** `gert_summary`
 
        """A class method that returns the relevant data comparison based on grouping 
        comparison (e.g., X v. Y) rather than for each comment individually.
        
        get_summary() is built upon the get_dictionary method. If visualize is set to True, it also shows a simple visualization of all the
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

And a subclass functionality to analyze single instances of text:

**(3)** `get_data`
        
        """A class method that operates on the text level to extract data and metadata from a comment. It applies a set of tranfromartion to it based on
        boolean arguments and returns a dictionary contatining the relevant information. If visualization is set to true, it 
        
        get_data() is built upon the get_dictionary method. It applies the the same set of transformation on the string level.
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

# Static methods

The COMETA module offers a set of `@staticmethods` that can be used independently from its main class methods:

- `load`

        """A method that reads .csv and .tsv files.
        
       
        Args:
            file_type (str): The type of file containing the data (csv/tsv)
            file_path (str): The path to the file where the file is stored.
          
        Returns:
            df (object): A pandas DataFrame object
            
        
        """

- `count_hashtags`

        """A function to be run on comments. It returns the number of hashtags.
        
        Args:
            text (str): Any string
        
        Returns:
            count (int): A count of the number of hashtags contained in the string
        """
- `count_url`

        """A function to be run on comments. It returns the number of urls.
        Args:
            text (str): Any string
        Returns:
            count (int): A count of the number of urls
        """

- `count_user_tags`

        """A function to be run on comments. It returns the number of user tags.
        Args:
            text (str): Any string
        Returns:
            count (int): A count of the number of user tags
        """

- `preprocessor`

        """A function to be run on the comments through apply to clean them.
        
        The function applies a series of transformation to the comments. Hashtags,
        urls, and user tags are removed.Digits and leading/trailing spaces are also removed.
        
        Args:
            text (str): Any string
        
        Returns:
            txt (str): The input text without hashtags, urls, etc.
        
        
        """

- `punctuation_removal`

        """A function to be run on comments. It returns comments without punctuation.
        
        Args:
            text (str): Any string
        Returns:
            txt (str): The input text without punctuation
        """

- `count_emoji`

        """A function to be run on comments. It returns the number of emojis.
        
        Args:
            text (str): Any string
        Returns:
            count (int): A count of the number of emojis
        """

- `demojizer`

        """A function to be run on comments. It returns the number of urls.
        
        Args:
            text (str): Any string
        Returns:
            txt (str): The input text without emojis
        """

- `comment_length`

        """A function to be run on comments. It returns the length of the comments.
        
        Args:
            text (str): Any string
        Returns:
            count (int): The comment length
        """

- `word_counts`

        """A function to be run on comments. It returns a dictionary containing the word counts.
        
        Args:
            l (list): Any list of strings
        Returns:
            counts (dict): A word-counts dictionary
        """

- `type_token_ration`

        """"A function to calculate type-token ratio.
        
        A function to calculate the type-token ratio on the words in a string. The type-token
        ratio is defined as the number of unique word types divided by the number
        of total words. ATTENTION: requires the COMETA.word_counts() to run.
        
        Args:
            text (str): Any string
        
        Returns:
            
            float: A float expressing the comments TTR
        """

- `visualize_pos`

        """A function used to visualize POS-tagged comments.
        visualize_pos() is a void function. If 'visualize = True',
        a displacy visualization will appear on top of the TextAnalyzer
        get_data() output.
        
        Args:
            tokens (list): Any list of strings
        
        Returns:
            NonValue-Returning
        
        """


# Authors
- Leonardo Grotti <a href="https://www.linkedin.com/in/leonardo-grotti-a8a64a205/" target="_blank"><img src="https://img.shields.io/badge/LinkedIn-%230077B5.svg?&style=flat-square&logo=linkedin&logoColor=white" alt="LinkedIn"></a>

# Q&A

**Q**: *Why does cometa start from distribution 0.0.3?*

**A**: The first two distributions of the package were erased due to some naming conflicts. 
