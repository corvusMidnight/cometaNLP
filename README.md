# cometaNLP
![PyPI](https://img.shields.io/pypi/v/cometaNLP?label=pypi%20package)
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
- `count_hashtags`
- `count_url`
- `count_user_tags`
- `preprocessor`
- `punctuation_removal`
- `count_emoji`
- `demojizer`
- `comment_length`
- `word_counts`
- `type_token_ration`
- `visualize_pos`


# Authors
- Leonardo Grotti <a href="https://www.linkedin.com/in/leonardo-grotti-a8a64a205/" target="_blank"><img src="https://img.shields.io/badge/LinkedIn-%230077B5.svg?&style=flat-square&logo=linkedin&logoColor=white" alt="LinkedIn"></a>

# Q&A

**Q**: *Why does cometa start from distribution 0.0.3?*

**A**: The first two distributions of the package were erased due to some naming conflicts. 
