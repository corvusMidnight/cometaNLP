# Changelog

## v0.0.3a

- Added documentation (Google Style) for the source code
- Added badges and information to package README.md 

## v0.0.4

- Fixed documentation for the source code
- Added badges and information to package README.md 

## v0.0.5

- Fixed documentation for the source code and README.md
- Added information to package README.md 

## v0.0.6

- Fixed an error which caused users who did not have spacy language models installed to run into an error
- Corrected typos in README.md

## v0.0.7

- Fixed an error which causes users who did not have nltk's "punkt" module downaloaded to run into a lookupError

## v0.0.8

- Changed module structure: COMETA is now two different module. TextAnalyzer (superclass) and DataWrapper (subclass). TextAnalyzer has now one main class behavior: analyzing text strings. It replaces the old 'get_data' method from the old 'TextAnalyzer' subclass. The old 'get_dictionary' and 'get_summary' functions of the COMETA module have been replaced by the 'data_wrapper' and 'data_wrapper_summary' functions in the DataWrapper module, which also inherits all the static methods and functions of its parent class. 'TextAnalyzer'.

- The README.md has been rewritten accordignly and now contains a table of content and code demo.
