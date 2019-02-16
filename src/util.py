from nltk.stem import PorterStemmer
from textblob import Word
from textblob import TextBlob
import re

class Preprocessor:
    """ Defines some actions in order to pre process data

      :param str data: the data will be used for generating the word cloud
      """

    # removal of suffices, like “ing”, “ly”, “s”--Dont use it
    # Lemmatization is a more effective option than stemming because it converts
    # the word into its root word, rather than just stripping the suffices.
    # It makes use of the vocabulary and does a morphological analysis to obtain the root word.
    # Therefore, we usually prefer using lemmatization over stemming.

    @staticmethod
    def remove_stemming(line: str)-> str:
        stemming = PorterStemmer()

        return " ".join([stemming.stem(word) for word in line.split()])

    @staticmethod
    def remove_non_words(line: str)-> str:
        return re.sub("[^a-zA-Z]"," ", line)

    @staticmethod
    def lemmatization(line: str) -> str:
        return " ".join([Word(word).lemmatize() for word in line.split()])

    @staticmethod
    def spelling_correction(line: str) -> str:
        return " ".join([str(TextBlob(word).correct()) for word in line.split()])

#
#
# #import WordNet Lemmatizer from nltk
# from nltk.stem import WordNetLemmatizer
# wordnet_lemmatizer = WordNetLemmatizer()
#
# lines_with_lemmas=[]
# #stop words contain the set of stop words
# for line in lines:
#  temp_line=[]
#  for word in lines:
#   temp_line.append (wordnet_lemmatizer.lemmatize(word))
#  string=' '
#  lines_with_lemmas.append(string.join(temp_line))
# lines=lines_with_lemmas