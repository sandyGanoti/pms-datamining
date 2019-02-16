from nltk.stem import PorterStemmer
from textblob import Word
from textblob import TextBlob


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
    def remove_stemming(line: str):
        stemming = PorterStemmer()

        return " ".join([stemming.stem(word) for word in line.split()])

    @staticmethod
    def lemmatization(line: str) -> str:
        return " ".join([Word(word).lemmatize() for word in line.split()])

    @staticmethod
    def spelling_correction(line: str) -> str:
        return " ".join([str(TextBlob(word).correct()) for word in line.split()])
