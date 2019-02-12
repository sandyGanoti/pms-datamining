from typing import Set
from nltk.corpus import stopwords
import io
import string
from textblob import TextBlob
from nltk.stem import PorterStemmer
from textblob import Word
import csv
from nltk import word_tokenize
from .repeated_categories import Util
from .word_cloud_creator import WordCloudGenerator


class Preprocessor:
    """ Defines some actions in order to pre process data

      :param str data: the data will be used for generating the word cloud
      """
    @staticmethod
    def _spelling_correction(line: str)-> str:
        return ' '.join([str(TextBlob(word).correct()) for word in line.split()])

    # removal of suffices, like “ing”, “ly”, “s”--Dont use it
    # Lemmatization is a more effective option than stemming because it converts
    # the word into its root word, rather than just stripping the suffices.
    # It makes use of the vocabulary and does a morphological analysis to obtain the root word.
    # Therefore, we usually prefer using lemmatization over stemming.
    @staticmethod
    def _remove_stemming(line: str):
        stemming = PorterStemmer()

        return ' '.join([stemming.stem(word) for word in line.split()])

    @staticmethod
    def _lemmatization(line: str)-> str:
        return ' '.join([Word(word).lemmatize() for word in line.split()])

    @staticmethod
    def preprocess_and_store_content(file_name: str, most_repeated_categories: Set[str]):
        stop_words = set(stopwords.words("english")).union(list(string.punctuation))

        new_file_desc = io.open(file_name.replace(".csv", "") + "_preprocessed.csv", "w", encoding="utf8")

        if file_name and not file_name.isspace():
            with open(file_name) as tsv:
                for line in csv.reader(tsv, dialect="excel-tab"):
                    if line[4] not in most_repeated_categories:
                        continue

                    line = line[3]
                    line = ' '.join([word for word in word_tokenize(line) if word.lower() not in stop_words and not word.isdigit()])
                    line = Preprocessor._lemmatization(line)
                    # line = Preprocessor.spelling_correction(line)
                    line = Preprocessor._remove_stemming(line)

                    new_file_desc.write(line + "\n")

        new_file_desc.close()


def main():

    print("sandu")

    # most_repeated_categories will return
    # {'Business': 2735, 'Football': 3121, 'Politics': 2683, 'Film': 2240, 'Technology': 1487}
    # for the train_set
    most_repeated_categories = Util.most_repeated_categories(5, "../data/train_set.csv")

    # store content of the x most repeated categories on a new file
    # preprocess the new file in order to clean your data and feed them to word_cloud_generator
    Preprocessor.preprocess_and_store_content("../data/train_set.csv", most_repeated_categories)

    fileDesc = open("../../data/train_set_preprocessed.csv", "r")
    data = fileDesc.read();
    WordCloudGenerator.create_word_cloud(data)


if __name__ == '__main__':
    main()