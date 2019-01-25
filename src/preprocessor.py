from nltk.corpus import stopwords
import re
import collections
import itertools
import operator
import io
import string


class Preprocessor:
    def _remove_punctuation(self, text):
        remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
        text = text.translate(remove_punctuation_map)
        return text

    def remove_stop_words(self, file_name):
        if file_name and not file_name.isspace():
            nltk_stop_words = stopwords.words("english")

            file_desc = io.open(file_name, "r", encoding="utf8")
            new_file_desc = io.open(file_name + "_stopWords", "w", encoding="utf8")

            for line in file_desc:
                print(line)
                line = self._removePunctuation(line)
                text = ' '.join([word for word in line.split() if word.lower() not in nltk_stop_words])
                print(text)
                new_file_desc.write(text)

                file_desc.close()
                new_file_desc.close()
        else:
            print
            "not a valid input"
            return

    def removeStemming(self):
        print("stemming")
