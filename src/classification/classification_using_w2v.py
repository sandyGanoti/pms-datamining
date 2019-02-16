import csv
from typing import List
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from collections import defaultdict
from src.util import Preprocessor
from nltk import word_tokenize
from nltk.corpus import stopwords
import string
import nltk
from sklearn.pipeline import Pipeline
from gensim.models import Word2Vec
import numpy

def document_to_sentences(document: str)->List[str]:
    document = " ".join([word for word in word_tokenize(document) if not word.isdigit()])
    document = Preprocessor.remove_non_words(document)
    document = Preprocessor.lemmatization(document)
    document = Preprocessor.remove_stemming(document)

    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
    return document.split()


def word_to_vec(train_file_name: str, test_file_name: str):
    categories = list()
    sentences = list()
    i=0
    with open(train_file_name) as tsv:
        for line in csv.reader(tsv, dialect="excel-tab"):
            if not line[3] == "Content":
                i += 1
                if i == 2:
                    break
                sentences.append(document_to_sentences(line[3]))
            if not line[4] == "Category":
                categories.append(line[4])
    print(len(sentences))

    # Set values for various parameters
    num_features = 300  # Word vector dimensionality
    min_word_count = 40  # Minimum word count
    num_workers = 4  # Number of threads to run in parallel
    context = 10  # Context window size
    downsampling = 1e-3  # Downsample setting for frequent words

    model = Word2Vec(sentences, workers=num_workers, \
                              size=num_features, min_count=min_word_count, \
                              window=context, sample=downsampling)
    # model.init_sims(replace=True)
    # model.train(sentences, total_examples=len(sentences), epochs=10)
    w2v = dict(zip(model.wv.index2word, model.wv.syn0))


    trainDataVecs = get_avg_feature_vecs(sentences, model, num_features)

    testDataVecs = get_avg_feature_vecs(['sandu is here'], model, num_features)

    forest = RandomForestClassifier(n_estimators=100)
    forest = forest.fit(trainDataVecs, categories)
    result = forest.predict(testDataVecs)

    print(result)

def main():
    word_to_vec("../../data/train_set.csv", "../../data/test_set.csv")


def make_feature_vec(words, model, num_features):
    """
    Average the word vectors for a set of words
    """
    feature_vec = numpy.zeros((num_features,), dtype="float32")  # pre-initialize (for speed)
    nwords = 0.
    index2word_set = set(model.wv.index2word)  # words known to the model

    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            feature_vec = numpy.add(feature_vec, model[word])

    feature_vec = numpy.divide(feature_vec, nwords)
    return feature_vec

# //TODO na allakse ta reviews se documents
def get_avg_feature_vecs(reviews, model, num_features):
    """
    Calculate average feature vectors for all reviews
    """
    counter = 0.
    review_feature_vecs = numpy.zeros((len(reviews), num_features), dtype='float32')  # pre-initialize (for speed)

    for review in reviews:
        review_feature_vecs[counter] = make_feature_vec(review, model, num_features)
        counter = counter + 1.
    return review_feature_vecs

if __name__ == "__main__":
    main()
