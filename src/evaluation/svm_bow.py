from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

import csv
import string
from gensim.utils import simple_preprocess
from nltk import word_tokenize
from src.util import Preprocessor

# text classification
# In order to run machine learning algorithms we need to convert the text files
# into numerical feature vectors
# We will be using bag of words model for our example


def doc_to_bow(dictionary, data):
    return list(map(lambda doc: dictionary.doc2bow(doc), data))


def create_bow(file_name, test_file_name):
    # if file_name and not file_name.isspace():
    #     with open(file_name, "r") as fd:
    #         reader = csv.reader(fd)
    #         data = [r for r in reader]
    #         dictionary = Dictionary(documents=data)
    #
    # print("Found {} words.".format(len(dictionary.values())))
    #
    # dictionary.filter_extremes(no_below=2)
    # dictionary.compactify()  # Reindexes the remaining words after filtering
    # print("Left with {} words.".format(len(dictionary.values())))
    #
    # doc2bow = doc_to_bow(dictionary, data)
    #
    # Set values for various parameters
    num_features = 200  # Word vector dimensionality
    min_word_count = 2  # Minimum word count
    num_workers = 4  # Number of threads to run in parallel
    context = 6  # Context window size
    downsampling = 1e-3  # Downsample setting for frequent words
    #
    # documents = list(read_input(file_name))
    #
    # # Initialize and train the model
    # w2v_model = Word2Vec(documents, size=150, window=10, min_count=2, workers=10)
    # w2v_model.train(documents, total_examples=len(documents), epochs=10)
    stop_words = set(stopwords.words("english")).union(list(string.punctuation))

    categories = list()
    documents = list()
    with open(file_name) as tsv:
        for line in csv.reader(tsv, dialect="excel-tab"):
            if not line[4] == "Category":
                categories.append(line[4])

            if not line[3] == "Content":
                # a list of tokens
                tokens = simple_preprocess(line[3])
                documents.append(tokens)

    model = Word2Vec(size=150, window=10, min_count=2, sg=1, workers=10)
    model.build_vocab(documents)
    model.train(sentences=documents, total_examples=len(documents), epochs=model.iter)

    vectorizer = CountVectorizer(
        analyzer="word",
        tokenizer=None,
        preprocessor=None,
        stop_words=None,
        max_features=5000,
    )

    train_data = vectorizer.fit_transform(documents)
    train_data = train_data.toarray()

    word_vectors = model.wv
    print(len(word_vectors.vocab))

    rf = RandomForestClassifier(n_estimators=200, n_jobs=-1)
    rf.fit(train_data, categories)

    stop_words = set(stopwords.words("english")).union(list(string.punctuation))
    test_documents = list()
    with open(test_file_name) as tsv:
        for line in csv.reader(tsv, dialect="excel-tab"):
            if not line[3] == "Content":
                doc = line[3]
                doc = " ".join(
                    [
                        word
                        for word in word_tokenize(doc)
                        if word.lower() not in stop_words and not word.isdigit()
                    ]
                )
                doc = Preprocessor.lemmatization(doc)
                doc = Preprocessor.remove_stemming(doc)
                test_documents.append(doc)

    test_data = vectorizer.transform(test_documents)
    test_data = test_data.toarray()

    rf_prediction = rf.predict_proba(test_data)
    print(rf_prediction)


def main():
    create_bow("../../data/train_set.csv", "../../data/test_set.csv")


if __name__ == "__main__":
    main()
