from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# text classification
# In order to run machine learning algorithms we need to convert the text files
# into numerical feature vectors
# We will be using bag of words model for our example


def main():

    print("svm_bow")
    twenty_train = fetch_20newsgroups(subset="train", shuffle=True)

    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(twenty_train.data)
    X_train_counts.shape

    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    print(X_train_tfidf.shape)

    clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)

    # use Pipeline instead of the above code to avoid write all these
    text_clf = Pipeline(
        [
            ("vect", CountVectorizer()),
            ("tfidf", TfidfTransformer()),
            ("clf", MultinomialNB()),
        ]
    )
    text_clf = text_clf.fit(twenty_train.data, twenty_train.target)


if __name__ == "__main__":
    main()


def doc_to_bow(dictionary, data):
    return list(map(lambda doc: dictionary.doc2bow(doc), data))


def create_bow(file_name):
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

    documents = list(read_input(file_name))

    # Initialize and train the model
    w2v_model = Word2Vec(documents, size=150, window=10, min_count=2, workers=10)
    w2v_model.train(documents, total_examples=len(documents), epochs=10)
