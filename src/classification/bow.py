from gensim.corpora import Dictionary
import csv
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


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


def read_input(train_file_name, test_file_name: str= None):
    trainDF = DataFrame()

    categories = list()
    docs = list()
    with open(train_file_name) as tsv:
        for line in csv.reader(tsv, dialect="excel-tab"):
            if not line[3] == "Content":
                docs.append(line[3])
            if not line[4] == "Category":
                categories.append(line[4])
    trainDF['text'] = docs
    trainDF['label'] = categories
    print("sandu")

    models = [
        RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
        # linear used because linear should scale better to large numbers of samples.
        LinearSVC()
    ]

    train_x, valid_x, train_y, valid_y = train_test_split(trainDF['text'], trainDF['label'])

    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(train_x)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    clf = RandomForestClassifier().fit(X_train_tfidf, train_y)

    svc = LinearSVC().fit(X_train_tfidf, train_y)

    a = "Jeremy Corbyn gives Labour MPs free vote 	 is to offer a free vote " \
        "to MPs on David Cameron’s proposals for UK to bomb Isis in Syria but " \
        "will make clear that Labour party policy is to oppose airstrikes. The " \
        "leader will also press Cameron to delay the vote until Labour’s concerns " \
        "about the justification for the bombing are addressed, as part of a deal " \
        "he has thrashed out with the deputy leader, Tom Watson, and other senior " \
        "members of the shadow cabinet over the weekend. His decision averts the " \
        "threat of a mass shadow cabinet walkout while making it clear that his own " \
        "firmly held opposition to airstrikes is official Labour party policy, backed " \
        "by the membership. It will also create a dilemma for Downing Street about" \
        " whether to press ahead with the vote this week, because undecided Labour " \
        "MPs are likely to be tempted to back Corbyn’s call for a longer timetable. " \
        "Cameron has been expected to try for a vote on Wednesday but he has" \
        " said he will not do so unless he is sure there is a clear majority" \
        " in favour of strikes. It is understood has been no discussion with No 10 " \
        "about Labour’s proposals to put off the vote."

    print(clf.predict(count_vect.transform([a])))
    print(svc.predict(count_vect.transform([a])))

    # encoder = LabelEncoder()
    # train_y = encoder.fit_transform(train_y)
    # valid_y = encoder.fit_transform(valid_y)
    #
    # token = text.Tokenizer()
    # token.fit_on_texts(trainDF['text'])
    # word_index = token.word_index


def main():
    # create_bow("../../data/train_set_preprocessed.csv")
    read_input("../../data/train_set.csv")

if __name__ == '__main__':
    main()