import csv
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from src.util import Preprocessor
from nltk import word_tokenize
from nltk.corpus import stopwords
import string
from sklearn.pipeline import Pipeline


def read_input(train_file_name, test_file_name: str = None):
    train_df = DataFrame()

    categories = list()
    docs = list()
    with open(train_file_name) as tsv:
        for line in csv.reader(tsv, dialect="excel-tab"):
            if not line[3] == "Content":
                docs.append(line[3])
            if not line[4] == "Category":
                categories.append(line[4])
    train_df["text"] = docs
    train_df["label"] = categories

    train_x, valid_x, train_y, valid_y = train_test_split(
        train_df["text"], train_df["label"]
    )

    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(train_x)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    clf = RandomForestClassifier(n_estimators=200).fit(X_train_tfidf, train_y)
    svc = LinearSVC().fit(X_train_tfidf, train_y)

    test_categories = DataFrame()
    document_ids = list()
    predictions = list()

    from sklearn.pipeline import Pipeline

    Pipeline

    stop_words = set(stopwords.words("english")).union(list(string.punctuation))
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

                # rf_prediction = clf.predict(count_vect.transform([doc]))
                svc_prediction = svc.predict(count_vect.transform([doc]))

                document_ids.append(line[1])
                predictions.append(svc_prediction)

    test_categories["Test_Document_ID"] = document_ids
    test_categories["Predicted_Category"] = predictions

    test_categories.to_csv(
        "../../data/testSet_categories.csv",
        sep="\t",
        encoding="utf-8",
        header=["Test_Document_ID", "Predicted_Category"],
    )


def main():
    read_input("../../data/train_set.csv", "../../data/test_set.csv")


if __name__ == "__main__":
    main()
