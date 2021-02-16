import csv
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from src.util import Preprocessor
from nltk import word_tokenize
from nltk.corpus import stopwords
import string


def read_input(train_file_name, test_file_name: str = None):
    stop_words = set(stopwords.words("english")).union(list(string.punctuation))

    categories = list()
    docs = list()
    # TODO make this with preprocessing a function and call it documentToWords and call it to the diff places
    # probably pull it outshide of this.. maybe to the util preprocessor and make the other methods private
    with open(train_file_name) as tsv:
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
                docs.append(doc)
            if not line[4] == "Category":
                categories.append(line[4])

    # Initialize the "CountVectorizer" object, which is scikit-learn's
    # bag of words tool.
    count_vectorizer = CountVectorizer()
    count_vect_dict = count_vectorizer.fit_transform(docs).toarray()

    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(count_vect_dict)
    X_train_tfidf = X_train_tfidf.toarray()

    clf = RandomForestClassifier(n_estimators=200).fit(X_train_tfidf, categories)
    svc = LinearSVC().fit(X_train_tfidf, categories)

    test_categories = DataFrame()
    document_ids = list()
    predictions = list()

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
                svc_prediction = svc.predict(count_vectorizer.transform([doc]))

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
