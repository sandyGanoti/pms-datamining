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
    twenty_train = fetch_20newsgroups(subset='train', shuffle=True)

    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(twenty_train.data)
    X_train_counts.shape

    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    print(X_train_tfidf.shape)

    clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)

    # use Pipeline instead of the above code to avoid write all these
    text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),('clf', MultinomialNB())])
    text_clf = text_clf.fit(twenty_train.data, twenty_train.target)



if __name__ == '__main__':
    main()