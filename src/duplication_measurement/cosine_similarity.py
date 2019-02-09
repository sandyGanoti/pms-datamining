from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk, string
import io
from nltk.corpus import stopwords
import csv
import gensim
from nltk.tokenize import word_tokenize
from pandas import DataFrame

# python3 -m pip install sklearn
# use this command inside ur virtual env if the build from requirements file fails

# There are a few text similarity metrics
# Cosine Similarity is on of the most common ones.

# TFIDF  is a numerical statistic that is intended to reflect how important a word is to a document

# The greater the value of
# θ, the less the value of cos θ, thus the less the similarity between two documents


def main():

    print("Cosine Similarity")

    documents = [
        "I like beer and pizza",
        "I love pizza and pasta",
        "I prefer wine over beer",
        "Thou shalt not pass",
    ]

    # A dictionary of unique terms found
    # make use of the preprocessed data!!!

    # Let n be the number of documents and m be the number of unique terms.
    # Then we have an n by m tf matrix!

    # Inverse document frequency (tf-idf) is an adjustment to term frequency
    # the adjustment deals with the problem that generally speaking certain terms do occur
    # more than others. Thus, tf-idf scales up the importance of rarer terms and scales down the
    # importance of more frequent terms relative to the whole corpus

    # It can be useful to measure similarity
    # not on vanilla bag - of - words matrix, but on transformed one.One choice is to apply tf - idf transformation.

    lemmatizer = nltk.stem.WordNetLemmatizer()
    def lem_tokens(tokens):
        return [lemmatizer.lemmatize(token) for token in tokens]

    remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
    # If we want more meaningful terms in their
    # dictionary forms, lemmatization is preferred.

    def lem_normalize(text):
        return lem_tokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

    def cos_similarity(file_name: str, theta: float=0.7):
        if file_name and not file_name.isspace():
            new_file_desc = io.open("../../data/duplicatePairs.csv", "w", encoding="utf8")
            stop_words = set(stopwords.words("english")).union(list(string.punctuation))
            tf_idf_vec = TfidfVectorizer()

            data_frame = DataFrame.from_csv(file_name, sep='\t')
            number_of_rows = len(data_frame.index)

            for i in range(number_of_rows):
                id0 = data_frame.iloc[i][0]

                document0 = data_frame.iloc[i][2]
                document0 = ' '.join([word for word in word_tokenize(document0) if
                                      word.lower() not in stop_words and not word.isdigit()])
                for j in range(i+1, number_of_rows):
                    id1 = data_frame.iloc[j][0]

                    document1 = data_frame.iloc[j][2]
                    document1 = ' '.join([word for word in word_tokenize(document1) if word.lower() not in stop_words and not word.isdigit()])

                    tf_idf = tf_idf_vec.fit_transform(
                        [document0, document1]
                    )  # finds the tfidf score with normalization

                    cos_similarity = cosine_similarity(tf_idf)
                    matrix_len = len(cos_similarity)
                    for ii in range(matrix_len):
                        for jj in range(ii, matrix_len):
                            # avoid measurement for each doc to itself as obviously the similarity will be 1.0
                            if ii == jj:
                                continue
                            if cos_similarity[ii][jj] > theta:
                            # if cos_similarity[ii][jj] < theta:
                                result = "{}{}{}".format(str(id0) + "\t", str(id1) + "\t", str(cos_similarity[ii][jj]))
                                # print(result)
                                new_file_desc.write(result + "\n")

            new_file_desc.close()

    cos_similarity("../../data/train_set.csv", theta=0.3)

if __name__ == "__main__":
    main()
