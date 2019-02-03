from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk, string

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

    lemmer = nltk.stem.WordNetLemmatizer()

    def LemTokens(tokens):
        return [lemmer.lemmatize(token) for token in tokens]

    remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
    # If we want more meaningful terms in their
    # dictionary forms, lemmatization is preferred.

    def LemNormalize(text):
        return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

    # LemVectorizer = CountVectorizer(tokenizer=LemNormalize, stop_words='english')
    # LemVectorizer.fit_transform(documents)
    # print(LemVectorizer.vocabulary_)
    #
    # tf_matrix = LemVectorizer.transform(documents).toarray()
    # print(tf_matrix)
    #
    # print(tf_matrix.shape)
    #
    # tfidfTran = TfidfTransformer(norm="l2")
    # tfidfTran.fit(tf_matrix)
    # print(tfidfTran.idf_)
    #
    # def idf(n, df):
    #     result = math.log((n + 1.0) / (df + 1.0)) + 1
    #     return result
    #
    # print("The idf for terms that appear in one document: " + str(idf(4, 1)))
    # print("The idf for terms that appear in two documents: " + str(idf(4, 2)))

    # tf_idf = TfidfVec.fit_transform(documents)  # finds the tfidf score with normalization
    #
    # print(cosine_similarity(tf_idf))

    def cos_similarity(textlist):
        # TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')

        TfidfVec = TfidfVectorizer()

        tf_idf = TfidfVec.fit_transform(
            textlist
        )  # finds the tfidf score with normalization

        cos_similarity = cosine_similarity(tf_idf)
        matrix_len = len(cos_similarity)
        for i in range(matrix_len):
            for j in range(matrix_len):
                if i == j:
                    continue

                print("{}, {}: {}".format(i, j, cos_similarity[i][j]))

    #             TODO: write the lines with cosine_similarity on the duplicates files!
    #              TODO: Also set the threshold θ

    cos_similarity(documents)


if __name__ == "__main__":
    main()
