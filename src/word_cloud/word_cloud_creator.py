from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plot


class WordCloudGenerator:
    """ Creates a word cloud image using data user applies

      :param str data: the data will be used for generating the word cloud
      """
    @staticmethod
    def create_word_cloud(data):
        stop_words = set(STOPWORDS)
        stop_words.add("will")

        word_cloud = WordCloud(stopwords=stop_words, background_color='black', width=1200, height=1000).generate(data)

        plot.imshow(word_cloud)
        plot.axis('off')
        plot.savefig('../data/word_cloud.png', dpi=300)
        plot.show()
