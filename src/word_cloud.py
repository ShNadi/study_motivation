import pandas as pd
from wordcloud import WordCloud
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import load_dataset


def remove_stopwords(text):
    stopword_list = list(stopwords.words('Dutch'))
    text = ' '.join([word for word in text.split() if word not in stopword_list])
    return text

def show_wordcloud(df, n=100):

    # df['removed_stopwords'] = df.apply(lambda x: remove_stopwords(x['motivation']), axis=1)

    # Plot wordcloud of 100 most frequent words
    text = ' '.join(df['motivation'])
    wordcloud = WordCloud(
        width=3000,
        height=2000,
        background_color='white',
        max_words=n
    ).generate(text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.title("Wordcloud of 100 most frequent words")
    plt.show()


if __name__ == "__main__":
    df = load_dataset.read_df()
    show_wordcloud(df, 100)


