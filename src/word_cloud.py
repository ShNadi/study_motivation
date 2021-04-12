import pandas as pd
from wordcloud import WordCloud
from nltk.corpus import stopwords
import matplotlib.pyplot as plt


def remove_stopwords(text):
    stopword_list = list(stopwords.words('Dutch'))
    text = ' '.join([word for word in text.split() if word not in stopword_list])
    return text

def show_wordcloud(df, n=100):

    df['motivation'] = df.motivation.str.lower()

    df['removed_stopwords'] = df.apply(lambda x: remove_stopwords(x['motivation']), axis=1)

    # Plot wordcloud of 100 most frequent words
    text = ' '.join(df['removed_stopwords'])
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
    df = pd.read_csv('../data/processed/motivation.csv')
    # show_wordcloud(df, 100)

    # bsadummy_class0 = df[df['bsa_dummy'] == 0]
    # show_wordcloud(bsadummy_class0, 100)

    # bsadummy_class1 = df[df['bsa_dummy'] == 1]
    # show_wordcloud(bsadummy_class1, 100)

    # reenrolled_class0 = df[df['reenrolled'] == 0]
    # show_wordcloud(reenrolled_class0, 100)

    # reenrolled_class1 = df[df['reenrolled'] == 1]
    # show_wordcloud(reenrolled_class1, 100)




