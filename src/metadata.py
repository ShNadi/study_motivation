import pandas as pd
import string
from nltk.corpus import stopwords


def count_punctuation(df):
    # Counts the number of punctuations in the text
    count = lambda l1, l2: sum([1 for x in l1 if x in l2])
    df['count_punct'] = df.motivation.apply(lambda s: count(s, string.punctuation))
    df.to_csv(r'..\data\processed\motivation_liwc_meta.csv', index=False)


def count_stopwords(df):
    # Counts the number of stopwords in the text
    stopword_list = set(stopwords.words('Dutch'))
    df['count_stopwords'] = df['motivation'].str.split().apply(lambda x: len(set(x) & stopword_list))
    df.to_csv(r'..\data\processed\motivation_liwc_meta.csv', index=False)




# Add extra columns to the dataset
if __name__ == "__main__":
    # df = pd.read_csv(r'..\data\processed\liwc_results.csv')
    # print(count_punctuation(df))

    df = pd.read_csv(r'..\data\processed\motivation_liwc_meta.csv')
    count_stopwords(df)






