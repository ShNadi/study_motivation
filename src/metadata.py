import pandas as pd
import string


def count_punctuation (df):
    count = lambda l1,l2: sum([1 for x in l1 if x in l2])
    df['count_punct'] = df.motivation.apply(lambda s: count(s, string.punctuation))
    df.to_csv(r'..\data\processed\motivation_liwc_meta.csv', index=False)




if __name__ == "__main__":
    df = pd.read_csv(r'..\data\processed\liwc_results.csv')
    print(count_punctuation (df))


