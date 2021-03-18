from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np

def tfidf_words(c=1, n=20):
    df = pd.read_csv(r'..\data\processed\motivation_liwc_meta.csv')

    df_class = df[df['bsa_dummy'] == c]
    vect = TfidfVectorizer(min_df=5).fit(df_class['motivation'])
    X = vect.transform(df['motivation'])

    max_value = X.max(axis=0).toarray().ravel()
    sorted_by_tfidf = max_value.argsort()
    feature_names = np.array(vect.get_feature_names())
    print('\n {} features with highest tfidf for class {}:\n{} '.format(n, c, feature_names[sorted_by_tfidf[-n:]]))

if __name__=='__main__':
    tfidf_words()