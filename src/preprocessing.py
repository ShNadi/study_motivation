import string
from nltk.corpus import stopwords


def remove_punctuation(text):
    s = ''.join([i for i in text if i not in frozenset(string.punctuation)])
    return s


def remove_stopwords(text):
    stopword_list = list(stopwords.words('Dutch'))
    text = ' '.join([word for word in text.split() if word not in stopword_list])
    return text

def clean_text(df):
    df['motivation'] = df.apply(lambda x: remove_punctuation(x['motivation']), axis=1)
    df['motivation'] = df.apply(lambda x: remove_stopwords(x['motivation']), axis=1)
    df.dropna(inplace=True, subset=['motivation'])
    print('stopwords and punctuations are removed from the text!')
    return df
# def stem_text(df):

