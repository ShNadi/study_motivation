import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize
from nltk.tag.perceptron import PerceptronTagger
import nltk.data
import re
import gensim
from gensim.utils import simple_preprocess
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import gensim.corpora as corpora
from pprint import pprint
from contextlib import redirect_stdout
import pickle
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import os
from sklearn.utils import shuffle

# pyLDAvis.enable_notebook()

stop_words = stopwords.words('Dutch')


def lda_func(num_topics=10):
    """ LDA Topic modeling function

    This function
    """

    # Reading the dataset
    df = pd.read_csv(r'..\data\processed\motivation_liwc_meta_pos.csv', usecols=['motivation', 'bsa_dummy'])
    df = shuffle(df)

    # Data Preprocessing- tokenization, remove stopwords, lemmatization, stemming - Prepare data for LDA Analysis
    # Remove punctuation
    df['motivation_processed'] = df['motivation'].map(lambda x: re.sub('[,\.!?]', '', x))

    # Removing stopwords
    data = df.motivation_processed.values.tolist()
    data_words = list(sent_to_words(data))
    # remove stop words
    data_words = remove_stopwords(data_words)

    # Create Dictionary
    id2word = corpora.Dictionary(data_words)
    # Create Corpus
    texts = data_words
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]

    # LDA model training *******************************

    # number of topics
    # num_topics = 10
    # Build LDA model
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=num_topics, random_state=0)
    # Print the Keyword in the 10 topics
    with open(r'../results/output/lda/lda_output.txt', 'w') as f:
        with redirect_stdout(f):
            print(lda_model.print_topics())

    # doc_lda = lda_model[corpus]

    # Visualize the topics
    LDAvis_data_filepath = os.path.join(r'../results/output/lda/ldavis_prepared_' + str(num_topics))
    # # this is a bit time consuming - make the if statement True
    # # if you want to execute visualization prep yourself
    if 1 == 1:
        LDAvis_prepared = gensimvis.prepare(lda_model, corpus, id2word)
        with open(LDAvis_data_filepath, 'wb') as f:
            pickle.dump(LDAvis_prepared, f)
    # load the pre-prepared pyLDAvis data from disk
    with open(LDAvis_data_filepath, 'rb') as f:
        LDAvis_prepared = pickle.load(f)
    pyLDAvis.save_html(LDAvis_prepared, r'../results/output/lda/ldavis_prepared_' + str(num_topics) + '.html')
    # LDAvis_prepared



def sent_to_words(sentences):
    for sentence in sentences:
        # deacc=True removes punctuations
        yield (gensim.utils.simple_preprocess(str(sentence), deacc=True))


def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc))
             if word not in stop_words] for doc in texts]


if __name__ == '__main__':
    lda_func()