from sklearn.utils import shuffle
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import re
import numpy as np
import pandas as pd
from contextlib import redirect_stdout
import pickle
import os

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import matplotlib.pyplot as plt

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

# Filter warning messages
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

# Download list of Dutch stopwords
stop_words = stopwords.words('Dutch')


def lda_document_topic_distribution():
    """ Document Topic Distribution

     Using LDA, this function distinguish 10 topic category from the text
     and calculates document-topic matrix.


    """

    # Load and shuffle the dataset
    df = pd.read_csv(r'..\data\processed\motivation_liwc_meta_pos.csv')
    df = shuffle(df, random_state=100)

    # Convert to list
    data = df.motivation.values.tolist()

    # Remove new line characters
    data = [re.sub(r'\s+', ' ', sent) for sent in data]

    # Remove distracting single quotes
    data = [re.sub(r"\'", "", sent) for sent in data]

    # Tokenize words and Clean-up text
    data_words = list(sent_to_words(data))

    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)  # higher threshold fewer phrases.
    # trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    # trigram_mod = gensim.models.phrases.Phraser(trigram)

    # Remove Stop Words
    data_words_nostops = remove_stopwords(data_words)

    # Form Bigrams
    data_words_bigrams = make_bigrams(data_words_nostops, bigram_mod)

    # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
    # python3 -m spacy download en
    nlp = spacy.load(r'C:\ProgramData\Miniconda3\Lib\site-packages\nl_core_news_sm\nl_core_news_sm-2.2.5', disable=['parser', 'ner'])

    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(data_words_bigrams, nlp, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

    # Create Dictionary
    id2word = corpora.Dictionary(data_lemmatized)

    # Create Corpus
    texts = data_lemmatized

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]

    # Build LDA model- number of topics=10
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                 id2word=id2word,
                                                 num_topics=10,
                                                 random_state=0,
                                                 update_every=1,
                                                 chunksize=100,
                                                 passes=10,
                                                 alpha='auto',
                                                 per_word_topics=True,
                                                 minimum_probability=0.0)


    # Save theLDA model to disk
    filename = r'..\data\model\LDA_bestmodel_n10\LDA_bestmodel_n10.sav'
    pickle.dump(lda_model, open(filename, 'wb'))

    lda_report(lda_model, corpus, data_lemmatized, id2word)
    get_document_topics = [lda_model.get_document_topics(item) for item in corpus]
    v = get_document_topics
    a = np.array(v)
    df_topic = pd.DataFrame(a[:, :, 1],
                       columns=['topic1', 'topic2', 'topic3', 'topic4', 'topic5', 'topic6', 'topic7', 'topic8',
                                'topic9', 'topic10'])
    # Join main dataframe and topic dataframe
    df_final = df.join(df_topic)

    # Save dataset to disk
    df_final.to_csv(r'..\data\processed\motivation_liwc_meta_pos_topic.csv', index=False)
    print(df_final)




def sent_to_words(sentences):
    """ Tokenize words and Clean-up text

This function tokenize each sentence into a list of words,
removing punctuations and unnecessary characters altogether.

    :param sentences: list
    :type sentences: list
    """
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations



def remove_stopwords(texts):
    """ Remove Stopwords

    :param texts: input text
    :type texts: list
    :return: the text without stopwords
    :rtype: list
    """
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts, bigram_mod):
    """
This function makes bigrams form the text data.
    :param texts:
    :type texts: list
    :param bigram_mod:
    :type bigram_mod: list
    :return:
    :rtype: data_words_bigrams
    """
    return [bigram_mod[doc] for doc in texts]

# def make_trigrams(texts, bigram_mod, trigram_mod):
#     return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, nlp, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation

    for Dutch lemmatization download 'nl_core_news_sm' using following command:
     python -m spacy download nl_core_news_sm
    """
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

def lda_report(lda_model, corpus, data_lemmatized, id2word):
    """
    This function reports Perplexity, Coherence Score and also visualize the topics.
    :param lda_model:
    :type lda_model:
    :param corpus:
    :type corpus:
    :param data_lemmatized:
    :type data_lemmatized:
    :param id2word:
    :type id2word:
    """
    with open(r'../results/output/lda/Best_lda_model/lda_bm_n10.txt', 'w') as f:
        with redirect_stdout(f):
            print(lda_model.print_topics())

    # Compute Perplexity
    with open(r'../results/output/lda/Best_lda_model/lda_output_bm_n10.txt', 'w') as f:
        with redirect_stdout(f):
            print('\nPerplexity: ',
                  lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
    # coherence_model_lda = CoherenceModel(model=lda_model, texts=data_words_bigrams, dictionary=id2word,
    #                                      coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()

    with open(r'../results/output/lda/Best_lda_model/lda_output_bm_n10.txt', 'a') as f:
        with redirect_stdout(f):
            print('\nCoherence Score: ', coherence_lda)

    # Visualize the topics
    # vis = gensimvis.prepare(lda_model, corpus, id2word)
    LDAvis_data_filepath = os.path.join(r'../results/output/lda/Best_lda_model/ldavis_prepared_n10')

    if 1 == 1:
        LDAvis_prepared = gensimvis.prepare(lda_model, corpus, id2word)
        with open(LDAvis_data_filepath, 'wb') as f:
            pickle.dump(LDAvis_prepared, f)
    # load the pre-prepared pyLDAvis data from disk
    with open(LDAvis_data_filepath, 'rb') as f:
        LDAvis_prepared = pickle.load(f)
    pyLDAvis.save_html(LDAvis_prepared, r'../results/output/lda/Best_lda_model/ldavis_n10.html')



if __name__ == "__main__":
    lda_document_topic_distribution()
