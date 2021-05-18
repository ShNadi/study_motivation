from sklearn.utils import shuffle
from nltk.corpus import stopwords
import re
import pandas as pd
from pprint import pprint
from contextlib import redirect_stdout
from matplotlib import pyplot as plt

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
# spacy for lemmatization
import spacy


# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

stop_words = stopwords.words('Dutch')
# Add list of new stopwords to the NLTK stopwords
with open('../results/output/dutch_stopwords/stopwords_new.txt') as f:
    content = f.readlines()
# Remove whitespace characters like `\n` at the end of each line
content = [x.strip() for x in content]

new_stopword_list = content + stop_words

def calculate_best_topic_number():

    # Load and shuffle the dataset
    df = pd.read_csv(r'..\data\processed\motivation_liwc_meta_pos_lang.csv')
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
    nlp = spacy.load(r'C:\ProgramData\Miniconda3\Lib\site-packages\nl_core_news_sm\nl_core_news_sm-2.2.5',
                     disable=['parser', 'ner'])

    # Do lemmatization keeping only noun, adj, vb, adv
    # data_lemmatized = lemmatization(data_words_bigrams, nlp, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

    # Create Dictionary
    # id2word = corpora.Dictionary(data_lemmatized)
    id2word = corpora.Dictionary(data_words_bigrams)

    # Create Corpus
    texts = data_words_bigrams
    # texts = data_lemmatized

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]

    # model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=data_lemmatized,
    #                                                         start=5, limit=45, step=5)
    model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=data_words_bigrams,
                                                            start=10, limit=16, step=1)

    # Show graph
    limit = 16;
    start = 10;
    step = 1;
    x = range(start, limit, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.savefig(
        r'D:\project\5-soppe-motivation\Motivation-Karlijn\motivation\study_motivation\results\output\lda\tst'
        r'\topic_score'
        r'.png')


    with open(
            r'D:\project\5-soppe-motivation\Motivation-Karlijn\motivation\study_motivation\results\output\lda\tst\lda_output_scores.txt',
            'w') as f:
        with redirect_stdout(f):
            for m, cv in zip(x, coherence_values):
                print("Num Topics =", m, " has Coherence Value of", round(cv, 4))

    i = 0
    j = 5
    while i < len(model_list):
        optimal_model = model_list[i]
        model_topics = optimal_model.show_topics(formatted=False)
        with open(
                r'D:\project\5-soppe-motivation\Motivation-Karlijn\motivation\study_motivation\results\output\lda\tst\lda_output_' + str(
                        j) + '.txt', 'w') as f:
            with redirect_stdout(f):
                pprint(optimal_model.print_topics(num_words=10))
                i += 1
                j += 5


def compute_coherence_values(dictionary, corpus, texts,limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                 id2word=dictionary,
                                                 num_topics=num_topics,
                                                 random_state=0,
                                                 update_every=1,
                                                 chunksize=100,
                                                 passes=10,
                                                 per_word_topics=True,
                                                 minimum_probability=0.0)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values




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

if __name__=="__main__":
    calculate_best_topic_number()