import pandas as pd
import spacy
from spacy.lang.nl.examples import sentences
from collections import defaultdict
import time


def calculate_the_word_types(data):
    ''''
    Does part-of-speak tagging and calculates the number of word types for each document.
    '''
    nouns = defaultdict(lambda: 0)
    verbs = defaultdict(lambda: 0)
    adjectives = defaultdict(lambda: 0)
    numbers = defaultdict(lambda: 0)

    nlp = spacy.load(r"C:\ProgramData\Miniconda3\Lib\site-packages\nl_core_news_sm\nl_core_news_sm-2.2.5")

    # count all tokens, but not the punctuations
    for i, row in data.iterrows():
        doc = nlp(row["motivation"])
        data.at[i, "nr_token"] = len(list(map(lambda x: x.text,
                                              filter(lambda x: x.pos_ != 'PUNCT', doc))))

        # count only the adjectives
        for a in map(lambda x: x.lemma_, filter(lambda x: x.pos_ == 'ADJ', doc)):
            adjectives[a] += 1
        data.at[i, "nr_adj"] = len(list(map(lambda x: x.text,
                                            filter(lambda x: x.pos_ == 'ADJ', doc))))

        # count only the nouns
        for n in map(lambda x: x.lemma_, filter(lambda x: x.pos_ == 'NOUN', doc)):
            nouns[n] += 1
        data.at[i, "nr_noun"] = len(list(map(lambda x: x.text,
                                             filter(lambda x: x.pos_ == 'NOUN', doc))))

        # count only the verbs
        for v in map(lambda x: x.lemma_, filter(lambda x: (x.pos_ == 'AUX') | (x.pos_ == 'VERB'), doc)):
            verbs[v] += 1
        data.at[i, "nr_verb"] = len(list(map(lambda x: x.text,
                                             filter(lambda x: (x.pos_ == 'AUX') | (x.pos_ == 'VERB'), doc))))

        # count only the numbers
        for n in map(lambda x: x.lemma_, filter(lambda x: x.pos_ == 'NUM', doc)):
            numbers[n] += 1
        data.at[i, "nr_number"] = len(list(map(lambda x: x.text,
                                               filter(lambda x: x.pos_ == 'NUM', doc))))

    # return data
    df.to_csv(r'..\data\processed\motivation_liwc_meta_pos.csv', index=False)


if __name__ == "__main__":

    start = time.time()
    df = pd.read_csv(r'..\data\processed\motivation_liwc_meta.csv')
    calculate_the_word_types(df)
    end = time.time()
    print(end - start)
