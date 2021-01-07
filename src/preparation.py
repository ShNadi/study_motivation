import pandas as pd
from wordcloud import WordCloud
from nltk.corpus import stopwords
import matplotlib.pyplot as plt


def read_ds():
    # Read 3 data sets for years 2014, 2015 and 2016 separately
    df_2014 = pd.read_excel(r'U:\data_karlijn\transfer_754109_files_6ab59d2d\2014_motivation_study success.xlsx')
    df_2014['year'] = 2014
    df_2015 = pd.read_excel(r'U:\data_karlijn\transfer_754109_files_6ab59d2d\2015_motivation_study success.xlsx')
    df_2015['year'] = 2015
    df_2016 = pd.read_excel(r'U:\data_karlijn\transfer_754109_files_6ab59d2d\2016_motivation_study success.xlsx')
    df_2016['year'] = 2016
    df_2015.rename(columns={"cohort": "COHORT"}, inplace=True)

    # print dataset_2014 shape, column_name
    print('\033[1m \033[94m Dataset_2014:\033[0m \033[0m')
    print('\033[1m \033[1m Shape:\033[0m \033[0m {}\n'.format(df_2014.shape))
    print('\033[1m \033[1m name of columns:\033[0m \033[0m {}\n\n'.format(df_2014.columns.values))

    # print dataset_2015 shape, column_name, number of english motivations
    print('\033[1m \033[94m Dataset_2015:\033[0m \033[0m')
    print('\033[1m \033[1m Shape:\033[0m \033[0m {}\n'.format(df_2015.shape))
    print('\033[1m \033[1m name of columns:\033[0m \033[0m {}\n'.format(df_2015.columns.values))
    print('\033[1m \033[1m number of english motivation\033[0m \033[0m {}\n\n'.format(
        df_2015[df_2015['motivation_EN'].notnull()].shape))

    # print dataset_2016 shape, column_name, number of english motivations
    print('\033[1m \033[94m Dataset_2016:\033[0m \033[0m')
    print('\033[1m \033[1m Shape:\033[0m \033[0m {}\n'.format(df_2016.shape))
    print('\033[1m \033[1m name of columns:\033[0m \033[0m {}\n'.format(df_2016.columns.values))
    print('\033[1m \033[1m number of english motivation:\033[0m \033[0m {}\n\n'.format(
        df_2016[df_2016['motivation_EN'].notnull()].shape))

    return df_2014, df_2015, df_2016

def dropnull(df_2014, df_2015, df_2016):
    df_2014.drop(['Unnamed: 4'], axis=1, inplace=True)
    df_2015.drop(['Unnamed: 5', 'motivation_EN'], axis=1, inplace=True)
    df_2016.drop(['motivation_EN'], axis=1, inplace=True)

    df_2014.dropna(subset=['motivation'], inplace=True)
    df_2015.dropna(subset=['motivation'], inplace=True)
    df_2016.dropna(subset=['motivation'], inplace=True)


def concat(df_2014, df_2015, df_2016):
    frames = [df_2014, df_2015, df_2016]
    df = pd.concat(frames)
    return df

def remove_stopwords(text):
    stopword_list = list(stopwords.words('Dutch'))
    text = ' '.join([word for word in text.split() if word not in stopword_list])
    return text

def word_cloud(df, n):

    df['removed_stopwords'] = df.apply(lambda x: remove_stopwords(x['motivation']), axis=1)

    # Plot wordcloud of 100 most frequent words
    text = ' '.join(df['removed_stopwords'])
    wordcloud = WordCloud(
        width=3000,
        height=2000,
        background_color='black',
        max_words=n
    ).generate(text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.title("Wordcloud of 100 most frequent words")
    plt.show()

