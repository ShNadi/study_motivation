import pandas as pd
from imblearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import seaborn as sns
import numpy as np
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from contextlib import redirect_stdout
import pickle


def svm_func(df, filename):

    # keep test_set apart
    df_train, df_test = train_test_split(df, test_size=0.25, stratify=df['bsa_dummy'], shuffle=True,
                                             random_state=0)
    X_train = df_train['motivation']
    y_train = df_train['bsa_dummy']
    X_test = df_test['motivation']
    y_test = df_test['bsa_dummy']

    # df_train.to_csv(r'..\data\processed\train_set.csv', index=False)
    # df_test.to_csv(r'..\data\processed\test_set.csv', index=False)
    print("test_set and train_set are stored in 'data\preprocessed' folder.")

    # # separate train/evaluation set
    # X_train, X_val, y_train, y_val = train_test_split(df['motivation'], df['bsa_dummy'],
    #                                                   stratify=df['bsa_dummy'],
    #                                                   test_size=0.25, random_state=0)

    stopword_list = list(stopwords.words('Dutch'))

    pipe = make_pipeline(TfidfVectorizer(lowercase=True, stop_words=stopword_list), SVC(class_weight='balanced'))
    scores = cross_val_score(pipe, X_train, y_train, cv=5)
    pickle.dump(pipe, open("../data/model/"+ filename+ "_model.sav", 'wb'))

    print('5-fold cross validation scores:', scores)
    print('average of 5-fold cross validation scores:', scores.mean())

    pipe.fit(X_train, y_train)
    predictions = pipe.predict(X_test)
    print("Accuracy for SVM on test_set: %s" % accuracy_score(y_test, predictions))
    cm = confusion_matrix(y_test, predictions)
    print(classification_report(y_test, predictions))

    with open(r'../results/output/classification_reports/'+ filename+'.txt', 'w') as f:
        with redirect_stdout(f):
            print('5-fold cross validation scores:', scores)
            # print('/n')
            print('average of 5-fold cross validation scores:', scores.mean())
            # print('/n')
            print("Accuracy for SVM on test_set: %s" % accuracy_score(y_test, predictions))
            # print('/n')
            print(classification_report(y_test, predictions))

    sns.heatmap(cm / np.sum(cm), annot=True, fmt='.2%', cmap='Blues')
    plt.show()


if __name__=='__main__':
    # df = pd.read_csv('../data/processed/motivation.csv')
    # svm_func(df, 'svm_cross_val')


    # df_2014 = pd.read_csv('../data/raw/motivatoin_2014.csv')
    # df_2014['bsa_dummy'] = df_2014['bsa_dummy'].replace({0: 1, 1: 0})
    # df_2014['reenrolled'] = df_2014['reenrolled'].replace({0: 1, 1: 0})
    # svm_func(df_2014, 'svm_df_2014')

    # df_2015 = pd.read_csv('../data/raw/motivation_2015.csv')
    # df_2015.dropna(subset=['reenrolled', 'bsa_dummy', 'motivation'], inplace=True)
    # df_2015['bsa_dummy'] = df_2015['bsa_dummy'].replace({0: 1, 1: 0})
    # df_2015['reenrolled'] = df_2015['reenrolled'].replace({0: 1, 1: 0})
    # svm_func(df_2015, 'svm_df_2015')

    df_2016 = pd.read_csv('../data/raw/motivation_2016.csv')
    df_2016.dropna(subset=['bsa_dummy', 'motivation'], inplace=True)
    df_2016['bsa_dummy'] = df_2016['bsa_dummy'].replace({0: 1, 1: 0})
    svm_func(df_2016, 'svm_df_2016')