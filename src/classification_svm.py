from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from IPython.display import display


def cut_testset(df):
    df = pd.read_csv(r'..\data\processed\clean_set.csv')
    df_train, df_test = train_test_split(df, test_size=0.25, stratify=df['bsa_dummy'], shuffle=True, random_state=0)

    df_train.to_csv(r'..\data\processed\train_set2.csv', index=False)
    df_test.to_csv(r'..\data\processed\test_set2.csv', index=False)
    print("testset and trainset are stoerd in 'data\preprocessed' folder.")


def train_svm(X_train, y_train, model_name='svm_model'):

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(X_train.values.astype('U'))
    clf = SVC(kernel='rbf', class_weight='balanced', C=1.0).fit(X, y_train)
    # clf = SVC(kernel='rbf', class_weight='balanced', C=100, gamma=0.001).fit(X, y_train)

    # save the model to disk
    pickle.dump(clf, open("../data/model/" + model_name + ".sav", 'wb'))

    X_v = vectorizer.transform(X_train.values.astype('U'))
    predict_train = clf.predict(X_v)

    print("train Accuracy for SVM: %s" % accuracy_score(y_train, predict_train))

    print("svm_model.sav is saved in 'data\model' folder")
    return vectorizer

def predict_svm(vectorizer, X_val, y_val, model_name='svm_model'):
    # load the model from disk
    clf = pickle.load(open('../data/model/'+ model_name + '.sav', 'rb'))

    X_v = vectorizer.transform(X_val.values.astype('U'))
    predictions = clf.predict(X_v)

    print("Final Accuracy for SVM: %s" % accuracy_score(y_val, predictions))
    cm = confusion_matrix(y_val, predictions)
    print(classification_report(y_val, predictions))


def svm_cross(X_train, y_train):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(X_train.values.astype('U'))
    clf = SVC(kernel='rbf', class_weight='balanced', C=1.0)
    scoring = {'acc': 'accuracy',
               'prec_macro': 'precision_macro',
               'rec_macro': 'recall_macro',
               'f1_macro': 'f1_macro'}
    res = cross_validate(clf, X, y_train, cv=5, return_train_score=True, scoring=scoring, n_jobs=-1)
    res_df = pd.DataFrame(res)
    display(res_df)
    print("Mean times and scores:\n", res_df.mean())


def grid_searchcv(df):
    print(__doc__)

    # Split the dataset in two equal parts
    X_train, X_test, y_train, y_test = train_test_split(df['motivation'], df['bsa_dummy'], stratify=df['bsa_dummy'],
                                                        test_size=0.5, random_state=0)

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(X_train.values.astype('U'))
    X_test = vectorizer.transform(X_test.values.astype('U'))

    # Set the parameters by cross-validation
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                         'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

    scores = ['precision', 'recall']

    for score in scores:
        with open('gridsearch.txt', 'a') as f:
            print("# Tuning hyper-parameters for %s" % score, file=f)
            print(file=f)

            clf = GridSearchCV(
                SVC(class_weight='balanced'), tuned_parameters, scoring='%s_macro' % score
            )
            clf.fit(X, y_train)

            print("Best parameters set found on development set:", file=f)
            print(file=f)
            print(clf.best_params_, file=f)
            print(file=f)
            print("Grid scores on development set:", file=f)
            print(file=f)
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                print("%0.3f (+/-%0.03f) for %r"
                      % (mean, std * 2, params), file=f)
            print(file=f)

            print("Detailed classification report:", file=f)
            print(file=f)
            print("The model is trained on the full development set.", file=f)
            print("The scores are computed on the full evaluation set.", file=f)
            print(file=f)
            y_true, y_pred = y_test, clf.predict(X_test)
            print(classification_report(y_true, y_pred), file=f)
            print(file=f)
    print('outputs are written in results folder')