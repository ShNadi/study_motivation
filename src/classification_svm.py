from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score



def cut_testset(df):
    df = pd.read_csv(r'..\data\processed\clean_set.csv')
    df_train, df_test = train_test_split(df, test_size=0.25, stratify=df['bsa_dummy'], shuffle=True, random_state=0)

    df_train.to_csv(r'..\data\processed\train_set2.csv', index=False)
    df_test.to_csv(r'..\data\processed\test_set2.csv', index=False)
    print("testset and trainset are stoerd in 'data\preprocessed' folder.")


def train_svm(X_train, y_train):

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(X_train.values.astype('U'))
    clf = SVC(kernel='rbf', class_weight='balanced', C=1.0).fit(X, y_train)
    # save the model to disk
    pickle.dump(clf, open(r'..\data\model\svm_model.sav', 'wb'))

    X_v = vectorizer.transform(X_train.values.astype('U'))
    predict_train = clf.predict(X_v)

    print("train Accuracy for SVM: %s" % accuracy_score(y_train, predict_train))

    print("svm_model.sav is saved in 'data\model' folder")
    return vectorizer

def predict_svm(vectorizer, X_val, y_val):
    # load the model from disk
    clf = pickle.load(open(r'..\data\model\svm_model.sav', 'rb'))

    X_v = vectorizer.transform(X_val.values.astype('U'))
    predictions = clf.predict(X_v)

    print("Final Accuracy for SVM: %s" % accuracy_score(y_val, predictions))
    cm = confusion_matrix(y_val, predictions)
    print(classification_report(y_val, predictions))


