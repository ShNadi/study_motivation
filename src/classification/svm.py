# Last update 25-07-2021
# Remove reenrolled from the list of features
# change the target variable from bsa_dummy to reenrolled
# add df['program'] to the list of features


import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


def svm_numeric(df):
    # In target column (bsa_dummy), 0 stands for bsa obtained and 1 stands for bsa not obtained

    # Remove extra columns
    del df['language']
    del df['motivation']
    # del df['program']
    del df['studentnr_crypt']

    df = df.fillna(method='ffill')

    # Select categorical features
    categorical_features = ['cohort', 'field', 'prior_educ', 'previously_enrolled', 'multiple_requests', 'gender',
                            'interest', 'ase', 'year', 'program']

    # Select numeric features
    numeric_features = ['age', 'HSGPA', 'WC', 'WPS', 'Sixltr',
                        'Dic', 'funct', 'pronoun', 'ppron', 'i',
                        'we', 'you', 'shehe', 'they', 'ipron',
                        'article', 'verb', 'auxverb', 'past', 'present',
                        'future', 'adverb', 'preps', 'conj', 'negate',
                        'quant', 'number', 'swear', 'social', 'family',
                        'friend', 'humans', 'affect', 'posemo', 'negemo',
                        'anx', 'anger', 'sad', 'cogmech', 'insight',
                        'cause', 'discrep', 'tentat', 'certain', 'inhib',
                        'incl', 'excl', 'percept', 'see', 'hear',
                        'feel', 'bio', 'body', 'health', 'sexual',
                        'ingest', 'relativ', 'motion', 'space', 'time',
                        'work', 'achieve', 'leisure', 'home', 'money',
                        'relig', 'death', 'assent', 'nonfl', 'filler',
                        'pronadv', 'shehethey', 'AllPunc', 'Period', 'Comma',
                        'Colon', 'SemiC', 'QMark', 'Exclam', 'Dash',
                        'Quote', 'Apostro', 'Parenth', 'OtherP', 'count_punct',
                        'count_stopwords', 'nr_token', 'nr_adj', 'nr_noun', 'nr_verb',
                        'nr_number', 'topic1', 'topic2', 'topic3', 'topic4',
                        'topic5', 'topic6', 'topic7', 'topic8', 'topic9',
                        'topic10', 'topic11', 'topic12', 'topic13', 'topic14',
                        'topic15']

    # Change object (string) type of features to float
    change_type = ['WPS', 'Sixltr',
                   'Dic', 'funct', 'pronoun', 'ppron', 'i',
                   'we', 'you', 'shehe', 'they', 'ipron',
                   'article', 'verb', 'auxverb', 'past', 'present',
                   'future', 'adverb', 'preps', 'conj', 'negate',
                   'quant', 'number', 'swear', 'social', 'family',
                   'friend', 'humans', 'affect', 'posemo', 'negemo',
                   'anx', 'anger', 'sad', 'cogmech', 'insight',
                   'cause', 'discrep', 'tentat', 'certain', 'inhib',
                   'incl', 'excl', 'percept', 'see', 'hear',
                   'feel', 'bio', 'body', 'health', 'sexual',
                   'ingest', 'relativ', 'motion', 'space', 'time',
                   'work', 'achieve', 'leisure', 'home', 'money',
                   'relig', 'death', 'assent', 'nonfl', 'filler',
                   'pronadv', 'shehethey', 'AllPunc', 'Period', 'Comma',
                   'Colon', 'SemiC', 'QMark', 'Exclam', 'Dash',
                   'Quote', 'Apostro', 'Parenth', 'OtherP']
    df[change_type] = df[change_type].apply(lambda x: x.str.replace(',', '.'))
    df[change_type] = df[change_type].astype(float).fillna(0.0)

    # Scaling features
    # Apply standard scaler to numerical features
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])

    # Apply one hot-encoding for categorical columns
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    # Combine both numerical and categorical column
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

    # Define the SMOTE and Logistic Regression algorithms
    smt = SMOTE(random_state=42)
    lor = SVC(kernel='linear')

    # Chain all the steps using the Pipeline module
    clf = Pipeline([('preprocessor', preprocessor), ('smt', smt),
                    ('lor', lor)])

    # Split the data into train and test folds and fit the train set using chained pipeline
    y = df['bsa_dummy']
    # y = df['reenrolled']
    X = df.drop('bsa_dummy', axis=1)
    # X = df.drop('reenrolled', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=50, shuffle=True, stratify=y)
    clf.fit(X_train, y_train)

    # Train score
    print('train score: ', clf.score(X_train, y_train))
    with open('../../results/output/classification_reports/svm/remove_reenrolled_bsa_is_target/report.txt', 'a+') as f:
        print('train score: ', clf.score(X_train, y_train), file=f)

    # Test score
    print('test score: ', clf.score(X_test, y_test))
    with open('../../results/output/classification_reports/svm/remove_reenrolled_bsa_is_target/report.txt', 'a+') as f:
        print('\n', file=f)
        print('test score: ', clf.score(X_test, y_test), file=f)

    # Predict results on the test set
    clf_predicted = clf.predict(X_test)

    # Build confusion matrix
    confusion = confusion_matrix(y_test, clf_predicted)
    print(confusion)
    with open('../../results/output/classification_reports/svm/remove_reenrolled_bsa_is_target/report.txt', 'a+') as f:
        print('\n', confusion, file=f)

    # Print classification report
    print(classification_report(y_test, clf_predicted, target_names=['0', '1']))
    with open('../../results/output/classification_reports/svm/remove_reenrolled_bsa_is_target/report.txt', 'a+') as f:
        print('\n', classification_report(y_test, clf_predicted, target_names=['0', '1']), file=f)

    # Extract feature importance
    importance = clf.steps[2][1].coef_
    feature_names = numeric_features + categorical_features

    # Zip feature importance and feature names in the format of dictionary
    coef_dict = {}
    for coef, feat in zip(clf.steps[2][1].coef_[0, :], feature_names):
        coef_dict[feat] = coef

    # Sort feature_importance values
    coef_dict = dict(sorted(coef_dict.items(), key=lambda item: item[1]))

    # Turn dictionary to series
    feature_importance = pd.Series(list(coef_dict.values()), index=coef_dict.keys())
    with open('../../results/output/classification_reports/svm/remove_reenrolled_bsa_is_target/feature_importance.txt'
              '', 'w') as f:
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(feature_importance, file=f)

    # Plot feature importance
    feature_importance.plot.barh(figsize=(15, 25))
    plt.show()


if __name__ == '__main__':
    df = pd.read_csv(r'..\..\data\processed\motivation_liwc_meta_pos_topic_n15.csv')
    svm_numeric(df)
