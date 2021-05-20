import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model  import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


def logistic_numeric(df):
    # Remove extra columns
    del df['language']
    del df['motivation']
    del df['program']
    del df['studentnr_crypt']

    df = df.fillna(method='ffill')

    # Select categorical features
    categorical_features = ['cohort', 'field', 'prior_educ', 'previously_enrolled', 'multiple_requests', 'gender',
                            'interest', 'ase', 'reenrolled', 'year']

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
    # Apply standard scaler and polynomial features algorithms to numerical features
    numeric_transformer = Pipeline(steps=[('poly', PolynomialFeatures(degree=2)),
                                          ('scaler', StandardScaler())])

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
    lor = LogisticRegression(solver='sag', C=50)

    # Chain all the steps using the Pipeline module
    clf = Pipeline([('preprocessor', preprocessor), ('smt', smt),
                    ('lor', lor)])

    # Split the data into train and test folds and fit the train set using chained pipeline
    y = df['bsa_dummy']
    X = df.drop('bsa_dummy', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=50)
    clf.fit(X_train, y_train)

    # Train score
    print('train score: ', clf.score(X_train, y_train))

    # Test score
    print('test score: ', clf.score(X_test, y_test))

    # Predict results on the test set
    clf_predicted = clf.predict(X_test)

    # Build confusion matrix
    confusion = confusion_matrix(y_test, clf_predicted)
    print(confusion)

    # Print classification report
    print(classification_report(y_test, clf_predicted, target_names=['0', '1']))

    # Extract feature importance
    importance = clf.steps[2][1].coef_
    feature_names = numeric_features + categorical_features

    # Zip feature importance and feature names in the format of dictionary
    coef_dict = {}
    for coef, feat in zip(clf.steps[2][1].coef_[0, :], feature_names):
        coef_dict[feat] = coef

    # Turn dictionary to series
    feature_importance = pd.Series(list(coef_dict.values()), index=coef_dict.keys())
    print(feature_importance)

    # Plot feature importance
    feature_importance.plot.barh(figsize=(10, 20))
    plt.show()





if __name__=='__main__':
    df = pd.read_csv(r'..\data\processed\motivation_liwc_meta_pos_topic_n15.csv')
    logistic_numeric(df)