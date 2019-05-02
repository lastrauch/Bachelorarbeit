from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.feature_selection import chi2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold  # import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle

def to_tfidf(df):
    df.fillna("")
    df['category_id'] = df['Hyperpartisan'].factorize()[0]
    category_id_df = df[['Hyperpartisan', 'category_id']].sort_values('category_id')
    category_to_id = dict(category_id_df.values)
    id_to_category = dict(category_id_df[['category_id', 'Hyperpartisan']].values)
    df.head()

    tfidf= TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='utf-8', ngram_range=(1, 2))
    features = tfidf.fit_transform(df.Content)
    labels = df.category_id
    features.shape

    return features, labels


def predict(features, labels):
    models = [
        RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
        LinearSVC(),
        MultinomialNB(),
        LogisticRegression(random_state=0, solver='liblinear'),
    ]
    CV = 10
    entriesAc = []
    entriesPRE = []
    entriesRE = []
    entriesF1= []
    for model in models:
        model_name = model.__class__.__name__
        accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
        precisions = cross_val_score(model, features, labels, scoring='precision', cv=CV)
        recalls = cross_val_score(model, features, labels, scoring='recall', cv=CV)
        f1s = cross_val_score(model, features, labels, scoring='f1', cv=CV)
        for fold_idx, accuracy, in enumerate(accuracies):
            entriesAc.append((model_name, fold_idx, accuracy))
        for fold_idx, accuracy, in enumerate(precisions):
            entriesPRE.append((model_name, fold_idx, accuracy))
        for fold_idx, accuracy, in enumerate(recalls):
            entriesRE.append((model_name, fold_idx, accuracy))
        for fold_idx, accuracy, in enumerate(f1s):
            entriesF1.append((model_name, fold_idx, accuracy))
    cv_df_ac = pd.DataFrame(entriesAc, columns=['model_name', 'fold_idx', 'accuracy'])
    cv_df_pre = pd.DataFrame(entriesPRE, columns=['model_name', 'fold_idx', 'precision'])
    cv_df_re = pd.DataFrame(entriesRE, columns=['model_name', 'fold_idx', 'recall'])
    cv_df_f1 = pd.DataFrame(entriesF1, columns=['model_name', 'fold_idx', 'f1'])

    return cv_df_ac, cv_df_pre, cv_df_re, cv_df_f1


def grid_search(features, target):
    print '-1'
    parameter_candidates = {'n_estimators': [100, 300, 500, 800, 1000], 'criterion': ['gini', 'entropy'],'bootstrap': [True, False], "max_depth": [3,8,15], "min_samples_leaf": [1,2,4], "random_state": [0,1,2]}
    clf = GridSearchCV(estimator=RandomForestClassifier(), param_grid=parameter_candidates, scoring = 'accuracy', n_jobs=-1, cv=10)
    clf.fit(features, target)
    print('Best Parameters:', clf.best_params_)
    print('Best Result:', clf.best_score_)
    f = open('/home/lstrauch/Bachelorarbeit/env/Predictions/GridSearch.txt', 'w')
    f.write('RandomForestClassifier:')
    f.write("\n")
    f.write('Best Parameters:')
    f.write(clf.best_params_)
    f.write("\n")
    f.write('Best Result Accuracy:')
    f.write(clf.best_score_)
    f.close


def to_txt(ac, pre, re, f1, title):
    print(ac.groupby('model_name').accuracy.mean())
    print(pre.groupby('model_name').precision.mean())
    print(re.groupby('model_name').recall.mean())
    print(f1.groupby('model_name').f1.mean())

    f = open(title, 'w')
    f.write('Accuracy:')
    f.write("\n")
    f.write(str(ac.groupby('model_name').accuracy.mean()))
    f.write("\n")
    f.write("\n")
    f.write('Precision:')
    f.write("\n")
    f.write(str(pre.groupby('model_name').precision.mean()))
    f.write("\n")
    f.write("\n")
    f.write('Recall:')
    f.write("\n")
    f.write(str(re.groupby('model_name').recall.mean()))
    f.write("\n")
    f.write("\n")
    f.write('F1-Score:')
    f.write("\n")
    f.write(str(f1.groupby('model_name').f1.mean()))
    f.close()


def main():
    df = pd.read_csv('/home/lstrauch/Bachelorarbeit/env/Data/Preprocessed_ByArticle.csv', encoding='utf-8',
                     engine='python')
    #df2 = pd.read_csv('/home/lstrauch/Bachelorarbeit/env/Preprocessing/test_publisher_csv.csv', encoding='utf-8',engine='python')
    #df3 = df.append(df2)
    #shuffle(df3)
    features, labels = to_tfidf(df)
    grid_search(features, labels)

    # features, labels = to_tfidf(df3)

    # ac, pre, re, f1 = predict(features, labels)
    # to_txt(ac, pre, re, f1, 'C:\\Users\\lastrauc\\Documents\\Pythonprojekte\\venv\\Predictions\\Cross_Validation.txt')


main()