from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def to_tfidf(df):
    df.fillna("")
    df['category_id'] = df['Hyperpartisan'].factorize()[0]
    category_id_df = df[['Hyperpartisan', 'category_id']].sort_values('category_id')
    category_to_id = dict(category_id_df.values)
    tfidf= TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='utf-8', ngram_range=(1, 2))
    features = tfidf.fit_transform(df.Content)
    labels = df.category_id
    features.shape

    return features, labels


def random_forest(features, target):
    parameter_candidates = {'n_estimators': [100, 300, 500, 800, 1000], 'criterion': ['gini', 'entropy'],'bootstrap': [True, False], "max_depth": [3,8,15], "min_samples_leaf": [1,2,4], "random_state": [0,1,2]}
    clf = GridSearchCV(estimator=RandomForestClassifier(), param_grid=parameter_candidates, scoring = 'accuracy', n_jobs=-1, cv=10)
    clf.fit(features, target)
    print('Best Parameters:', clf.best_params_)
    print('Best Result:', clf.best_score_)
    f = open('/home/lstrauch/Bachelorarbeit/env/Predictions/GridSearch.txt', 'w')
    f.write('RandomForestClassifier:')
    f.write("\n")
    f.write('Best Parameters:')
    f.write(str(clf.best_params_))
    f.write("\n")
    f.write('Best Result Accuracy:')
    f.write(str(clf.best_score_))
    f.close


def linearsvc(features, target):
    parameter_candidates = {'C':[1,10,100,1000],
                            'gamma':[1,0.1,0.001,0.0001],
                            'kernel': ['rbf', 'sigmoid', 'linear']}
    clf = GridSearchCV(estimator=SVC(), param_grid=parameter_candidates, scoring = 'accuracy', n_jobs=-1, cv=10)
    clf.fit(features, target)
    print('Best Parameters:', clf.best_params_)
    print('Best Result:', clf.best_score_)
    f = open('/home/lstrauch/Bachelorarbeit/env/Predictions/GridSearch_LinearSVC_Publisher.txt', 'w')
    f.write('RandomForestClassifier:')
    f.write("\n")
    f.write('Best Parameters:')
    f.write(str(clf.best_params_))
    f.write("\n")
    f.write('Best Result Accuracy:')
    f.write(str(clf.best_score_))
    f.close

def multinomialnb(features, target):
    parameter_candidates = {'alpha': np.linspace(0.5, 1.5, 6),
                            'fit_prior': [True, False]}
    clf = GridSearchCV(estimator=MultinomialNB(), param_grid=parameter_candidates, scoring = 'accuracy', n_jobs=-1, cv=10)
    clf.fit(features, target)
    print('Best Parameters:', clf.best_params_)
    print('Best Result:', clf.best_score_)
    f = open('/home/lstrauch/Bachelorarbeit/env/Predictions/GridSearch_MultinomialNB_Article.txt', 'w')
    f.write("\n")
    f.write('Best Parameters:')
    f.write(str(clf.best_params_))
    f.write("\n")
    f.write('Best Result Accuracy:')
    f.write(str(clf.best_score_))
    f.close


def logisticregression():
    parameter_candidates = {'classifier__penalty' : ['l1', 'l2'],
                            'classifier__C' : np.logspace(-4, 4, 20),
                            'classifier__solver' : ['liblinear'],
                            'classifier__n_estimators' : list(range(10,101,10)),
                            'classifier__max_features' : list(range(6,32,5))}
    clf = GridSearchCV(estimator=LogisticRegression(), param_grid=parameter_candidates, scoring = 'accuracy', n_jobs=-1, cv=10)
    clf.fit(features, target)
    print('Best Parameters:', clf.best_params_)
    print('Best Result:', clf.best_score_)
    f = open('/home/lstrauch/Bachelorarbeit/env/Predictions/GridSearch_LogisticRegression.txt', 'w')
    f.write("\n")
    f.write('Best Parameters:')
    f.write(str(clf.best_params_))
    f.write("\n")
    f.write('Best Result Accuracy:')
    f.write(str(clf.best_score_))
    f.close


def main():
    df = pd.read_csv('/home/lstrauch/Bachelorarbeit/env/Data/Preprocessed_ByArticle.csv', encoding='utf-8',
                     engine='python')
    df2 = pd.read_csv('/home/lstrauch/Bachelorarbeit/env/Data/Preprocessed_ByPublisher.csv', encoding='utf-8',
                     engine='python')
    features, labels = to_tfidf(df2)

    linearsvc(features, labels)
    #multinomialnb(features, labels)


main()



