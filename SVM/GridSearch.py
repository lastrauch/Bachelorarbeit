from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn import preprocessing
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np


def to_tfidf(df):
    df.fillna("")
    df['category_id'] = df['Hyperpartisan'].factorize()[0]
    category_id_df = df[['Hyperpartisan', 'category_id']].sort_values('category_id')
    category_to_id = dict(category_id_df.values)
    id_to_category = dict(category_id_df[['category_id', 'Hyperpartisan']].values)
    df.head()

    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='utf-8', ngram_range=(1, 2))
    features = tfidf.fit_transform(df.Content)
    labels = df.category_id
    features.shape
    #features = preprocessing.scale(features,with_mean=False)

    return features, labels


def random_forest(features, target):
    parameter_candidates = {'n_estimators': [100, 500, 1000],
                            'criterion': ['gini', 'entropy'],
                            "max_depth": [3,8,15],
                            "min_samples_leaf": [1,2,4]}
    clf = GridSearchCV(estimator=RandomForestClassifier(), param_grid=parameter_candidates, scoring='accuracy', n_jobs=-1, cv=10)
    print 'done  clf1'
    clf2 = GridSearchCV(estimator=RandomForestClassifier(), param_grid=parameter_candidates, scoring='precision', n_jobs=-1, cv=10)
    print 'done  clf2'
    clf3 = GridSearchCV(estimator=RandomForestClassifier(), param_grid=parameter_candidates, scoring='recall', n_jobs=-1, cv=10)
    print 'done  clf3'
    clf4 = GridSearchCV(estimator=RandomForestClassifier(), param_grid=parameter_candidates, scoring='f1', n_jobs=-1, cv=10)
    print 'done  clf4'
    clf.fit(features, target)
    print 'done  clf1.fit'
    clf2.fit(features, target)
    print 'done  clf2.fit'
    clf3.fit(features, target)
    print 'done  clf3.fit'
    clf4.fit(features, target)
    print 'done  clf4.fit'
    print 'Best Parameters Accuracy:', clf.best_params_
    print 'Best Result Accuracy:', clf.best_score_
    print 'Best Parameters Precision: ', clf2.best_params_
    print 'Best Result Precision:', clf2.best_score_
    print 'Best Parameters Recall: ', clf3.best_params_
    print 'Best Result Recall:', clf3.best_score_
    print 'Best Parameters F1: ', clf4.best_params_
    print 'Best Result F1:', clf4.best_score_

    f = open('/home/lstrauch/Bachelorarbeit/env/Predictions/GridSearch_RandomForest_Article.txt', 'w')
    f.write('Best Parameters in Accuracy:')
    f.write(str(clf.best_params_))
    f.write("\n")
    f.write('Best Result in Accuracy:')
    f.write(str(clf.best_score_))
    f.write("\n")
    f.write('Best Parameters in Precision:')
    f.write(str(clf2.best_params_))
    f.write("\n")
    f.write('Best Result in Precision:')
    f.write(str(clf2.best_score_))
    f.write("\n")
    f.write('Best Parameters in Recall:')
    f.write(str(clf3.best_params_))
    f.write("\n")
    f.write('Best Result in Recall:')
    f.write(str(clf3.best_score_))
    f.write("\n")
    f.write('Best Parameters in F1:')
    f.write(str(clf4.best_params_))
    f.write("\n")
    f.write('Best Result in F1:')
    f.write(str(clf4.best_score_))
    f.close


def svc(features, target):
    parameter_candidates = {'C': [10, 100, 1000],
                            'gamma': [1, 0.001, 0.0001],
                            'kernel': ['linear', 'rbf', 'sigmoid']}
    clf = GridSearchCV(estimator=SVC(), param_grid=parameter_candidates, scoring = 'accuracy', n_jobs=-1, cv=10)
    print 'done  clf1'
    clf2 = GridSearchCV(estimator=SVC(), param_grid=parameter_candidates, scoring = 'precision', n_jobs=-1, cv=10)
    print 'done  clf2'
    clf3 = GridSearchCV(estimator=SVC(), param_grid=parameter_candidates, scoring = 'recall', n_jobs=-1, cv=10)
    print 'done  clf3'
    clf4 = GridSearchCV(estimator=SVC(), param_grid=parameter_candidates, scoring = 'f1', n_jobs=-1, cv=10)
    print 'done  clf4'
    clf.fit(features, target)
    print 'done  clf1.fit'
    clf2.fit(features, target)
    print 'done  clf2.fit'
    clf3.fit(features, target)
    print 'done  clf3.fit'
    clf4.fit(features, target)
    print 'done  clf4.fit'
    print 'Best Parameters Accuracy:', clf.best_params_
    print 'Best Result Accuracy:', clf.best_score_
    print 'Best Parameters Precision: ', clf2.best_params_
    print 'Best Result Precision:', clf2.best_score_
    print 'Best Parameters Recall: ', clf3.best_params_
    print 'Best Result Recall:', clf3.best_score_
    print 'Best Parameters F1: ', clf4.best_params_
    print 'Best Result F1:', clf4.best_score_
    f = open('/home/lstrauch/Bachelorarbeit/env/Predictions/GridSearch_SVC_Publisher.txt', 'w')
    f.write('Best Parameters in Accuracy:')
    f.write(str(clf.best_params_))
    f.write("\n")
    f.write('Best Result in Accuracy:')
    f.write(str(clf.best_score_))
    f.write("\n")
    f.write('Best Parameters in Precision:')
    f.write(str(clf2.best_params_))
    f.write("\n")
    f.write('Best Result in Precision:')
    f.write(str(clf2.best_score_))
    f.write("\n")
    f.write('Best Parameters in Recall:')
    f.write(str(clf3.best_params_))
    f.write("\n")
    f.write('Best Result in Recall:')
    f.write(str(clf3.best_score_))
    f.write("\n")
    f.write('Best Parameters in F1:')
    f.write(str(clf4.best_params_))
    f.write("\n")
    f.write('Best Result in F1:')
    f.write(str(clf4.best_score_))
    f.close


def multinomialnb(features, target):
    parameter_candidates = {'alpha': np.linspace(0.5, 1.5, 6),
                            'fit_prior': [True, False]}
    clf = GridSearchCV(estimator=MultinomialNB(), param_grid=parameter_candidates, scoring='accuracy', n_jobs=-1, cv=10)
    clf2 = GridSearchCV(estimator=MultinomialNB(), param_grid=parameter_candidates, scoring='precision', n_jobs=-1, cv=10)
    clf3 = GridSearchCV(estimator=MultinomialNB(), param_grid=parameter_candidates, scoring='recall', n_jobs=-1, cv=10)
    clf4 = GridSearchCV(estimator=MultinomialNB(), param_grid=parameter_candidates, scoring='f1', n_jobs=-1, cv=10)
    clf.fit(features, target)
    clf2.fit(features, target)
    clf3.fit(features, target)
    clf4.fit(features, target)
    print 'Best Parameters Accuracy:', clf.best_params_
    print 'Best Result Accuracy:', clf.best_score_
    print 'Best Parameters Precision: ', clf2.best_params_
    print 'Best Result Precision:', clf2.best_score_
    print 'Best Parameters Recall: ', clf3.best_params_
    print 'Best Result Recall:', clf3.best_score_
    print 'Best Parameters F1: ', clf4.best_params_
    print 'Best Result F1:', clf4.best_score_

    f = open('/home/lstrauch/Bachelorarbeit/env/Predictions/GridSearch_MultinomialNB_Publisher.txt', 'w')
    f.write('Best Parameters in Accuracy:')
    f.write(str(clf.best_params_))
    f.write("\n")
    f.write('Best Result in Accuracy:')
    f.write(str(clf.best_score_))
    f.write("\n")
    f.write('Best Parameters in Precision:')
    f.write(str(clf2.best_params_))
    f.write("\n")
    f.write('Best Result in Precision:')
    f.write(str(clf2.best_score_))
    f.write("\n")
    f.write('Best Parameters in Recall:')
    f.write(str(clf3.best_params_))
    f.write("\n")
    f.write('Best Result in Recall:')
    f.write(str(clf3.best_score_))
    f.write("\n")
    f.write('Best Parameters in F1:')
    f.write(str(clf4.best_params_))
    f.write("\n")
    f.write('Best Result in F1:')
    f.write(str(clf4.best_score_))
    f.close


def logisticregression(features, target):
    parameter_candidates = {'penalty': ['l1', 'l2'],
                            'C': np.logspace(-4, 4, 20),
                            'solver': ['liblinear']}
    clf = GridSearchCV(estimator=LogisticRegression(), param_grid=parameter_candidates, scoring = 'accuracy', n_jobs=-1, cv=10)
    print 'done  clf1'
    clf2 = GridSearchCV(estimator=LogisticRegression(), param_grid=parameter_candidates, scoring='precision', n_jobs=-1, cv=10)
    print 'done  clf2'
    clf3 = GridSearchCV(estimator=LogisticRegression(), param_grid=parameter_candidates, scoring='recall', n_jobs=-1, cv=10)
    print 'done  clf3'
    clf4 = GridSearchCV(estimator=LogisticRegression(), param_grid=parameter_candidates, scoring='f1', n_jobs=-1, cv=10)
    print 'done  clf4'
    clf.fit(features, target)
    print 'done  clf1.fit'
    clf2.fit(features, target)
    print 'done  clf2.fit'
    clf3.fit(features, target)
    print 'done  clf3.fit'
    clf4.fit(features, target)
    print 'done  clf4.fit'
    print 'Best Parameters Accuracy:', clf.best_params_
    print 'Best Result Accuracy:', clf.best_score_
    print 'Best Parameters Precision: ', clf2.best_params_
    print 'Best Result Precision:', clf2.best_score_
    print 'Best Parameters Recall: ', clf3.best_params_
    print 'Best Result Recall:', clf3.best_score_
    print 'Best Parameters F1: ', clf4.best_params_
    print 'Best Result F1:', clf4.best_score_
    f = open('/home/lstrauch/Bachelorarbeit/env/Predictions/GridSearch_LogisticRegression_Publisher_L1L2.txt', 'w')
    f.write('Best Parameters in Accuracy:')
    f.write(str(clf.best_params_))
    f.write("\n")
    f.write('Best Result in Accuracy:')
    f.write(str(clf.best_score_))
    f.write("\n")
    f.write('Best Parameters in Precision:')
    f.write(str(clf2.best_params_))
    f.write("\n")
    f.write('Best Result in Precision:')
    f.write(str(clf2.best_score_))
    f.write("\n")
    f.write('Best Parameters in Recall:')
    f.write(str(clf3.best_params_))
    f.write("\n")
    f.write('Best Result in Recall:')
    f.write(str(clf3.best_score_))
    f.write("\n")
    f.write('Best Parameters in F1:')
    f.write(str(clf4.best_params_))
    f.write("\n")
    f.write('Best Result in F1:')
    f.write(str(clf4.best_score_))
    f.close


def main():
    #df = pd.read_csv('/home/lstrauch/Bachelorarbeit/env/Data/Preprocessed_ByArticle.csv', encoding='utf-8', engine='python')
    df2 = pd.read_csv('/home/lstrauch/Bachelorarbeit/env/Data/Preprocessed_ByPublisher.csv', encoding='utf-8', engine='python')
    #features, labels = to_tfidf(df)
    features, labels = to_tfidf(df2)

    svc(features, labels)
    #multinomialnb(features, labels)
    #random_forest(features, labels)
    #logisticregression(features, labels)


main()



