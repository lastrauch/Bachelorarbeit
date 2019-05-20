from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import To_TFIDF as TfIdf
import To_WordEmbeddings as Wb


def random_forest(features, target):
    parameter_candidates = {'n_estimators': [100, 500, 1000],
                            'criterion': ['gini', 'entropy'],
                            "max_depth": [3, 8, 15],
                            "min_samples_leaf": [1, 2, 4]}
    clf = GridSearchCV(estimator=RandomForestClassifier(), param_grid=parameter_candidates, scoring='accuracy',
                       n_jobs=1, cv=10)
    clf.fit(features, target)
    print 'done  clf1.fit'

    print 'Best Parameters Accuracy:', clf.best_params_
    print 'Best Result Accuracy:', clf.best_score_

    f = open('/home/lstrauch/Bachelorarbeit/env/Predictions/TEST_GridSearch_RandomForest_Publisher_WordEmbeddings.txt', 'w')
    f.write('Best Parameters in Accuracy:')
    f.write(str(clf.best_params_))
    f.write("\n")
    f.write('Best Result in Accuracy:')
    f.write(str(clf.best_score_))
    f.close


def svc(features, target):
    parameter_candidates = {'C': [10, 100, 1000],
                            'gamma': [1, 0.001, 0.0001],
                            'kernel': ['linear', 'rbf', 'sigmoid']}
    clf = GridSearchCV(estimator=SVC(), param_grid=parameter_candidates, scoring='accuracy', n_jobs=1, cv=10)
    clf.fit(features, target)
    print 'done  clf1.fit'

    print 'Best Parameters Accuracy:', clf.best_params_
    print 'Best Result Accuracy:', clf.best_score_

    f = open('/home/lstrauch/Bachelorarbeit/env/Predictions/TEST_GridSearch_SVC_Publisher.txt', 'w')
    f.write('Best Parameters in Accuracy:')
    f.write(str(clf.best_params_))
    f.write("\n")
    f.write('Best Result in Accuracy:')
    f.write(str(clf.best_score_))

    f.close


def multinomialnb(features, target):
    parameter_candidates = {'alpha': np.linspace(0.5, 1.5, 6),
                            'fit_prior': [True, False]}
    clf = GridSearchCV(estimator=MultinomialNB(), param_grid=parameter_candidates, scoring='accuracy', n_jobs=1, cv=10)
    clf.fit(features, target)

    print 'Best Parameters Accuracy:', clf.best_params_
    print 'Best Result Accuracy:', clf.best_score_

    f = open('/home/lstrauch/Bachelorarbeit/env/Predictions/TEST_GridSearch_MultinomialNB.txt', 'w')
    f.write('Best Parameters in Accuracy:')
    f.write(str(clf.best_params_))
    f.write("\n")
    f.write('Best Result in Accuracy:')
    f.write(str(clf.best_score_))
    f.close


def logisticregression1(features, target):
    parameter_candidates1 = {'penalty': ['l1', 'l2'],
                             'C': np.logspace(-4, 4, 20),
                             'solver': ['liblinear']}
    clf1 = GridSearchCV(estimator=LogisticRegression(), param_grid=parameter_candidates1, scoring = 'accuracy', n_jobs=1, cv=10)

    clf1.fit(features, target)
    print 'done  clf1.fit'

    print 'Best Parameters Accuracy:', clf1.best_params_
    print 'Best Result Accuracy:', clf1.best_score_

    f = open('/home/lstrauch/Bachelorarbeit/env/Predictions/TEST_GridSearch_LogisticRegression_Publisher_L1L2_WordEmbeddings.txt', 'w')
    f.write('Best Parameters in Accuracy:')
    f.write(str(clf1.best_params_))
    f.write("\n")
    f.write('Best Result in Accuracy:')
    f.write(str(clf1.best_score_))
    f.close


def logisticregression2(features, target):
    parameter_candidates2 = {'penalty': ['l2'],
                             'C': np.logspace(-4, 4, 20),
                             'solver': ['liblinear', 'newton-cg', 'lbfgs'],
                             'max_iter': [1000, 10000]}
    clf2 = GridSearchCV(estimator=LogisticRegression(), param_grid=parameter_candidates2, scoring='accuracy', n_jobs=1,
                        cv=10)

    clf2.fit(features, target)
    print 'done  clf2.fit'

    print 'Best Parameters Accuracy:', clf2.best_params_
    print 'Best Result Accuracy:', clf2.best_score_

    f = open('/home/lstrauch/Bachelorarbeit/env/Predictions/TEST_GridSearch_LogisticRegression_Publisher_L2_WordEmbeddings.txt', 'w')
    f.write('Best Parameters in Accuracy:')
    f.write(str(clf2.best_params_))
    f.write("\n")
    f.write('Best Result in Accuracy:')
    f.write(str(clf2.best_score_))
    f.close


def main():
    #------------------------------------------------Articles By Article-----------------------------------------------
    # df_article = pd.read_csv('/home/lstrauch/Bachelorarbeit/env/Data/Preprocessed_ByArticle.csv', encoding='utf-8',
    #                          engine='python')
    # labels_article = df_article.Hyperpartisan
    #------------------------------------TF-IDF:--------------------------------------------
    # print 'Article_Tf-Idf'
    # features_tfidf_article = TfIdf.tf_idf_forContent(df_article.Content)
    # svc(features_tfidf_article, labels_article)
    # multinomialnb(features_tfidf_article, labels_article)
    # random_forest(features_tfidf_article, labels_article)
    # logisticregression1(features_tfidf_article, labels_article)
    # logisticregression2(features_tfidf_article, labels_article)
    # # -----------------------------------Word Embeddings:--------------------------------------
    # print 'Article_WordEmbeddings'
    # features_wb_article = Wb.to_vector_for_content(df_article.Content)
    # random_forest(features_wb_article, labels_article)
    # print 'Done Random Forest'
    # logisticregression1(features_wb_article, labels_article)
    # print 'Done LogistIC Regression 1'
    # logisticregression2(features_wb_article, labels_article)
    # print 'Done LogistIC Regression 2'
    # # -----------------------------------------------Articles by Publisher----------------------------------------------
    df_publisher = pd.read_csv('/home/lstrauch/Bachelorarbeit/env/Data/Preprocessed_ByPublisher.csv', encoding='utf-8',
                               engine='python')
    labels_publisher = df_publisher.Hyperpartisan
    # # ----------------------------------TF-IDF:------------------------------
    # print 'Publisher_Tf-Idf'
    # features_tfidf_publisher = TfIdf.tf_idf_forContent(df_publisher.Content)
    # multinomialnb(features_tfidf_publisher, labels_publisher)
    # print 'Done NB'
    # random_forest(features_tfidf_publisher, labels_publisher)
    # print 'Done Random Forest'
    # logisticregression1(features_tfidf_publisher, labels_publisher)
    # print 'Done Logistic Regression 1'
    # logisticregression2(features_tfidf_publisher, labels_publisher)
    # print 'Done Logistic Regression 2'
    # svc(features_tfidf_publisher, labels_publisher)
    # print 'Done Svc'
    # ----------------------------------Word Embeddings:----------------------------------------------------
    print 'Publisher_WordEmbeddings'
    features_wb_publisher = Wb.to_vector_for_content(df_publisher.Content)
    random_forest(features_wb_publisher, labels_publisher)
    print 'Done Random Forest'
    logisticregression1(features_wb_publisher, labels_publisher)
    print 'Done logistic Regression 1'
    logisticregression2(features_wb_publisher, labels_publisher)
    print 'Done logistic Regression 2'


main()
