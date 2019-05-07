# coding=utf-8
from gensim.models import KeyedVectors
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def load_fasttext():
    model = KeyedVectors.load_word2vec_format('/home/lstrauch/Bachelorarbeit/env/SVM/wiki-news-300d-1M.vec/wiki-news-300d-1M.vec')

    print 'done loading'
    return model


def load_modelVocablulary(model):
    words = []
    for word in model.vocab:
        words.append(word)

    return words


def sent_vectorizer(sent, model):
    sent_vec = []
    numw = 0
    for w in sent:
        try:
            if numw == 0:
                sent_vec = model[w]
            else:
                sent_vec = np.add(sent_vec, model[w])
            numw += 1
        except:
            pass

    return np.asarray(sent_vec) / numw


def split_dataset(sentences, model, df):
    v = []
    for sentence in sentences:
        v.append(sent_vectorizer(sentence, model))

    v2 = df.Hyperpartisan.factorize()[0]

    return v, v2


def classify(features, target):
    parameter_candidates = {'n_estimators': [100, 300, 500, 800, 1000],
                            'criterion': ['gini', 'entropy'],
                            'bootstrap': [True, False],
                            "max_depth": [3,8,15],
                            "min_samples_leaf": [1,2,4],
                            "random_state": [0,1,2]}
    clf = GridSearchCV(estimator=RandomForestClassifier(), param_grid=parameter_candidates, scoring= 'accuracy', n_jobs=-1, cv=10)
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


def main():
    df = pd.read_csv('/home/lstrauch/Bachelorarbeit/env/Data/Preprocessed_ByArticle.csv', encoding='utf-8',
                     engine='python')
    df.fillna("")
    model = load_fasttext()
    load_modelVocablulary(model)
    features, target = split_dataset(df.Content, model, df)
    classify(features, target)


if __name__ == '__main__':
    main()
