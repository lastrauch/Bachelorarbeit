import pandas as pd
import numpy as np

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.model_selection import cross_val_score

import To_TFIDF as TfIdf
import To_WordEmbeddings as Wb


def predict(features, labels):
    models = [
        RandomForestClassifier(n_estimators=500, criterion='gini', max_depth=15, min_samples_leaf=2),
        # MultinomialNB(alpha=0.7, fit_prior=False),
        LogisticRegression(penalty='l2', C=206.913808111479, solver='liblinear'),
        LogisticRegression(penalty='l2', C=206.913808111479, solver='newton-cg')
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
    df_article = pd.read_csv('/home/lstrauch/Bachelorarbeit/env/Data/Preprocessed_ByArticle.csv', encoding='utf-8',
                     engine='python')
    #df_publisher = pd.read_csv('/home/lstrauch/Bachelorarbeit/env/Data/Preprocessed_ByPublisher.csv', encoding='utf-8',engine='python')

    features = TfIdf.tf_idf_forContent(df_article.Content)
    labels = df_article.Hyperpartisan

    ac, pre, re, f1 = predict(features, labels)
    to_txt(ac, pre, re, f1, '/home/lstrauch/Bachelorarbeit/env/Predictions/TEST_CrossValidation_Article_WordEmbeddings.txt')


main()