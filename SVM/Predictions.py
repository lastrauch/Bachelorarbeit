# coding=utf-8
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import To_TFIDF as TfIdf
import To_WordEmbeddings as Wb


def classifier(model, features, target):
    clf = model
    clf.fit(features, target)

    return clf


def train_publisher(wordembeddings, train_x, train_y):
    if not wordembeddings:
        # trained_model_svc = classifier(SVC(kernel='rbf', C=10, gamma=1), train_x, train_y)
        trained_model_randomforest_tfidf = classifier(RandomForestClassifier(min_samples_leaf=1, n_estimators=500, criterion='gini', max_depth=15), train_x, train_y)
        trained_model_nb_tfidf = classifier(MultinomialNB(alpha=0.5, fit_prior=True), train_x, train_y)
        trained_model_logistic1_tfidf = classifier(LogisticRegression(penalty='l2', C=4.281332398719396, solver='liblinear'), train_x, train_y)
        # trained_model_logistic2_tfidf = classifier(LogisticRegression(penalty='l2', C=0.0006951927961775605, solver='newton-cg', n_jobs=-1), train_x, train_y)

        # save_trained_model('/home/lstrauch/Bachelorarbeit/env/TrainedModels/TrainedModel_SVC_Article_TrainedOnPublisher.sav', trained_model_svc_tfidf)
        save_trained_model('/home/lstrauch/Bachelorarbeit/env/TrainedModels/TrainedModel_RandomForest_TrainedOnPublisher.sav', trained_model_randomforest_tfidf)
        save_trained_model('/home/lstrauch/Bachelorarbeit/env/TrainedModels/TrainedModel_MultinomialNB_TrainedOnPublisher.sav', trained_model_nb_tfidf)
        save_trained_model('/home/lstrauch/Bachelorarbeit/env/TrainedModels/TrainedModel_LogisticRegression-Liblinear_TrainedOnPublisher.sav', trained_model_logistic1_tfidf)
        # save_trained_model('/home/lstrauch/Bachelorarbeit/env/TrainedModels/TrainedModel_LogisticRegression-NoL1_TrainedOnPublisher.sav', trained_model_logistic2_tfidf)
    else:
        # trained_model_randomforest_wb = classifier(RandomForestClassifier(min_samples_leaf=1, n_estimators=500, criterion='gini', max_depth=15), train_x, train_y)
        trained_model_logistic1_wb = classifier(LogisticRegression(penalty='l2', C=1438.4498882876696, solver='newton-cg'), train_x, train_y)

        # save_trained_model('/home/lstrauch/Bachelorarbeit/env/TrainedModels/TrainedModel_RandomForest_TrainedOnPublisher_WordEmbeddings.sav', trained_model_randomforest_wb)
        save_trained_model('/home/lstrauch/Bachelorarbeit/env/TrainedModels/TrainedModel_LogisticRegression-Liblinear_TrainedOnPublisher_WordEmbeddings.sav', trained_model_logistic1_wb)


def make_predictions(trained_model, test_x, wordembeddings, df, title):
    predictions = load_trained_model(trained_model).predict(test_x)
    if not wordembeddings:
        writetotxt(df, title, 645, predictions)
    else:
        writetotxt(df, title, 150000, predictions)
    print "Done writing predictions"


def save_trained_model(file, model):
    filename = file
    joblib.dump(model, filename)


def load_trained_model(filename):
    loaded_model = joblib.load(filename)

    return loaded_model


def writetotxt(df, title, length, predictions):
    id = []
    for i in range(length):
        num = df.loc[i, 'ArticleID']
        if num < 10:
            str_id = '000000'+str(num)
        elif num < 100:
            str_id = '00000'+str(num)
        elif num < 1000:
            str_id = '0000'+str(num)
        elif num < 10000:
            str_id = '000' + str(num)
        elif num < 100000:
            str_id = '00' + str(num)
        elif num < 1000000:
            str_id = '0' + str(num)
        else:
            str_id = str(num)
        id.append(str_id)

    col = {'article id': id,
           'prediction': predictions}

    tf = pd.DataFrame(col)
    tf.to_csv(title, encoding='utf-8', index=False, sep=' ', header=None)


def publisher_article():
    df_publisher = pd.read_csv('/home/lstrauch/Bachelorarbeit/env/Data/Preprocessed_ByPublisher.csv', encoding='utf-8', engine='python')
    df_publisher.fillna("")
    df_article = pd.read_csv('/home/lstrauch/Bachelorarbeit/env/Data/Preprocessed_ByArticle.csv', encoding='utf-8', engine='python')
    df_article.fillna("")
    train_y = df_publisher.Hyperpartisan

    #Tf-Idf:
    x_train_tfidf, x_test_tfidf = TfIdf.tf_idf_forTrainingAndTesting(df_publisher.Content, df_article.Content)
    train_publisher(False, x_train_tfidf, train_y)
    make_predictions('/homelstrauch/Bachelorarbeit/env/TrainedModels/TrainedModel_RandomForest_TrainedOnPublisher.sav',
                     x_test_tfidf, False, df_article,
                     '/home/lstrauch/Bachelorarbeit/env/Predictions/Predictions_RandomForestClassifier_Publisher-Article.txt')
    make_predictions('/home/lstrauch/Bachelorarbeit/env/TrainedModels/TrainedModel_MultinomialNB_TrainedOnPublisher.sav',
                     x_test_tfidf, False, df_article,
                     '/home/lstrauch/Bachelorarbeit/env/Predictions/Predictions_MultinomialNB_Publisher-Article.txt')
    make_predictions('/home/lstrauch/Bachelorarbeit/env/TrainedModels/TrainedModel_LogisticRegression-Liblinear_TrainedOnPublisher.sav',
                     x_test_tfidf, False, df_article,
                     '/home/lstrauch/Bachelorarbeit/env/Predictions/Predictions_LogisticRegressionLiblinear_Publisher-Article.txt')

    #Word Embeddings:
    x_train_wb, x_test_wb = Wb.to_vector_for_training_and_testing(df_publisher.Content, df_article.Content)
    train_publisher(True, x_train_wb, train_y)
    make_predictions('/home/lstrauch/Bachelorarbeit/env/TrainedModels/TrainedModel_LogisticRegression-Liblinear_TrainedOnPublisher.sav',
                     x_test_tfidf, True, df_article,
                     '/home/lstrauch/Bachelorarbeit/env/Predictions/Predictions_LogisticRegressionLiblinear_Publisher-Article_WordEmbeddings.txt')


def publisher_validation():
    df_publisher = pd.read_csv('/home/lstrauch/Bachelorarbeit/env/Data/Preprocessed_ByPublisher.csv', encoding='utf-8', engine='python')
    df_publisher.fillna("")
    df_validtion = pd.read_csv('/home/lstrauch/Bachelorarbeit/env/Data/Preprocessed_Validation.csv', encoding='utf-8')
    df_validtion.fillna("")
    train_y = df_publisher.Hyperpartisan

    # Tf-Idf:
    x_train_tfidf, x_test_tfidf = TfIdf.tf_idf_forTrainingAndTesting(df_publisher.Content, df_validtion.Content)
    train_publisher(False, x_train_tfidf, train_y)
    make_predictions('/home/lstrauch/Bachelorarbeit/env/TrainedModels/TrainedModel_RandomForest_TrainedOnPublisher.sav',
                     x_test_tfidf, False, df_validtion,
                     '/home/lstrauch/Bachelorarbeit/env/Predictions/Predictions_RandomForestClassifier_Publisher-Validation.txt')
    make_predictions('/home/lstrauch/Bachelorarbeit/env/TrainedModels/TrainedModel_MultinomialNB_TrainedOnPublisher.sav',
                     x_test_tfidf, False, df_validtion,
                     '/home/lstrauch/Bachelorarbeit/env/Predictions/Predictions_MultinomialNB_Publisher-Validation.txt')
    make_predictions('/home/lstrauch/Bachelorarbeit/env/TrainedModels/TrainedModel_LogisticRegression-Liblinear_TrainedOnPublisher.sav',
                     x_test_tfidf, False, df_validtion,
                     '/home/lstrauch/Bachelorarbeit/env/Predictions/Predictions_LogisticRegressionLiblinear_Publisher-Validation.txt')

    # Word Embeddings:
    x_train_wb, x_test_wb = Wb.to_vector_for_training_and_testing(df_publisher.Content, df_validtion.Content)
    train_publisher(True, x_train_publisher_wb, train_y)
    make_predictions('/home/lstrauch/Bachelorarbeit/env/TrainedModels/TrainedModel_LogisticRegression-Liblinear_TrainedOnPublisher.sav',
                     x_test_wb, True, df_validtion,
                     '/home/lstrauch/Bachelorarbeit/env/Predictions/Predictions_LogisticRegressionLiblinear_Publisher-Validation_WordEmbeddings.txt')


def main():
    #Train on Publisher - predict on Article
    publisher_article()
    #Train on Publisher - predict on Validation
    #publisher_validation()


if __name__ == '__main__':
    main()
