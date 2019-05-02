# coding=utf-8
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
from sklearn.externals import joblib


def split_dataset(feature, target, testsize):
    X_train, X_test, y_train, y_test = train_test_split(feature, target, test_size=testsize, random_state=0)

    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X_train)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    return X_train_tfidf, X_test, y_train, y_test, count_vect


def tf_id(X_train):
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X_train)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    return X_train_tfidf

def classifier(model, features, target):
    clf = model
    clf.fit(features, target)

    return clf


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


def save_trained_model(file, model):
    filename = file
    joblib.dump(model, filename)


def load_trained_model(filename):
    loaded_model = joblib.load(filename)

    return loaded_model



def main():
    df = pd.read_csv('/home/lstrauch/Bachelorarbeit/env/Data/Preprocessed_ByPublisher.csv', encoding='utf-8', engine='python')
    df.fillna("")

    df_Validtion = pd.read_csv('/home/lstrauch/Bachelorarbeit/env/Data/Preprocessed_Validation.csv', encoding='utf-8')
    df_Validtion.fillna("")

    train_x, count_vect = tf_id(df['Content'])
    train_y = df['Hyperpartisan']
    tfidf_transformer = TfidfTransformer()

    test_y = tf_id(df_Validtion['Content'])


    #Train the Model
    # trained_model1 = classifier(LinearSVC(), train_x, train_y)
    # trained_model2 = classifier(RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0), train_x, train_y)
    # trained_model3 = classifier(MultinomialNB(), train_x, train_y)
    # trained_model4 = classifier(LogisticRegression(random_state=0, solver='liblinear'), train_x, train_y)
    #
    # save_trained_model('/home/lstrauch/Bachelorarbeit/env/Data/TrainedModel_LinearSVC.sav', trained_model1)
    # save_trained_model('/home/lstrauch/Bachelorarbeit/env/Data/TrainedModel_RandomForest.sav', trained_model2)
    # save_trained_model('/home/lstrauch/Bachelorarbeit/env/Data/TrainedModel_MultinomialNB.sav', trained_model3)
    # save_trained_model('/home/lstrauch/Bachelorarbeit/env/Data/TrainedModel_LogisticRegression.sav', trained_model4)


    #Load the trained model and make predictions
    predictions1 = load_trained_model('/home/lstrauch/Bachelorarbeit/env/Data/TrainedModel_LinearSVC.sav').predict(test_y)
    predictions2 = load_trained_model('/home/lstrauch/Bachelorarbeit/env/Data/TrainedModel_RandomForest.sav').predict(test_y)
    predictions3 = load_trained_model('/home/lstrauch/Bachelorarbeit/env/Data/TrainedModel_MultinomialNB.sav').predict(test_y)
    predictions4 = load_trained_model('/home/lstrauch/Bachelorarbeit/env/Data/TrainedModel_LogisticRegression.sav').predict(test_y)


    #Write the predictions into a text file
    writetotxt(df, '/home/lstrauch/Bachelorarbeit/env/Predictions/LinearSVC_Predictions.txt', 600000, predictions1)
    writetotxt(df, '/home/lstrauch/Bachelorarbeit/env/Predictions/RandomForestClassifier_Predictions.txt', 600000, predictions2)
    writetotxt(df, '/home/lstrauch/Bachelorarbeit/env/Predictions/MultinomialNB_Predictions.txt', 600000, predictions3)
    writetotxt(df, '/home/lstrauch/Bachelorarbeit/env/Predictions/LogisticRegression_Predictions.txt', 600000, predictions4)


if __name__ == '__main__':
    main()
