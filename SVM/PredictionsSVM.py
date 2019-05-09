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
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def tf_id(X_train, X_test):
    count_vect = CountVectorizer()
    tfidf_transformer = TfidfTransformer()

    X_train_counts = count_vect.fit_transform(X_train)
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    X_test_counts = count_vect.transform(X_test)
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)

    return X_train_tfidf, X_test_tfidf


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
    df2 = pd.read_csv('/home/lstrauch/Bachelorarbeit/env/Data/Preprocessed_ByArticle.csv', encoding='utf-8', engine='python')
    df.fillna("")
    df2.fillna("")

    # df_Validtion = pd.read_csv('/home/lstrauch/Bachelorarbeit/env/Data/Preprocessed_Validation.csv', encoding='utf-8')
    # df_Validtion.fillna("")

    train_x, test_x = tf_id(df.Content, df2.Content)
    train_y = df['Hyperpartisan']

    print train_x.shape
    print test_x.shape


    #Train the Model
    trained_model1 = classifier(SVC(kernel='rbf', C=10, gamma=1), train_x, train_y)
    trained_model2 = classifier(RandomForestClassifier(bootstrap=False, min_samples_leaf=2, n_estimators=100, random_state=2, criterion='entropy', max_depth=8), train_x, train_y)
    trained_model3 = classifier(MultinomialNB(alpha=1.3, fit_prior=False), train_x, train_y)
    trained_model4 = classifier(LogisticRegression(penalty='l1', C=0.03359818286283781, solver='liblinear'), train_x, train_y)
    trained_model5 = classifier(LogisticRegression(penalty='l2', C=0.0006951927961775605, solver='newton-cg'), train_x, train_y)
    #
    save_trained_model('/home/lstrauch/Bachelorarbeit/env/Data/TrainedModel_SVC_Article_PubArt_GridSearchFeature.sav', trained_model1)
    save_trained_model('/home/lstrauch/Bachelorarbeit/env/Data/TrainedModel_RandomForest_PubArt_GridSearchFeature.sav', trained_model2)
    save_trained_model('/home/lstrauch/Bachelorarbeit/env/Data/TrainedModel_MultinomialNB_PubArt_GridSearchFeature.sav', trained_model3)
    save_trained_model('/home/lstrauch/Bachelorarbeit/env/Data/TrainedModel_LogisticRegression-Liblinear_PubArt_GridSearchFeature.sav', trained_model4)
    save_trained_model('/home/lstrauch/Bachelorarbeit/env/Data/TrainedModel_LogisticRegression-NoL1_PubArt_GridSearchFeature.sav', trained_model5)


    #Load the trained model and make predictions
    predictions1 = load_trained_model('/home/lstrauch/Bachelorarbeit/env/Data/TrainedModel_SVC_PubArt_GridSearchFeature.sav').predict(test_x)
    predictions2 = load_trained_model('/home/lstrauch/Bachelorarbeit/env/Data/TrainedModel_RandomForest_PubArt_GridSearchFeature.sav').predict(test_x)
    predictions3 = load_trained_model('/home/lstrauch/Bachelorarbeit/env/Data/TrainedModel_MultinomialNB_PubArt_GridSearchFeature.sav').predict(test_x)
    predictions4 = load_trained_model('/home/lstrauch/Bachelorarbeit/env/Data/TrainedModel_LogisticRegression-Liblinear_PubArt_GridSearchFeature.sav').predict(test_x)
    predictions5 = load_trained_model('/home/lstrauch/Bachelorarbeit/env/Data/TrainedModel_LogisticRegression-NoL1_PubArt_GridSearchFeature.sav').predict(test_x)


    #Write the predictions into a text file
    writetotxt(df, '/home/lstrauch/Bachelorarbeit/env/Predictions/Predictions_SVC_PubArt_GridSearchFeature.txt', 600000, predictions1)
    writetotxt(df, '/home/lstrauch/Bachelorarbeit/env/Predictions/Predictions_RandomForestClassifier_PubArt_GridSearchFeature.txt', 600000, predictions2)
    writetotxt(df, '/home/lstrauch/Bachelorarbeit/env/Predictions/Predictions_MultinomialNB__PubArt_GridSearchFeature.txt', 645, predictions3)
    writetotxt(df, '/home/lstrauch/Bachelorarbeit/env/Predictions/Predictions_LogisticRegression-Liblinear_PubArt_GridSearchFeature.txt', 645, predictions4)
    writetotxt(df, '/home/lstrauch/Bachelorarbeit/env/Predictions/Predictions_LogisticRegression-NoL1_PubArt_GridSearchFeature.txt', 645, predictions5)


if __name__ == '__main__':
    main()
