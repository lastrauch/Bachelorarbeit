# coding=utf-8
from gensim.models import KeyedVectors
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier


def load_fasttext():
    model = KeyedVectors.load_word2vec_format('/home/lstrauch/Bachelorarbeit/env/SVM/wiki-news-300d-1M.vec/wiki-news-300d-1M.vec')

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
    V = []
    for sentence in sentences:
        V.append(sent_vectorizer(sentence, model))

    V2 = df['Hyperpartisan'].factorize()[0]

    X_train = V[0:516]
    X_test = V[516:645]
    print X_test

    y_train = V2[0:516]
    y_test = V2[516:645]
    print y_test

    return X_train, X_test, y_train, y_test


def classify(X_train, X_test, Y_train, Y_test):
    classifier = MLPClassifier(alpha=0.7, max_iter=400)
    classifier.fit(X_train, Y_train)

    df_results = pd.DataFrame(data=np.zeros(shape=(1, 3)), columns=['classifier', 'train_score', 'test_score'])
    train_score = classifier.score(X_train, Y_train)
    test_score = classifier.score(X_test, Y_test)

    print(classifier.predict_proba(X_test))
    print(classifier.predict(X_test))

    df_results.loc[1, 'classifier'] = "MLP"
    df_results.loc[1, 'train_score'] = train_score
    df_results.loc[1, 'test_score'] = test_score
    df_results.loc[1].to_csv('WodEmbeddings_MLPC.txt', encoding='utf-8')
    print(df_results.loc[1])


def main():
    df = pd.read_csv('/home/lstrauch/Bachelorarbeit/env/Data/Preprocessed_ByPublisher.csv', encoding='utf-8',
                     engine='python')
    df.fillna("")
    model = load_fasttext()
    load_modelVocablulary(model)
    split_dataset(df['Content'], model, df)
    X_train, X_test, y_train, y_test = split_dataset(df['Content'], model, df)
    classify(X_train, X_test, y_train, y_test)


if __name__ == '__main__':
    main()
