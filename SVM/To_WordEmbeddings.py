from gensim.models import KeyedVectors
import pandas as pd
import numpy as np


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


def to_vector_for_training_and_testing(sentences1, sentences2):
    model = load_fasttext()
    x_train = []
    x_test = []

    for sentence in sentences1:
        x_train.append(sent_vectorizer(sentence, model))

    for sentence in sentences2:
        x_test.append(sent_vectorizer(sentence, model))

    return x_train, x_test


def to_vector_for_content(sentences):
    model = load_fasttext()
    wb_array = []

    for sentence in sentences:
        wb_array.append(sent_vectorizer(sentence, model))

    return wb_array
