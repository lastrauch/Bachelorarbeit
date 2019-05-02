import fasttext
import pandas as pd
from sklearn.externals import joblib


def skipgram_model(data):
    model = fasttext.cbow(data, 'model')
    #print model.words  # list of words in dictionary
    print model['president']


def load_model():
    model = fasttext.load_model('cbowModel.bin')
    print model['machine']


def main():
    df = pd.read_csv('/home/lstrauch/Bachelorarbeit/env/Data/Preprocessed_ByArticle.csv', encoding='utf-8')
    data = "".join(df['Content'].astype(str))
    skipgram_model(data)
    #load_model()


if __name__ == '__main__':
    main()
