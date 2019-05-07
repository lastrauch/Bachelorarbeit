# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from string import punctuation
import gc
import nltk
from io import open
import re
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")



publisher = []
article = []
tokenize_publisherT = []
tokenize_articleT = []
tokenize_publisherP = []
tokenize_articleP = []

lower_line = []

# =======================================================================================================================


def str_lower(df):
    s_title = pd.Series(df['Title'])
    s_content = pd.Series(df['Content'])
    df['Title'] = s_title.str.lower()
    df['Content'] = s_content.str.lower()


def remove_stop(df):
    df.fillna("")
    stop = stopwords.words('english')
    stop.append([':', '&',  ',', '[', ']', '{', '}', '<', '>', 'p', ' (', ' )', 'href=', '"', '-', '/p', '/', ' http', '?', '“', '’', '', ' {', ' }', '{'])

    df['Title'] = df['Title'].fillna("")
    df['Content'] = df['Content'].fillna("")
    df['Content'] = df['Content'].apply(lambda x: ' '.join([item for item in x.split() if item not in stop]))
    df['Title'] = df['Title'].apply(lambda x: ' '.join([item for item in x.split() if item not in stop]))
    df['Content'] = df['Content'].map(lambda x: re.sub(r"[^a-zA-Z0-9]+", ' ', x))
    df['Title'] = df['Title'].map(lambda x: re.sub(r"[^a-zA-Z0-9]+", ' ', x))
    df["Title"] = df["Title"].apply(nltk.word_tokenize)
    df["Content"] = df["Content"].apply(nltk.word_tokenize)
    df['Content'] = df['Content'].apply(lambda x: [stemmer.stem(y) for y in x])


def to_csv(df, output):
    df.to_csv(output, encoding='utf-8', index=False)


def main():
    df1 = pd.read_csv('/home/lstrauch/Bachelorarbeit/env/Data/ByArticle.csv', encoding='utf-8')
    str_lower(df1)
    remove_stop(df1)
    to_csv(df1, '/home/lstrauch/Bachelorarbeit/env/Data/Preprocessed_ByArticle.csv')
    print 'Done: "byArticle"'

    df = pd.read_csv('/home/lstrauch/Bachelorarbeit/env/Data/ByPublisher.csv', encoding='utf-8')
    str_lower(df)
    remove_stop(df)
    to_csv(df, '/home/lstrauch/Bachelorarbeit/env/Data/Preprocessed_ByPublisher.csv')
    print 'done "ByPublisher'


if __name__ == '__main__':
    main()
