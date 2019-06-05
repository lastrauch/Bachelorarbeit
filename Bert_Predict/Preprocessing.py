# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from lxml import etree
from bs4 import BeautifulSoup
import xml.etree.ElementTree as Et
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import re
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")


def parse_groundtruth(content):
    hyperpartisan = []
    for event, elem in content:
        for key, value, in elem.items():
            if key == 'hyperpartisan':
                if value == 'true':
                    hyperpartisan.append('True')
                else:
                    hyperpartisan.append('False')
        elem.clear()
    return hyperpartisan


def getId(content):
    id_array = []
    for event, elem in content:
        for key, value, in elem.items():
            if key == 'id':
                id_array.append(str(value))
            elem.clear()
    return id_array


def parse_content(content):
    content_array = []

    for event, elem in content:
        text = (elem.text or '') + ''.join(Et.tostring(e).encode('utf-8') for e in elem)
        soup = BeautifulSoup(text, features="lxml")
        text = cleanContent(soup)
        text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        text = ' '.join(text.split()).strip()
        content_array.append(text)
        elem.clear()

    return content_array


def cleanContent(soup):
    soup.prettify()
    for script in soup(["script", "style"]):
        script.extract()
    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '\n'.join(chunk for chunk in chunks if chunk)
    return text



def preprocess(input1, input2, tmp_output):
    xml_id = etree.iterparse(input1, tag='article')
    xml_content = etree.iterparse(input1, tag='article')
    xml_gt = etree.iterparse(input2, tag='article')
    id = getId(xml_id)
    content = parse_content(xml_content)
    hyperpartisan = parse_groundtruth(xml_gt)

    print len(id)
    print len(content)
    columns = {"ArticleID": id,
               "Content": content,
               "Hyperpartisan": hyperpartisan}

    df = pd.DataFrame(columns)
    s_content = pd.Series(df.Content)
    df.Content = s_content.str.lower()

    df.fillna("")
    stop = stopwords.words('english')
    stop.append(['href=', '"', '-', '/p', '/', ' http', '?', '“', '’', '', ' {', ' }', '{', 's', 't'])

    df.Content = df.Content.fillna("")
    df.Content = df.Content.map(lambda x: re.sub(r"[^a-zA-Z0-9]+", ' ', x))
    df.Content = df.Content.apply(lambda x: ' '.join([item for item in x.split() if item not in stop]))
    df.Content = df.Content.apply(lambda x:  ''.join([stemmer.stem(y) for y in x]))

    df.to_csv(tmp_output, encoding='utf-8', index=False, sep="\t".encode('utf-8'))


