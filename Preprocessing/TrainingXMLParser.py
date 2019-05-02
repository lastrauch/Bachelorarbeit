# -*- coding: utf-8 -*-

import xml.etree.ElementTree as Et
import GroundTruthParser as Gp
import progressbar
from lxml import etree
import pandas as pd
from contextlib import closing
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# =======================================================================================================================

# Feature-Attributes Article
id_array_article = []
unique_id_array_article = []
published_at_array_article = []
title_array_article = []
bias_array_article = []
content_array_publisher = []
hyperpartisan_array_article = Gp.get_hyperpartisan_article()

# =======================================================================================================================

# Feature Attributes Publisher
unique_id_array_publisher = []
id_array_publisher = []
published_at_array_publisher = []
title_array_publisher = []
bias_array_publisher = Gp.get_bias_array()
content_array_article = []
hyperpartisan_array_publisher = Gp.get_hyperpartisan_publisher()


# =======================================================================================================================

def get_unique_id(publisher):
    if publisher:
        return unique_id_array_publisher
    else:
        return unique_id_array_article


def get_id_array(publisher):
    if publisher:
        return id_array_publisher
    else:
        return id_array_article


def get_published_at_array(publisher):
    if publisher:
        return published_at_array_publisher
    else:
        return published_at_array_article


def get_title_array(publisher):
    if publisher:
        return title_array_publisher
    else:
        return title_array_article


def get_bias_array(publisher):
    if publisher:
        return bias_array_publisher
    else:
        return bias_array_article


def get_content_array(publisher):
    if publisher:
        return content_array_publisher
    else:
        return content_array_article


def get_hyperpartisan_array(publisher):
    if publisher:
        return hyperpartisan_array_publisher
    else:
        return hyperpartisan_array_article


# =======================================================================================================================

def parse_xml(data, id_array, published_at_array, title_array, bias_array, content_array, publisher):
    content = etree.iterparse(data, tag='article', encoding='utf-8')
    stop = list((stopwords.words('english')))
    stop.extend([':', '&',  ',', '[', ']', '{', '}', '<', '>', 'p', ' (', ' )', 'href=', '"', '-', '/p', '/', ' http', '?', '“', '’', '', ' {', ' }', '{'])
    if not publisher:
        for _ in range(645):
            bias_array.append('NULL')

    for event, elem in content:
            text = elem.text
            content = (text or '') + ''.join(Et.tostring(e).encode('utf-8') for e in elem)
            soup = BeautifulSoup(content, features="lxml")
            for script in soup(["script", "style"]):
                script.extract()
            text = soup.get_text()
            if stop in text:
                text.strip(stop)
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            text = word_tokenize(text)
            content_array.append(text)
            for key, value, in elem.items():
                if key == 'id':
                    id_array.append(value)
                elif key == 'published-at':
                    published_at_array.append(value)
                elif key == 'title':
                    title_array.append(value)
            elem.clear()

    if publisher:
        for _ in range(600000 - len(published_at_array)):
            published_at_array.append('/')
    else:
        for _ in range(645 - len(published_at_array)):
            published_at_array.append('/')


def assign_unique_id():
    c = 1
    for _ in range(645):
        s = 'AR-', str(c)
        unique_id_array_article.append(s)
        c += 1
    c = 1
    for _ in range(600000):
        s = 'PU-', str(c)
        unique_id_array_publisher.append(s)
        c += 1


def merge_arrays(array_1, array_2):
    result_array = []
    for elem in array_1:
        result_array.append(elem)
    for elem in array_2:
        result_array.append(elem)

    return result_array


def array_to_string(array):
    return_array = []
    for elem in array:
        return_array.append(str(elem))

    return return_array


def write_to_csv():
    columns = {'Unique_ID': merge_arrays(unique_id_array_article, unique_id_array_publisher),
               'Article_ID': merge_arrays(id_array_article, id_array_publisher),
               'Published_At': merge_arrays(published_at_array_article, published_at_array_publisher),
               'Title': merge_arrays(title_array_article, title_array_publisher),
               'Bias': merge_arrays(bias_array_article, bias_array_publisher),
               'Content': merge_arrays(content_array_article, content_array_publisher),
               'Hyperpartisan': merge_arrays(hyperpartisan_array_article, hyperpartisan_array_publisher)}
    table_frame = pd.DataFrame(columns)
    # table_frame['Unique_ID'].astype(str)
    # table_frame['Article_ID'].astype(str)
    # table_frame['Published_At'].astype(str)
    # table_frame['Title'].astype(str)
    # table_frame['Bias'].astype(str)
    # table_frame['Content'].astype(str)
    # table_frame['Hyperpartisan'].astype(str)
    print(table_frame)
    table_frame.to_csv('All_Training_Articles.csv', encoding='utf-8')


def write_to_csv_articles():
    columns = {'Unique_ID': unique_id_array_article,
               'Article_ID': id_array_article,
               'Published_At': published_at_array_article,
               'Title': title_array_article,
               'Bias': bias_array_article,
               'Content': content_array_article,
               'Hyperpartisan': hyperpartisan_array_article}
    table_frame = pd.DataFrame(columns)
    # table_frame['Unique_ID'].astype(str)
    # table_frame['Article_ID'].astype(str)
    # table_frame['Published_At'].astype(str)
    # table_frame['Title'].astype(str)
    # table_frame['Bias'].astype(str)
    # table_frame['Content'].astype(str)
    # table_frame['Hyperpartisan'].astype(str)
    print(table_frame)
    table_frame.to_csv('Articles_by_Article.csv', encoding='utf-8')


def write_to_csv_publisher():
    columns = {'Unique_ID': unique_id_array_publisher,
               'Article_ID': id_array_publisher,
               'Published_At': published_at_array_publisher,
               'Title': title_array_publisher,
               'Bias': bias_array_publisher,
               'Content': content_array_publisher,
               'Hyperpartisan': hyperpartisan_array_publisher}
    table_frame = pd.DataFrame(columns)
    # table_frame['Unique_ID'].astype(str)
    # table_frame['Article_ID'].astype(str)
    # table_frame['Published_At'].astype(str)
    # table_frame['Title'].astype(str)
    # table_frame['Bias'].astype(str)
    # table_frame['Content'].astype(str)
    # table_frame['Hyperpartisan'].astype(str)
    print(table_frame)
    table_frame.to_csv('Articles_by_Publisher.csv', encoding='utf-8')


def main():
    pfad_article_putty = '/home/lstrauch/Bachelorarbeit/env/Data/articles-training-byarticle-20181122.xml'
    pfad_publisher_putty = '/home/lstrauch/Bachelorarbeit/env/Data/articles-training-bypublisher-20181122.xml'    
    parse_xml(pfad_article_putty, id_array_article, published_at_array_article, title_array_article, bias_array_article, content_array_article, False)
    print('done pars8ing artcile')



main()
