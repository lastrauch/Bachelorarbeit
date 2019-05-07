# coding=utf-8
import xml.etree.ElementTree as ET
from lxml import etree

# Feature Atributes Article
id_array_article = []
published_at_array_article = []
title_array_article = []


# =======================================================================================================================

# Feature Attributes Publisher
id_array_publisher = []
published_at_array_publisher = []
title_array_publisher = []


# =======================================================================================================================

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


# =======================================================================================================================

def parse_features(content, publisher):
    for event, elem in content:
        for key, value, in elem.items():
            if publisher:
                if key == 'id':
                    id_array_publisher.append(str(value))
                elif key == 'published-at':
                    published_at_array_publisher.append(value)
                elif key == 'title':
                    title_array_publisher.append(value)
            else:
                if key == 'id':
                    id_array_article.append(str(value))
                elif key == 'published-at':
                    published_at_array_article.append(value)
                elif key == 'title':
                    title_array_article.append(value)
            elem.clear()

#------------------------------Comment this part for te Validation Data-------------------------------------------------

    if publisher:
        for _ in range(600000 - len(published_at_array_publisher)):
            published_at_array_publisher.append('/')
    else:
        for _ in range(645 - len(published_at_array_article)):
            published_at_array_article.append('/')

#-----------------------------------------------------------------------------------------------------------------------

