# coding=utf-8
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import re
from lxml import etree


# =======================================================================================================================

def parse_content(content):
    content_array = []

    for event, elem in content:
        text = (elem.text or '') + ''.join(ET.tostring(e).encode('utf-8') for e in elem)
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
