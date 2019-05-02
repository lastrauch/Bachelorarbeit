# coding=utf-8
import pandas as pd
import progressbar
import numpy as np
import matplotlib.pyplot as pyplot
from urlparse import urlparse

hyperpartisan_all = []
hyperpartisan_article = []
hyperpartisan_publisher = []

left = []
right = []
left_center = []
right_center = []
least = []
else_a = []

published_at_article = []
published_at_publisher = []

published_at_article_num = []
published_at_publisher_num = []

right_hyperpart = []
left_hyperpart = []
rightcenter_hyperpart = []
leftcenter_hyperpart = []
least_hyperpart = []
not_right_hyperpart = []
not_left_hyperpart = []
not_rightcenter_hyperpart = []
not_leftcenter_hyperpart = []
not_least_hyperpart = []

path_article = "C:\\Users\\lastrauc\\Documents\\Pythonprojekte\\venv\\Preprocessing\\Articles_by_Article.csv"
path_publisher = "C:\\Users\\lastrauc\\Documents\\Pythonprojekte\\venv\\Preprocessing\\Articles_by_Publisher.csv"
path_all = "C:\\Users\\lastrauc\\Documents\\Pythonprojekte\\venv\\Data\\All_Training_Articles.csv"
path_url = '/home/lstrauch/Bachelorarbeit/env/Data/Url_by_Article.csv'


def initialize():
    for i in range(0, 9):
        published_at_article_num.append(i)

    for i in range(0, 45):
        published_at_publisher_num.append(i)


def readDataHyperpartisan(df):
    c = 0
    with progressbar.ProgressBar(max_value=600645) as bar:
        for item in df['Hyperpartisan']:
            if item:
                if c < 645:
                    hyperpartisan_article.append(item)
                else:
                    hyperpartisan_publisher.append(item)
            bar.update(c)
            c += 1


def readDataBias(df):
    c = 0
    with progressbar.ProgressBar(max_value=600000) as bar:
        for item in df['Bias']:
            if item == 'left':
                left.append(item)
            elif item == 'right':
                right.append(item)
            elif item == 'left-center':
                left_center.append(item)
            elif item == 'right-center':
                right_center.append(item)
            elif item == 'least':
                least.append(item)
            else:
                else_a.append(item)
            bar.update(c)
            c += 1


def readDataPublisher(df):
    c = 0
    with progressbar.ProgressBar(max_value=600645) as bar:
        for item in df['Published_At']:
            if not isNan(item):
                str_item = str(item)
                year = str_item[0:4]
                if c < 645:
                    if year not in published_at_article:
                        published_at_article.append(year)
                    else:
                        index = published_at_article.index(year)
                        published_at_article_num[index] += 1
                else:
                    if year not in published_at_publisher:
                        published_at_publisher.append(year)
                    else:
                        index = published_at_publisher.index(year)
                        published_at_publisher_num[index] += 1
            bar.update(c)
            c += 1
        # bubbleSort(published_at_article)
        # bubbleSort(published_at_publisher)


def mapPubishedYear(df, size, publisher):
    dreizehn = 0
    siebzehn_a = 0
    siebzehn_p = 0
    sechzehn_a = 0
    sechzehn_p = 0
    achzehn_a = 0
    achzehn_p = 0
    ids13 = []
    ids17_a = []
    ids17_p = []
    ids16_a = []
    ids16_p = []
    ids18_a = []
    ids18_p = []

    for i in range(size):
        pub = df.loc[i, 'PublishedAt']
        hyp = df.loc[i, 'Hyperpartisan']
        id1 = df.loc[i, 'ArticleID']
        if hyp:
            str_pub = str(pub)
            year = str_pub[0:4]
            if publisher:
                if year == '2013':
                    dreizehn += 1
                    ids13.append(df.loc[i, 'ArticleID'])
                elif year == '2017':
                    siebzehn_p += 1
                    ids17_p.append(df.loc[i, 'ArticleID'])
                elif year == '2016':
                    sechzehn_p += 1
                    ids16_p.append(df.loc[i, 'ArticleID'])
                elif year == '2018':
                    achzehn_p += 1
                    ids18_p.append(df.loc[i, 'ArticleID'])
            else:
                if year == '2017':
                    siebzehn_a += 1
                    ids17_a.append(df.loc[i, 'ArticleID'])
                elif year == '2016':
                    sechzehn_a += 1
                    ids16_a.append(df.loc[i, 'ArticleID'])
                elif year == '2018':
                    achzehn_a += 1
                    ids18_a.append(df.loc[i, 'ArticleID'])
    if not publisher:
        return ids16_a, ids17_a, ids18_a

    else:
        return ids13, ids16_p, ids17_p, ids18_p



def url(df, ids, size):
    urls = []
    url_index = []
    str_urls = []
    for i in range(size):
        id = df.loc[i, 'ArticleID']
        url = df.loc[i, 'Url']
        for j in range(len(ids)):
            if ids[j] == id:
                if url not in urls:
                    urls.append(url)
                    url_index.append(1)
                else:
                    index = urls.index(url)
                    url_index[index] += 1

    return (urls, url_index)


def urls(publisher, df, df2):
    if not publisher:
        ids16a, ids17a, ids18a = mapPubishedYear(df, 645, False)
        urls, urlindex = url(df2, ids16a, 645)
        url_toTable(
            '/home/lstrauch/Bachelorarbeit/env/DataAnalysis/HTMLTable_Url_Article2016_Groesser.html',
            '/home/lstrauch/Bachelorarbeit/env/DataAnalysis/Csv_Url_Article2016_Groesser.csv', urls,
            urlindex)
        urls, urlindex = url(df2, ids17a, 645)
        url_toTable(
            '/home/lstrauch/Bachelorarbeit/env/DataAnalysis/HTMLTable_Url_Article2017_Groesser.html',
            '/home/lstrauch/Bachelorarbeit/env/DataAnalysis/Csv_Url_Article2017_Groesser.csv', urls,
            urlindex)
        urls, urlindex = url(df2, ids18a, 645)
        url_toTable(
            '/home/lstrauch/Bachelorarbeit/env/DataAnalysis/HTMLTable_Url_Article2018_Groesser.html',
            '/home/lstrauch/Bachelorarbeit/env/DataAnalysis/Csv_Url_Article2018_Groesser.csv', urls,
            urlindex)
    else:
        ids13, ids16p, ids17p, ids18p = mapPubishedYear(df, 600000, True)
        urls, urlindex = url(df2, ids13, 600000)
        url_toTable(
            '/home/lstrauch/Bachelorarbeit/env/DataAnalysis/HTMLTable_Url_Publisher2013_Groesser.html',
            '/home/lstrauch/Bachelorarbeit/env/DataAnalysis/Csv_Url_Publisher2013_Groesser.csv', urls,
            urlindex)
        urls, urlindex = url(df2, ids16p, 600000)
        url_toTable(
            '/home/lstrauch/Bachelorarbeit/env/DataAnalysis/HTMLTable_Url_Publisher2016_Groesser.html',
            '/home/lstrauch/Bachelorarbeit/env/DataAnalysis/Csv_Url_Publisher2016_Groesser.csv', urls,
            urlindex)
        urls, urlindex = url(df2, ids17p, 600000)
        url_toTable(
            '/home/lstrauch/Bachelorarbeit/env/DataAnalysis/HTMLTable_Url_Publisher2017_Groesser.html',
            '/home/lstrauch/Bachelorarbeit/env/DataAnalysis/Csv_Url_Publisher2017_Groesser.csv', urls,
            urlindex)
        urls, urlindex = url(df2, ids18p, 600000)
        url_toTable(
            '/home/lstrauch/Bachelorarbeit/env/DataAnalysis/HTMLTable_Url_Publisher2018_Groesser.html',
            '/home/lstrauch/Bachelorarbeit/env/DataAnalysis/Csv_Url_Publisher2018_Groesser.csv', urls,
            urlindex)


def toTable(published, published_num):
    col = {'Published Year of the Hyperpartisan Article': published,
           'Amount of Hyperpartisan articles this year': published_num}
    tf = pd.DataFrame(col)
    tf.to_html('Table_Published_Hyperpart.html', index=False)


def url_toTable(titleTable, titleCsv, url, urlindex, publisher):
    urls = []
    url_num = []
    for i in range(len(url)):
        if publisher:
            if urlindex[i] > 500:
                urls.append(url[i])
                url_num.append(urlindex[i])
        else:
            if urlindex[i] > 2:
                urls.append(url[i])
                url_num.append(urlindex[i])
    col = {'URL': urls,
           'Amount': url_num}
    tf = pd.DataFrame(col)
    tf.to_html(titleTable, index=False)
    tf.to_csv(titleCsv, encoding='utf-8', index=False)
    # urls = []
    # url_num = []
    # for i in range(len(url)):
    #     urls.append(url[i])
    #     url_num.append(urlindex[i])
    # col = {'URL': urls,
    #        'Amount': url_num}
    # tf = pd.DataFrame(col)
    # tf.to_html(titleTable, index=False)
    # tf.to_csv(titleCsv, encoding='utf-8', index=False)


def map_hyperpartisan_bias(df):
    for i in range(600000):
        if df.loc[i, 'Hyperpartisan']:
            if df.loc[i, 'Bias'] == 'right':
                right_hyperpart.append(df.loc[i, 'Article_ID'])
            elif df.loc[i, 'Bias'] == 'right-center':
                rightcenter_hyperpart.append(df.loc[i, 'Article_ID'])
            elif df.loc[i, 'Bias'] == 'left':
                left_hyperpart.append(df.loc[i, 'Article_ID'])
            elif df.loc[i, 'Bias'] == 'left-center':
                leftcenter_hyperpart.append(df.loc[i, 'Article_ID'])
            elif df.loc[i, 'Bias'] == 'least':
                least_hyperpart.append(df.loc[i, 'Article_ID'])
            # print(df.loc[i, 'Bias'], i, df.loc[i, 'Article_ID'])

    print('Right:', len(right_hyperpart))
    print('Right_Center:', len(rightcenter_hyperpart))
    print('Least:', len(least_hyperpart))
    print('Left_center:', len(leftcenter_hyperpart))
    print('Left:', len(left_hyperpart))


def piediagram(array, size, title):
    labels = 'True', 'False'
    size_hyper = len(array) / size
    size_non_hyper = (size - len(array)) / size
    sizes = [size_hyper, size_non_hyper]
    colors = ['green', 'red']
    pyplot.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
    pyplot.title(title)
    pyplot.axis('equal')
    pyplot.show()


def pieDiagram(array, size, title):
    labels = 'Hyperpartisan', 'Not Hyperpartisan'
    size_hyper = len(array) / size
    size_non_hyper = (size - len(array)) / size
    sizes = [size_hyper, size_non_hyper]
    colors = ['green', 'red']
    pyplot.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
    pyplot.title(title)
    pyplot.axis('equal')
    pyplot.show()


def histogramPublished(array, array_num, size):
    fig, ax = pyplot.subplots()
    labels = []
    for item in array:
        labels.append(item)
    sizes = []
    for item in array_num:
        sizes.append((item / size) * 100)
    y_pos = np.arange(len(labels))
    pyplot.title('Distribution of "Published-At" years')
    pyplot.ylabel('Amount in %')
    pyplot.xlabel('Year')
    pyplot.bar(y_pos, sizes, align='center', alpha=0.6)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation='vertical')
    pyplot.show()


def piediagramBias():
    labels = 'Right', 'Right-Center', 'Least', 'Left-Center', 'Left', 'NaN'
    sizes = [(len(right) / 600000), (len(right_center) / 600000), (len(least) / 600000), (len(left_center) / 600000),
             (len(left) / 600000), (len(else_a) / 600000)]
    colors = ['green', 'blue', 'yellow', 'brown', 'aqua', 'red']
    pyplot.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
    pyplot.title('Distribution of Bias')
    pyplot.axis('equal')
    pyplot.show()


def main():
    df = pd.read_csv('/home/lstrauch/Bachelorarbeit/env/Data/ByArticle.csv')
    df2 = pd.read_csv('/home/lstrauch/Bachelorarbeit/env/Data/Url_by_Article.csv')
    df3 = pd.read_csv('/home/lstrauch/Bachelorarbeit/env/Data/ByPublisher.csv')
    df4 = pd.read_csv('/home/lstrauch/Bachelorarbeit/env/Data/Url_by_Publisher.csv')

    urls(False, df, df2)
    urls(True, df3, df4)

    # initialize()
    # readData()
    # print_result()
    # histogramPublished(published_at_article, published_at_article_num, 645)
    # histogramPublished(published_at_publisher, published_at_publisher_num, 600000)
    # piediagram(hyperpartisan_article, 645, 'Distribution of Articles by Feature:"Hyperpartisan" - labeled by Article')
    # piediagram(hyperpartisan_publisher, 600000, 'Distribution of Articles by Feature:"Hyperpartisan" - labeled by Publisher')
    # piediagramBias()
    # histogramBias()
    # readDataBias(df)
    # map_hyperpartisan_bias(df)
    # pieDiagram(right_hyperpart, len(right), 'Hyperpartisan distribution of right winged Articles')
    # pieDiagram(rightcenter_hyperpart, len(right_center), 'Hyperpartisan distribution of right-center winged Articles')
    # pieDiagram(least_hyperpart, len(least), 'Hyperpartisan distribution of least winged Articles')
    # pieDiagram(leftcenter_hyperpart, len(left_center), 'Hyperpartisan distribution of left-center winged Articles')
    # pieDiagram(left_hyperpart, len(left), 'Hyperpartisan distribution of left winged Articles')


main()
