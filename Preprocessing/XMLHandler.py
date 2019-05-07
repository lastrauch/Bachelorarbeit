# coding=utf-8
import GroundTruth_Parser as groundtruth_parser
import ContentParser as content_parser
import FeatureExtraction as feature_extraction
from lxml import etree
import pandas as Pd

path_article_training = "/home/lstrauch/Bachelorarbeit/env/Data/articles-training-byarticle-20181122.xml"
path_publisher_training = "/home/lstrauch/Bachelorarbeit/env/Data/articles-training-bypublisher-20181122.xml"
path_article_gt = "/home/lstrauch/Bachelorarbeit/env/Data/ground-truth-training-byarticle-20181122.xml"
path_publisher_gt = "/home/lstrauch/Bachelorarbeit/env/Data/ground-truth-training-bypublisher-20181122.xml"
path_validation_gt = '/home/lstrauch/Bachelorarbeit/env/Data/ground-truth-validation-bypublisher-20181122.xml'
path_validation_training = '/home/lstrauch/Bachelorarbeit/env/Data/articles-validation-bypublisher-20181122.xml'

# ======================================================================================================================


def write_to_csv_articles(titlecsv, publisher):
    if publisher:
        xml_gt = etree.iterparse(path_publisher_gt, tag='article')
        xml_training = etree.iterparse(path_publisher_training, tag='article')
        content_training = etree.iterparse(path_publisher_training, tag='article')
    else:
        xml_gt = etree.iterparse(path_article_gt, tag='article')
        xml_training = etree.iterparse(path_article_training, tag='article')
        content_training = etree.iterparse(path_article_training, tag='article')

#------------------Uncomment this to parse the Validation-Datasets------------------------------------------------------

    # xml_gt = etree.iterparse(path_validation_gt, tag='article')
    # xml_training = etree.iterparse(path_validation_training, tag='article')
    # content_training = etree.iterparse(path_validation_training, tag='article')

#-----------------------------------------------------------------------------------------------------------------------

    feature_extraction.parse_features(xml_training, publisher)
    groundtruth_parser.parse_groundtruth(xml_gt, publisher)
    content = content_parser.parse_content(content_training)

    id = feature_extraction.get_id_array(publisher)
    published = feature_extraction.get_published_at_array(publisher)
    title = feature_extraction.get_title_array(publisher)
    bias = groundtruth_parser.get_bias_array(publisher)
    hyperpartisan = groundtruth_parser.get_hyperpartisan_array(publisher)

#------------------Uncomment this to for the Validation-Datasets--------------------------------------------------------

    # published = []
    # for _ in range(150000-len(published)):
    #     published.append('/')

#-----------------------------------------------------------------------------------------------------------------------

    columns = {"ArticleID": id,
               "PublishedAt": published,
               "Title": title,
               "Bias": bias,
               "Content": content,
               "Hyperpartisan": hyperpartisan}

    table_frame = Pd.DataFrame(columns)
    table_frame = table_frame[['ArticleID', 'PublishedAt', 'Title', 'Bias', 'Content', 'Hyperpartisan']]
    print(table_frame)
    table_frame.to_csv(titlecsv, encoding='utf-8', index=False)


def main():
    #Article Dataset
    write_to_csv_articles('/home/lstrauch/Bachelorarbeit/env/Data/ByArticle.csv', False)

    #Publisher Dataset
    # write_to_csv_articles('/home/lstrauch/Bachelorarbeit/env/Data/ByPublisher.csv', True)

    #Validationset
    #write_to_csv_articles('/home/lstrauch/Bachelorarbeit/env/Data/Preprocessed_Validation.csv', True)


if __name__ == '__main__':
    main()

