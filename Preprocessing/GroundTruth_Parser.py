from lxml import etree
from urlparse import urlparse

# Attributes for acrticles-training-by-Arcticle

hyperpartisan_array_article = []
id_array_article = []
hyperpartisan_array_article_id = []
bias_array_article = []
url_article = []

# =======================================================================================================================

# Attributes for arcticles-training-by-Publisher

hyperpartisan_array_publisher = []
hyperpartisan_array_publisher_id = []
id_array_publisher = []
bias_array_publisher = []
url_publisher = []

# =======================================================================================================================

# Getter-Methods
def get_bias_array(publisher):
    if publisher:
        return bias_array_publisher
    else:
        return bias_array_article


def get_hyperpartisan_array(publisher):
    if publisher:
        return hyperpartisan_array_publisher
    else:
        return hyperpartisan_array_article


# =======================================================================================================================

def parse_groundtruth(content, publisher):
    for event, elem in content:
        for key, value, in elem.items():
            if publisher:
                if key == 'id':
                    id_array_publisher.append(value)
                elif key == 'hyperpartisan':
                    hyperpartisan_array_publisher.append(value)
                elif key == 'bias':
                    bias_array_publisher.append(value)
                elif key == 'url':
                    url_publisher.append(value)
            else:
                if key == 'id':
                    id_array_article.append(value)
                elif key == 'hyperpartisan':
                    hyperpartisan_array_article.append(value)
                elif key == 'url':
                    url_article.append(value)
        elem.clear()

    if not publisher:
        for _ in range(645):
            bias_array_article.append('/')

    url_netloc = []
    for i in range(645):
        u = urlparse(url_article[i])
        url_netloc.append(u.netloc)

    print url_netloc



