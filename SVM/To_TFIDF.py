from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


def tf_idf_forTrainingAndTesting(x_train, x_test):
    count_vect = CountVectorizer()
    tfidf_transformer = TfidfTransformer()

    x_train_counts = count_vect.fit_transform(x_train)
    x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)

    x_test_counts = count_vect.transform(x_test)
    x_test_tfidf = tfidf_transformer.transform(x_test_counts)

    return x_train_tfidf, x_test_tfidf


def tf_idf_forContent(content):
    count_vect = CountVectorizer()
    tfidf_transformer = TfidfTransformer()

    content_counts = count_vect.fit_transform(content)
    content_tfidf = tfidf_transformer.fit_transform(content_counts)

    return content_tfidf
