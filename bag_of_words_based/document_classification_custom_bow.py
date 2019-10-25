import re
from collections import Counter
from pprint import pprint
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from minimal_example.data_reader import get_20newsgroups_data
from minimal_example.minimal_document_classification import encode_targets, benchmark


def identity_dummy_method(x):
    """
    just to fool the scikit-learn vectorizer
    """
    return x


def regex_tokenizer(text, pattern=r"(?u)\b\w\w+\b"):  # pattern stolen from scikit-learn
    return [m.group() for m in re.finditer(pattern, text)]


def text_to_bow(text) -> List[str]:
    return regex_tokenizer(text)


def get_tfidf_vectorizer():
    vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        preprocessor=identity_dummy_method,
        tokenizer=identity_dummy_method,
        max_df=0.75,
        min_df=2,
        max_features=30000,
        stop_words="english",
    )
    return vectorizer


if __name__ == "__main__":
    data_train = get_20newsgroups_data("train")
    data_test = get_20newsgroups_data("test")

    vectorizer = get_tfidf_vectorizer()
    matrix_train = vectorizer.fit_transform(
        [text_to_bow(text) for text, _ in data_train]
    )
    matrix_test = vectorizer.transform([text_to_bow(text) for text, _ in data_test])

    pprint(Counter([label for _, label in data_train]))

    label_encoder, targets_train, targets_test = encode_targets(data_train, data_test)

    clf = SGDClassifier(alpha=0.00001, loss="log", penalty="elasticnet", l1_ratio=0.2)
    benchmark(clf, matrix_train, matrix_test, targets_train, targets_test)
