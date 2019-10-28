import sys
sys.path.append('.')
from collections import Counter
from pprint import pprint
from time import time
from typing import List, Union

from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model.stochastic_gradient import BaseSGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

from minimal_example.data_reader import get_20newsgroups_data


def encode_targets(data_train, data_test):
    train_labels: List[str] = [label for _, label in data_train]
    test_labels: List[str] = [label for _, label in data_test]
    label_encoder = LabelEncoder()
    label_encoder.fit(train_labels)
    targets_train_encoded = label_encoder.transform(train_labels)
    targets_test_encoded = label_encoder.transform(test_labels)
    return label_encoder, targets_train_encoded, targets_test_encoded


def benchmark(
    clf: Union[BaseSGDClassifier, MultinomialNB],
    matrix_train,
    matrix_test,
    y_train,
    y_test,
):
    print("_" * 80)
    print(str(clf).split("(")[0])

    t0 = time()
    clf.fit(matrix_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred_train = clf.predict(matrix_train)
    pred_test = clf.predict(matrix_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.accuracy_score(y_train, pred_train)
    print("train-f1-micro:   %0.3f" % score)
    score = metrics.accuracy_score(y_test, pred_test)
    print("test-f1-micro:   %0.3f" % score)


if __name__ == "__main__":
    data_train = get_20newsgroups_data("train")
    data_test = get_20newsgroups_data("test")

    vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        max_df=0.75,
        min_df=2,
        max_features=30000,
        stop_words="english",
    )
    matrix_train = vectorizer.fit_transform([text for text, _ in data_train])
    print("n_samples: %d, n_features: %d" % matrix_train.shape)

    matrix_test = vectorizer.transform([text for text, _ in data_test])
    print("n_samples: %d, n_features: %d" % matrix_test.shape)

    pprint(Counter([label for _, label in data_train]))

    label_encoder, targets_train, targets_test = encode_targets(data_train, data_test)

    def benchmark_fun(clf):
        return benchmark(clf, matrix_train, matrix_test, targets_train, targets_test)

    benchmark_fun(
        SGDClassifier(alpha=0.00001, loss="log", penalty="elasticnet", l1_ratio=0.2)
    )

    benchmark_fun(MultinomialNB(alpha=0.01))
