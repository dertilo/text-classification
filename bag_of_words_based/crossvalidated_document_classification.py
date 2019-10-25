import pandas

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MultiLabelBinarizer

from bag_of_words_based.document_classification_custom_bow import (
    get_tfidf_vectorizer,
    text_to_bow,
)
from mlutil.crossvalidation import calc_mean_std_scores, ScoreTask
from minimal_example.data_reader import get_20newsgroups_data
from mlutil.classification_metrics import calc_classification_metrics


def score_fun(split, data, label_encoder, params):
    vectorizer = get_tfidf_vectorizer()
    train_idx, test_idx = split
    train_data = [data[i] for i in train_idx]
    test_data = [data[i] for i in test_idx]

    X_train = vectorizer.fit_transform([text_to_bow(text) for text, _ in train_data])
    X_test = vectorizer.transform([text_to_bow(text) for text, _ in test_data])

    clf = SGDClassifier(
        alpha=params["alpha"], loss="log", penalty="elasticnet", l1_ratio=0.2, tol=1e-3
    )
    y_train = label_encoder.transform([[label] for _, label in train_data])
    y_test = label_encoder.transform([[label] for _, label in test_data])
    clf.fit(X_train, np.argmax(y_train, axis=1))

    proba = clf.predict_proba(X_train)
    pred = np.array(proba == np.expand_dims(np.max(proba, axis=1), 1), dtype="int64")
    target_names = label_encoder.classes_.tolist()
    train_scores = calc_classification_metrics(
        proba, pred, y_train, target_names=target_names
    )

    proba = clf.predict_proba(X_test)
    pred = np.array(proba == np.expand_dims(np.max(proba, axis=1), 1), dtype="int64")

    test_scores = calc_classification_metrics(
        proba, pred, y_test, target_names=target_names
    )

    return {"train": train_scores, "test": test_scores}


def kwargs_builder(params={"alpha": 0.0001}):
    data_train = get_20newsgroups_data("train")
    label_encoder = MultiLabelBinarizer()
    label_encoder.fit([[label] for _, label in data_train])

    return {"data": data_train, "params": params, "label_encoder": label_encoder}


if __name__ == "__main__":
    data = kwargs_builder()["data"]

    task = ScoreTask(
        score_fun=score_fun,
        kwargs_builder=kwargs_builder,
        builder_kwargs={"params": {"alpha": 0.00001}},
    )

    splitter = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=111)
    splits = list(splitter.split(X=range(len(data)), y=[label for _, label in data]))

    m_scores_std_scores = calc_mean_std_scores(task, splits, n_jobs=0)
    m_scores_train = m_scores_std_scores["m_scores"]["train"]

    print(pandas.DataFrame(data=m_scores_train["averages"]).transpose())
    print(pandas.DataFrame(data=m_scores_train["labelwise"]).transpose())