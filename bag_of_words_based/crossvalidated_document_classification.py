import multiprocessing
import sys
sys.path.append('.')

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
    split_idx,split = split
    print('%s is working on split %d'%(multiprocessing.current_process(),split_idx))
    train_idx, test_idx = split
    train_data = [data[i] for i in train_idx]
    test_data = [data[i] for i in test_idx]

    matrix_test, matrix_train = vectorize_documents(test_data, train_data)

    clf = SGDClassifier(
        alpha=params["alpha"], loss="log", penalty="elasticnet", l1_ratio=0.2, tol=1e-3
    )
    targets_bin_test, targets_bin_train = binarize_targets(
        label_encoder, test_data, train_data
    )
    clf.fit(matrix_train, np.argmax(targets_bin_train, axis=1))

    target_names = label_encoder.classes_.tolist()

    return {
        "train": predict_and_score(clf, matrix_train, target_names, targets_bin_train),
        "test": predict_and_score(clf, matrix_test, target_names, targets_bin_test),
    }


def predict_and_score(clf, matrix, target_names, targets_bin):
    def binarize_probabilities(proba):
        pred = np.array(
            proba == np.expand_dims(np.max(proba, axis=1), 1), dtype="int64"
        )
        return pred

    proba = clf.predict_proba(matrix)
    pred = binarize_probabilities(proba)
    scores = calc_classification_metrics(
        proba, pred, targets_bin, target_names=target_names
    )
    return scores


def binarize_targets(label_encoder, test_data, train_data):
    targets_bin_train = label_encoder.transform([[label] for _, label in train_data])
    targets_bin_test = label_encoder.transform([[label] for _, label in test_data])
    return targets_bin_test, targets_bin_train


def vectorize_documents(test_data, train_data):
    def get_bows(data):
        return [text_to_bow(text) for text, _ in data]

    vectorizer = get_tfidf_vectorizer()
    matrix_train = vectorizer.fit_transform(get_bows(train_data))
    matrix_test = vectorizer.transform(get_bows(test_data))
    return matrix_test, matrix_train


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

    splitter = StratifiedShuffleSplit(n_splits=3, test_size=0.2, random_state=111)
    splits = list(enumerate(splitter.split(X=range(len(data)), y=[label for _, label in data])))

    m_scores_std_scores = calc_mean_std_scores(task, splits, n_jobs=3)
    m_scores_train = m_scores_std_scores["m_scores"]["train"]

    print(pandas.DataFrame(data=m_scores_train["averages"]).transpose())
    print(pandas.DataFrame(data=m_scores_train["labelwise"]).transpose())

"""
            PR-AUC   ROC-AUC  f1-score  precision    recall  support
micro     0.988747  0.998916  0.972968   0.972968  0.972968   9051.0
macro     0.985706  0.998827  0.975581   0.982132  0.972595   9051.0
weighted       NaN       NaN  0.975240   0.981227  0.972968   9051.0
samples        NaN       NaN  0.972968   0.972968  0.972968   9051.0
                            PR-AUC  PR-AUC-interp  ...    recall  support
alt.atheism               0.984554       0.579470  ...  0.973958    384.0
comp.graphics             0.984895      -1.000000  ...  0.967880    467.0
comp.os.ms-windows.misc   0.978486      -0.477344  ...  0.960536    473.0
comp.sys.ibm.pc.hardware  0.985913      -1.000000  ...  0.973164    472.0
comp.sys.mac.hardware     0.982002       0.516796  ...  0.966163    463.0
comp.windows.x            0.997511       0.586615  ...  0.995781    474.0
misc.forsale              0.993088      -1.000000  ...  0.985043    468.0
rec.autos                 0.979139       0.392293  ...  0.982456    475.0
rec.motorcycles           0.990023       0.389991  ...  0.972803    478.0
rec.sport.baseball        0.984150       0.362391  ...  0.970014    478.0
rec.sport.hockey          0.988237      -0.524478  ...  0.970833    480.0
sci.crypt                 0.988511       0.487916  ...  0.974790    476.0
sci.electronics           0.985977       0.484993  ...  0.967583    473.0
sci.med                   0.987331       0.410654  ...  0.967719    475.0
sci.space                 0.989538      -1.000000  ...  0.972574    474.0
soc.religion.christian    0.992158      -1.000000  ...  0.986778    479.0
talk.politics.guns        0.986582      -0.500628  ...  0.972540    437.0
talk.politics.mideast     0.985746       0.427984  ...  0.967480    451.0
talk.politics.misc        0.979674       0.594802  ...  0.966846    372.0
talk.religion.misc        0.970609       0.656880  ...  0.956954    302.0
"""
