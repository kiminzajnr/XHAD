"""
Data preparation, threshold selection and metrics shared by every experiment.

The protocol, identical for every detector:
    1. split rows chronologically: oldest 60% train, next 20% val, newest 20% test
    2. fit the feature scaling on NORMAL training rows only
    3. detectors are fitted on normal training rows only
    4. the alarm threshold (and XHAD's alpha) is chosen on validation
    5. the frozen threshold is applied once to test
"""

import numpy as np
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score, roc_curve)
from sklearn.preprocessing import MinMaxScaler

import config
import features


def prepare(dataset):
    """Load features, split chronologically and scale. Returns a dict."""
    X, y, templates = features.load(dataset)
    n = len(y)
    a = int(n * config.TRAIN_FRAC)
    b = int(n * (config.TRAIN_FRAC + config.VAL_FRAC))

    # log1p tames bursty counts (a failing node can repeat one line 20 times)
    X = np.log1p(X)
    normal_train = X[:a][y[:a] == 0]
    scaler = MinMaxScaler().fit(normal_train)

    return dict(
        templates=templates,
        X_train=scaler.transform(normal_train),
        X_train_all=scaler.transform(X[:a]), y_train_all=y[:a],
        X_val=scaler.transform(X[a:b]), y_val=y[a:b],
        X_test=scaler.transform(X[b:]), y_test=y[b:],
    )


def describe(d):
    """Split sizes, anomaly rates and how much of test was already seen in training."""
    seen = {r.tobytes() for r in d["X_train_all"]}
    in_train = np.array([r.tobytes() in seen for r in d["X_test"]])
    anom = d["y_test"] == 1
    return {
        "train_rows": int(len(d["y_train_all"])),
        "train_anomaly_rate": float(d["y_train_all"].mean()),
        "train_normal_rows_used": int(len(d["X_train"])),
        "val_rows": int(len(d["y_val"])),
        "val_anomaly_rate": float(d["y_val"].mean()),
        "test_rows": int(len(d["y_test"])),
        "test_anomaly_rate": float(d["y_test"].mean()),
        "n_features": int(d["X_train"].shape[1]),
        "test_rows_seen_in_train": float(in_train.mean()),
        "test_anomalies_seen_in_train": float(in_train[anom].mean()) if anom.any() else None,
    }


def choose_threshold(scores, y, rule=config.THRESHOLD_RULE):
    """Pick the alarm threshold on validation scores."""
    if rule == "youden":
        fpr, tpr, thr = roc_curve(y, scores)
        return float(thr[np.argmax(tpr - fpr)])
    p, r, thr = precision_recall_curve(y, scores)
    with np.errstate(invalid="ignore"):
        f1 = np.nan_to_num(2 * p[:-1] * r[:-1] / (p[:-1] + r[:-1]))
    return float(thr[np.argmax(f1)])


def metrics(scores, y, threshold):
    pred = scores >= threshold
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc_roc": float(roc_auc_score(y, scores)),
        "auc_pr": float(average_precision_score(y, scores)),
    }
