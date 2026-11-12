"""
Anomaly detectors. Each follows the same contract:

    fit(X_normal)  - train on normal training rows only
    score(X)       - anomaly score per row, higher = more anomalous

XHAD combines Isolation Forest and the autoencoder:
    S(x) = alpha * S_IF(x) + (1 - alpha) * S_AE(x)
where both component scores are min-max scaled using their score range on
normal training rows, so one fixed threshold means the same thing on val
and test.

Each component is trained in the regime it suits:
    Isolation Forest - ALL training rows, no labels. It can only split on
                       features that vary in its training data, so trained on
                       normal rows alone it never sees error-only templates.
    Autoencoder      - normal training rows, so it learns to rebuild normal
                       behaviour and fails on anything else.
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDOneClassSVM
from sklearn.pipeline import make_pipeline
import keras

import config
from evaluation import choose_threshold, metrics


class IForest:
    def __init__(self, seed):
        self.model = IsolationForest(**config.IF_PARAMS, random_state=seed, n_jobs=-1)

    def fit(self, X):
        self.model.fit(X)
        return self

    def score(self, X):
        return -self.model.score_samples(X)


class Autoencoder:
    """Dense autoencoder; anomaly score = mean squared reconstruction error."""

    def __init__(self, dataset, seed):
        self.params = config.AE_PARAMS[dataset]
        self.seed = seed

    def fit(self, X):
        keras.utils.set_random_seed(self.seed)
        inputs = keras.Input(shape=(X.shape[1],))
        h = inputs
        for units in self.params["hidden"]:
            h = keras.layers.Dense(units, activation="relu")(h)
        h = keras.layers.Dense(self.params["bottleneck"], activation="relu")(h)
        for units in reversed(self.params["hidden"]):
            h = keras.layers.Dense(units, activation="relu")(h)
        outputs = keras.layers.Dense(X.shape[1], activation="sigmoid")(h)

        t = config.AE_TRAIN
        self.model = keras.Model(inputs, outputs)
        self.model.compile(optimizer=keras.optimizers.Adam(t["lr"]), loss="mse")
        self.model.fit(
            X, X, epochs=t["epochs"], batch_size=t["batch_size"],
            validation_split=t["val_split"], verbose=0,
            callbacks=[keras.callbacks.EarlyStopping(
                patience=t["patience"], restore_best_weights=True)],
        )
        return self

    def score(self, X):
        recon = self.model.predict(X, batch_size=8192, verbose=0)
        return np.mean((X - recon) ** 2, axis=1)


class PCADetector:
    """Reconstruction error from a low-rank linear subspace (Xu et al., 2009)."""

    def __init__(self, dataset, seed):
        self.model = PCA(n_components=config.PCA_COMPONENTS[dataset], random_state=seed)

    def fit(self, X):
        self.model.fit(X)
        return self

    def score(self, X):
        recon = self.model.inverse_transform(self.model.transform(X))
        return np.mean((X - recon) ** 2, axis=1)


class OCSVM:
    """One-Class SVM via a Nystroem kernel approximation (exact OCSVM is O(n^2))."""

    def __init__(self, seed):
        self.seed = seed

    def fit(self, X):
        p = config.OCSVM_PARAMS
        self.model = make_pipeline(
            Nystroem(gamma=1.0 / X.shape[1], n_components=min(p["n_components"], len(X)),
                     random_state=self.seed),
            SGDOneClassSVM(nu=p["nu"], random_state=self.seed),
        ).fit(X)
        return self

    def score(self, X):
        return -self.model.decision_function(X)


class XHAD:
    """Weighted Isolation Forest + autoencoder ensemble."""

    def __init__(self, dataset, seed):
        self.iforest = IForest(seed)
        self.ae = Autoencoder(dataset, seed)
        self.alpha = 0.5
        self.threshold = None

    def fit(self, X_normal, X_unlabelled):
        self.iforest.fit(X_unlabelled)
        self.ae.fit(X_normal)
        # scale each component by its own range on normal training rows
        s_if, s_ae = self.iforest.score(X_normal), self.ae.score(X_normal)
        self._if_range = (s_if.min(), s_if.max())
        self._ae_range = (s_ae.min(), s_ae.max())
        return self

    def components(self, X):
        """Scaled (S_IF, S_AE). Can exceed 1 for rows unlike anything in training."""
        (lo_if, hi_if), (lo_ae, hi_ae) = self._if_range, self._ae_range
        s_if = (self.iforest.score(X) - lo_if) / (hi_if - lo_if)
        s_ae = (self.ae.score(X) - lo_ae) / (hi_ae - lo_ae)
        return s_if, s_ae

    def score(self, X):
        s_if, s_ae = self.components(X)
        return self.alpha * s_if + (1 - self.alpha) * s_ae

    def tune(self, X_val, y_val):
        """Pick alpha by validation F1, then fix the threshold. Returns val F1 per alpha."""
        s_if, s_ae = self.components(X_val)
        val_f1 = {}
        for a in config.ALPHAS:
            s = a * s_if + (1 - a) * s_ae
            val_f1[a] = metrics(s, y_val, choose_threshold(s, y_val))["f1"]
        self.alpha = max(val_f1, key=val_f1.get)
        self.threshold = choose_threshold(self.alpha * s_if + (1 - self.alpha) * s_ae, y_val)
        return val_f1
