"""
Does periodic retraining fix drift?

The main experiment trains once on the oldest 60% of rows. Here the test
period (newest 20%) is cut into k equal chunks, and before each chunk every
detector is retrained on earlier rows; the chunk just before it (same length)
picks alpha and the threshold:

    static    - trained once, exactly as in run_experiments.py
    expanding - train on all rows before that validation chunk
    sliding   - train on only the DRIFT_WINDOW_FRAC rows before it

Every k in DRIFT_CHUNKS is reported, so retraining frequency is a measured
sensitivity rather than a value tuned on the test period. Labels of past rows
are assumed known (an operator confirms past incidents), as in the main
experiment's training data. Chunk predictions are pooled over the test period.

Usage:
    python run_drift.py bgl
    python run_drift.py hdfs
Writes results/drift_<dataset>.json
"""

import json
import sys

import numpy as np

import config
from evaluation import choose_threshold, load, split, unseen_share
from models import XHAD, PCADetector

MODELS = ("XHAD", "Autoencoder", "Isolation Forest", "PCA")


def plans(n):
    """{label: [(train, val, test) index slices, one per retraining step]}"""
    a = int(n * config.TRAIN_FRAC)
    b = int(n * (config.TRAIN_FRAC + config.VAL_FRAC))
    w = int(n * config.DRIFT_WINDOW_FRAC)
    out = {"static": [(slice(0, a), slice(a, b), slice(b, n))]}
    for k in config.DRIFT_CHUNKS:
        edges = np.linspace(b, n, k + 1).astype(int)
        size = edges[1] - edges[0]
        steps = [(s - size, s, e) for s, e in zip(edges[:-1], edges[1:])]
        out[f"expanding/{k}"] = [(slice(0, v), slice(v, s), slice(s, e)) for v, s, e in steps]
        out[f"sliding/{k}"] = [(slice(v - w, v), slice(v, s), slice(s, e)) for v, s, e in steps]
    return out


def predict(dataset, seed, d):
    """Binary test predictions per model, each thresholded on validation."""
    xhad = XHAD(dataset, seed).fit(d["X_train"], d["X_train_all"])
    (if_v, ae_v), (if_t, ae_t) = xhad.components(d["X_val"]), xhad.components(d["X_test"])
    xhad.tune(d["X_val"], d["y_val"])
    pca = PCADetector(dataset, seed).fit(d["X_train"])

    preds = {"XHAD": xhad.score(d["X_test"]) >= xhad.threshold}
    for name, val, test in (("Autoencoder", ae_v, ae_t),
                            ("Isolation Forest", if_v, if_t),
                            ("PCA", pca.score(d["X_val"]), pca.score(d["X_test"]))):
        preds[name] = test >= choose_threshold(val, d["y_val"])
    return preds


def prf(pred, y):
    tp = int((pred & (y == 1)).sum())
    precision = tp / pred.sum() if pred.sum() else 0.0
    recall = tp / (y == 1).sum()
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def main(dataset):
    X, y, templates = load(dataset)
    results = {}

    for label, steps in plans(len(y)).items():
        preds = {m: {seed: [] for seed in config.SEEDS} for m in MODELS}
        truth, unseen = [], []
        for train, val, test in steps:
            d = split(X, y, templates, train, val, test)
            truth.append(d["y_test"])
            unseen.append(unseen_share(d) * len(d["y_test"]))
            for seed in config.SEEDS:
                for m, p in predict(dataset, seed, d).items():
                    preds[m][seed].append(p)

        y_true = np.concatenate(truth)
        results[label] = {"normal_test_rows_with_unseen_templates": float(sum(unseen) / len(y_true))}
        for m in MODELS:
            runs = [prf(np.concatenate(preds[m][seed]), y_true) for seed in config.SEEDS]
            results[label][m] = {key: {"mean": float(np.mean([r[key] for r in runs])),
                                       "std": float(np.std([r[key] for r in runs]))}
                                 for key in runs[0]}
        r = results[label]
        print(f"{label:<14} XHAD F1 {r['XHAD']['f1']['mean']:.3f}  AE {r['Autoencoder']['f1']['mean']:.3f}  "
              f"IF {r['Isolation Forest']['f1']['mean']:.3f}  PCA {r['PCA']['f1']['mean']:.3f}  "
              f"| unseen {100 * r['normal_test_rows_with_unseen_templates']:.1f}%", flush=True)

        # save after every setting, so an interrupted run keeps what it finished
        out = config.RESULTS_DIR / f"drift_{dataset}.json"
        out.write_text(json.dumps({"dataset": dataset, "chunks": config.DRIFT_CHUNKS,
                                   "window_frac": config.DRIFT_WINDOW_FRAC,
                                   "seeds": config.SEEDS, "results": results}, indent=2))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    dataset = sys.argv[1].lower() if len(sys.argv) > 1 else ""
    if dataset not in config.DATASETS:
        raise SystemExit("usage: python run_drift.py hdfs|bgl")
    main(dataset)
