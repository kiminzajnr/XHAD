"""
SHAP explanations for XHAD and the measurements of their quality.

KernelSHAP explains the ensemble score S(x) itself (not one component),
for anomalies XHAD correctly detects on the test split. Measured:

    local accuracy  - base value + sum of SHAP values reproduces S(x)
    agreement       - do the templates SHAP ranks highest match those that
                      separate anomalous from normal rows in the training
                      labels? (Spearman rho, top-k overlap vs. chance)
    consistency     - do similar anomalies get similar explanations?
                      (each anomaly vs. its nearest other anomaly)
    stability       - does the explanation survive a +/-1% input perturbation?

Also writes the expert-review sheet: the explained anomalies with their
top templates, ready for SREs to rate on a 1-5 scale.

Usage:
    python explain.py hdfs
    python explain.py bgl
Writes results/explain_<dataset>.json, results/expert_review_<dataset>.csv,
       figures/shap_global_<dataset>.png, figures/shap_example_<dataset>.png
"""

import csv
import json
import logging
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
from scipy.stats import spearmanr

import config
from evaluation import prepare
from models import XHAD

logging.getLogger("shap").setLevel(logging.ERROR)
K = config.SHAP_TOP_K


def top_k(row):
    return set(np.argsort(-np.abs(row))[:K])


def jaccard(a, b):
    return len(a & b) / len(a | b)


def pair_scores(sv_a, sv_b):
    """Spearman rho and top-k Jaccard between two explanations."""
    return spearmanr(sv_a, sv_b)[0], jaccard(top_k(sv_a), top_k(sv_b))


def mean_std(values):
    return {"mean": float(np.nanmean(values)), "std": float(np.nanstd(values))}


def short(template, n=60):
    return template if len(template) <= n else template[:n - 3] + "..."


def plot_bars(values, labels, title, path):
    order = np.argsort(np.abs(values))[-10:]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    # numeric positions: two templates can shorten to the same label
    ax.barh(range(len(order)), values[order],
            color=["#c0392b" if values[i] > 0 else "#2471a3" for i in order])
    ax.set_yticks(range(len(order)), [short(labels[i]) for i in order])
    ax.set_xlabel("SHAP value (contribution to XHAD anomaly score)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main(dataset):
    d = prepare(dataset)
    templates = d["templates"]
    seed = config.SEEDS[0]

    xhad = XHAD(dataset, seed).fit(d["X_train"], d["X_train_all"])
    xhad.tune(d["X_val"], d["y_val"])
    scores = xhad.score(d["X_test"])
    hits = np.where((scores >= xhad.threshold) & (d["y_test"] == 1))[0]

    # explain distinct patterns only; identical rows would inflate consistency
    _, first = np.unique(d["X_test"][hits], axis=0, return_index=True)
    hits = hits[np.sort(first)]
    rng = np.random.default_rng(seed)
    if len(hits) > config.SHAP_N_EXPLAIN:
        hits = np.sort(rng.choice(hits, config.SHAP_N_EXPLAIN, replace=False))
    X = d["X_test"][hits]
    print(f"{dataset.upper()}: alpha={xhad.alpha}, explaining {len(hits)} distinct detected anomalies")

    background = shap.kmeans(d["X_train"], config.SHAP_BACKGROUND)
    explainer = shap.KernelExplainer(xhad.score, background)
    explain = lambda rows: np.asarray(explainer.shap_values(
        rows, nsamples=config.SHAP_NSAMPLES, silent=True))

    sv = explain(X)
    base = float(np.ravel(explainer.expected_value)[0])
    local_err = np.abs(base + sv.sum(axis=1) - xhad.score(X))

    # agreement with what the labels say separates the classes (training data only)
    y_tr = d["y_train_all"]
    signal = np.abs(d["X_train_all"][y_tr == 1].mean(0) - d["X_train_all"][y_tr == 0].mean(0))
    importance = np.abs(sv).mean(axis=0)
    overlap = len(top_k(importance) & top_k(signal))
    chance = K * K / X.shape[1]

    # consistency: each anomaly vs. its nearest other anomaly
    consistency = []
    if len(X) > 1:
        dist = np.linalg.norm(X[:, None] - X[None], axis=2)
        np.fill_diagonal(dist, np.inf)
        consistency = [pair_scores(sv[i], sv[j]) for i, j in enumerate(dist.argmin(1))]

    # stability: re-explain after scaling every feature by up to +/-1%
    X_pert = X * (1 + rng.uniform(-config.SHAP_PERTURB, config.SHAP_PERTURB, X.shape))
    sv_pert = explain(X_pert)
    stability = [pair_scores(a, b) for a, b in zip(sv, sv_pert)]

    result = {
        "dataset": dataset,
        "seed": seed,
        "alpha": xhad.alpha,
        "n_explained": int(len(hits)),
        "local_accuracy_max_error": float(local_err.max()),
        "agreement": {
            "spearman": float(spearmanr(importance, signal)[0]),
            f"top{K}_overlap": overlap,
            f"top{K}_overlap_by_chance": chance,
        },
        "consistency": {
            "spearman": mean_std([c[0] for c in consistency]),
            f"top{K}_jaccard": mean_std([c[1] for c in consistency]),
        },
        "stability": {
            "spearman": mean_std([s[0] for s in stability]),
            f"top{K}_jaccard": mean_std([s[1] for s in stability]),
        },
        "global_top_templates": [
            {"template": str(templates[i]), "mean_abs_shap": float(importance[i])}
            for i in np.argsort(-importance)[:10]
        ],
    }

    config.RESULTS_DIR.mkdir(exist_ok=True)
    out = config.RESULTS_DIR / f"explain_{dataset}.json"
    out.write_text(json.dumps(result, indent=2))

    sheet = config.RESULTS_DIR / f"expert_review_{dataset}.csv"
    with open(sheet, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "test_row", "xhad_score", "threshold"]
                   + [f"reason_{r + 1}" for r in range(K)]
                   + ["makes_sense_1to5", "actionable_1to5", "comment"])
        for n, (row, s) in enumerate(zip(hits, sv), 1):
            reasons = [f"{templates[i]} ({s[i]:+.3f})" for i in np.argsort(-s)[:K]]
            w.writerow([n, int(row), f"{scores[row]:.3f}", f"{xhad.threshold:.3f}"] + reasons + ["", "", ""])

    config.FIGURES_DIR.mkdir(exist_ok=True)
    plot_bars(importance, templates, f"{dataset.upper()}: mean |SHAP| over detected anomalies",
              config.FIGURES_DIR / f"shap_global_{dataset}.png")
    worst = int(np.argmax(xhad.score(X)))
    plot_bars(sv[worst], templates, f"{dataset.upper()}: why test row {hits[worst]} was flagged",
              config.FIGURES_DIR / f"shap_example_{dataset}.png")

    print(f"\nlocal accuracy max error : {result['local_accuracy_max_error']:.2e}")
    print(f"agreement   Spearman     : {result['agreement']['spearman']:+.3f}   "
          f"top-{K} overlap {overlap}/{K} (chance {chance:.2f})")
    for name in ("consistency", "stability"):
        r = result[name]
        print(f"{name:<12}Spearman     : {r['spearman']['mean']:+.3f} +/- {r['spearman']['std']:.3f}   "
              f"top-{K} Jaccard {r[f'top{K}_jaccard']['mean']:.3f}")
    print(f"\nExample explanation (test row {hits[worst]}):")
    for i in np.argsort(-sv[worst])[:K]:
        print(f"  {sv[worst][i]:+.3f}  {short(templates[i], 90)}")
    print(f"\nSaved {out}, {sheet} and figures in {config.FIGURES_DIR}")


if __name__ == "__main__":
    dataset = sys.argv[1].lower() if len(sys.argv) > 1 else ""
    if dataset not in config.DATASETS:
        raise SystemExit("usage: python explain.py hdfs|bgl")
    main(dataset)
