"""
Explanations for XHAD and the measurements of their quality, SHAP vs. LIME.

Both methods explain the ensemble score S(x) itself (not one component),
for the same anomalies XHAD correctly detects on the test split, for every
seed. Measured per method:

    local accuracy  - SHAP: base value + sum of SHAP values vs. S(x)
                      LIME: R^2 of its local linear fit
    agreement       - do the templates ranked highest match those that
                      separate anomalous from normal rows in the training
                      labels? (Spearman rho, top-k overlap vs. chance)
    consistency     - do similar anomalies get similar explanations?
                      (each anomaly vs. its nearest other anomaly)
    stability       - does the explanation survive a +/-1% input perturbation?
    seconds         - time to explain one anomaly

The first seed also writes the expert-review sheet (the explained anomalies
with their top SHAP templates, for SREs to rate 1-5) and the figures.

Usage:
    python explain.py hdfs
    python explain.py bgl
    python explain.py hdfs --figures-only   # redraw figures from saved SHAP values
Writes results/explain_<dataset>.json, results/expert_review_<dataset>.csv,
       results/shap_<dataset>.npz,
       figures/shap_global_<dataset>.png, figures/shap_example_<dataset>.png
"""

import csv
import json
import logging
import re
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
from lime.lime_tabular import LimeTabularExplainer
from scipy.stats import spearmanr

import config
from evaluation import prepare
from models import XHAD

logging.getLogger("shap").setLevel(logging.ERROR)
K = config.SHAP_TOP_K
LIME_SAMPLES = 5000


def top_k(row):
    return set(np.argsort(-np.abs(row))[:K])


def jaccard(a, b):
    return len(a & b) / len(a | b)


def pair_scores(e_a, e_b):
    """Spearman rho and top-k Jaccard between two explanations."""
    return spearmanr(e_a, e_b)[0], jaccard(top_k(e_a), top_k(e_b))


def short(template, n=60):
    return template if len(template) <= n else template[:n - 3] + "..."


def figure_labels(templates, idx, width=55):
    """
    Template labels for a figure: the Java class is dropped
    ('INFO dfs.DataNode$PacketResponder: Received ...' -> 'INFO Received ...'),
    then labels are cut at the shortest width that keeps them all distinct.
    """
    full = [re.sub(r"^(\w+) [\w.$]+: ", r"\1 ", str(templates[i])) for i in idx]
    for w in range(width, 121, 5):
        cut = [short(t, w) for t in full]
        if len(set(cut)) == len(cut):
            return cut
    return full


def plot_bars(values, templates, xlabel, path):
    order = np.argsort(np.abs(values))[-10:]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    # numeric positions: two templates can shorten to the same label
    ax.barh(range(len(order)), values[order],
            color=["#c0392b" if values[i] > 0 else "#2471a3" for i in order])
    ax.set_yticks(range(len(order)), figure_labels(templates, order))
    ax.set_xlabel(xlabel)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_figures(dataset, templates):
    """Draw the paper's figures from results/shap_<dataset>.npz (first seed's SHAP values)."""
    s = np.load(config.RESULTS_DIR / f"shap_{dataset}.npz")
    config.FIGURES_DIR.mkdir(exist_ok=True)
    plot_bars(np.abs(s["shap"]).mean(axis=0), templates,
              "mean |SHAP value| over detected anomalies",
              config.FIGURES_DIR / f"shap_global_{dataset}.png")
    plot_bars(s["shap"][int(s["example"])], templates,
              "SHAP value (contribution to the XHAD anomaly score)",
              config.FIGURES_DIR / f"shap_example_{dataset}.png")


class Shap:
    def __init__(self, xhad, d, seed):
        self.explainer = shap.KernelExplainer(
            xhad.score, shap.kmeans(d["X_train"], config.SHAP_BACKGROUND))
        self.score = xhad.score

    def __call__(self, X):
        return np.asarray(self.explainer.shap_values(X, nsamples=config.SHAP_NSAMPLES, silent=True))

    def fidelity(self, X, e):
        """Largest gap between base + sum(SHAP) and the score (0 = exact)."""
        base = float(np.ravel(self.explainer.expected_value)[0])
        return float(np.abs(base + e.sum(axis=1) - self.score(X)).max())


class Lime:
    def __init__(self, xhad, d, seed):
        rng = np.random.default_rng(seed)
        sample = d["X_train"][rng.choice(len(d["X_train"]), min(5000, len(d["X_train"])), replace=False)]
        # library defaults (quartile discretisation), the usual reference setting
        self.explainer = LimeTabularExplainer(sample, mode="regression", random_state=seed)
        self.score = xhad.score
        self.r2 = []

    def __call__(self, X):
        out = np.zeros_like(X)
        for i, x in enumerate(X):
            exp = self.explainer.explain_instance(x, self.score, num_features=X.shape[1],
                                                  num_samples=LIME_SAMPLES)
            for j, w in next(iter(exp.local_exp.values())):
                out[i, j] = w
            self.r2.append(exp.score)
        return out

    def fidelity(self, X, e):
        """Mean R^2 of LIME's local linear fit (1 = perfect)."""
        return float(np.mean(self.r2))


def explain_seed(dataset, d, seed):
    """Explain the same detected anomalies with SHAP and LIME. Returns metrics + artefacts."""
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
    X_pert = X * (1 + rng.uniform(-config.SHAP_PERTURB, config.SHAP_PERTURB, X.shape))

    # what the training labels say separates the classes
    y_tr = d["y_train_all"]
    signal = np.abs(d["X_train_all"][y_tr == 1].mean(0) - d["X_train_all"][y_tr == 0].mean(0))
    nearest = []
    if len(X) > 1:
        dist = np.linalg.norm(X[:, None] - X[None], axis=2)
        np.fill_diagonal(dist, np.inf)
        nearest = list(enumerate(dist.argmin(1)))

    out = {"alpha": xhad.alpha, "n_explained": int(len(hits))}
    explanations = {}
    for name, method in (("SHAP", Shap), ("LIME", Lime)):
        explain = method(xhad, d, seed)
        start = time.time()
        e = explain(X)
        seconds = (time.time() - start) / len(X)
        e_pert = explain(X_pert)
        importance = np.abs(e).mean(axis=0)
        consistency = [pair_scores(e[i], e[j]) for i, j in nearest]
        stability = [pair_scores(a, b) for a, b in zip(e, e_pert)]
        out[name] = {
            "fidelity": explain.fidelity(X, e),
            "agreement_spearman": float(spearmanr(importance, signal)[0]),
            f"agreement_top{K}_overlap": len(top_k(importance) & top_k(signal)),
            "consistency_spearman": float(np.nanmean([c[0] for c in consistency])),
            f"consistency_top{K}_jaccard": float(np.mean([c[1] for c in consistency])),
            "stability_spearman": float(np.nanmean([s[0] for s in stability])),
            f"stability_top{K}_jaccard": float(np.mean([s[1] for s in stability])),
            "seconds_per_explanation": seconds,
        }
        explanations[name] = e
    return out, dict(xhad=xhad, hits=hits, X=X, scores=scores, shap=explanations["SHAP"])


def write_artefacts(dataset, d, art):
    templates, hits, sv, scores = d["templates"], art["hits"], art["shap"], art["scores"]
    threshold = art["xhad"].threshold

    sheet = config.RESULTS_DIR / f"expert_review_{dataset}.csv"
    with open(sheet, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "test_row", "xhad_score", "threshold"]
                   + [f"reason_{r + 1}" for r in range(K)]
                   + ["makes_sense_1to5", "actionable_1to5", "comment"])
        for n, (row, s) in enumerate(zip(hits, sv), 1):
            reasons = [f"{templates[i]} ({s[i]:+.3f})" for i in np.argsort(-s)[:K]]
            w.writerow([n, int(row), f"{scores[row]:.3f}", f"{threshold:.3f}"] + reasons + ["", "", ""])

    # the example figure shows the highest-scoring explained anomaly
    worst = int(np.argmax(art["xhad"].score(art["X"])))
    np.savez(config.RESULTS_DIR / f"shap_{dataset}.npz",
             test_rows=hits, shap=sv, example=worst, threshold=threshold)
    plot_figures(dataset, templates)
    print(f"\nExample explanation (test row {hits[worst]}):")
    for i in np.argsort(-sv[worst])[:K]:
        print(f"  {sv[worst][i]:+.3f}  {short(templates[i], 90)}")


def main(dataset):
    d = prepare(dataset)
    runs = []
    for n, seed in enumerate(config.SEEDS):
        print(f"seed {seed} ...", flush=True)
        result, art = explain_seed(dataset, d, seed)
        runs.append(result)
        if n == 0:
            write_artefacts(dataset, d, art)

    summary = {"alpha_per_seed": [r["alpha"] for r in runs],
               "n_explained_per_seed": [r["n_explained"] for r in runs],
               f"top{K}_overlap_by_chance": K * K / d["X_train"].shape[1]}
    for name in ("SHAP", "LIME"):
        summary[name] = {key: {"mean": float(np.mean([r[name][key] for r in runs])),
                               "std": float(np.std([r[name][key] for r in runs]))}
                         for key in runs[0][name]}

    print(f"\n{dataset.upper()} - explanation quality, mean +/- std over {len(runs)} seeds\n")
    print(f"{'measure':<28}{'SHAP':>20}{'LIME':>20}")
    for key in runs[0]["SHAP"]:
        print(f"{key:<28}" + "".join(f"{summary[m][key]['mean']:>12.3g} +/- {summary[m][key]['std']:<5.2g}"
                                     for m in ("SHAP", "LIME")))

    config.RESULTS_DIR.mkdir(exist_ok=True)
    out = config.RESULTS_DIR / f"explain_{dataset}.json"
    out.write_text(json.dumps({"dataset": dataset, "seeds": config.SEEDS,
                               "summary": summary, "runs": runs}, indent=2))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    dataset = sys.argv[1].lower() if len(sys.argv) > 1 else ""
    if dataset not in config.DATASETS:
        raise SystemExit("usage: python explain.py hdfs|bgl [--figures-only]")
    if "--figures-only" in sys.argv:
        plot_figures(dataset, prepare(dataset)["templates"])
    else:
        main(dataset)
