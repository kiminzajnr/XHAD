"""
Main detection experiment: XHAD vs. its components and baselines.

For every seed, each detector is fitted on normal training rows, its
threshold is chosen on validation, and it scores the untouched test
split exactly once. Rows reported:

    Isolation Forest   - XHAD's IF component alone        (ablation)
    IF (normal-only)   - IF trained on normal rows only   (ablation: blind spot)
    Autoencoder        - XHAD's AE component alone        (ablation)
    XHAD (alpha=0.5)   - ensemble without weight tuning   (ablation)
    XHAD               - ensemble with alpha tuned on validation
    PCA, One-Class SVM - classical baselines

Usage:
    python run_experiments.py hdfs
    python run_experiments.py bgl
    python run_experiments.py bgl --random-split   # inflation check only
Writes results/detection_<dataset>[_random].json
"""

import json
import sys
import time

import numpy as np

import config
from evaluation import choose_threshold, describe, metrics, prepare
from models import OCSVM, XHAD, IForest, PCADetector

METRICS = ("precision", "recall", "f1", "auc_roc", "auc_pr")


def evaluate(scores_val, scores_test, d, threshold=None):
    if threshold is None:
        threshold = choose_threshold(scores_val, d["y_val"])
    return metrics(scores_test, d["y_test"], threshold)


def timed(fn, *args):
    start = time.time()
    out = fn(*args)
    return out, time.time() - start


def run_seed(dataset, seed, d):
    """All detectors for one seed. Returns {model: metrics + timings}, alpha, val F1 curve."""
    out = {}
    X_train, X_val, X_test = d["X_train"], d["X_val"], d["X_test"]
    per_row_ms = 1000 / len(X_test)

    xhad, fit_s = timed(XHAD(dataset, seed).fit, X_train, d["X_train_all"])
    (if_val, ae_val), (if_test, ae_test) = xhad.components(X_val), xhad.components(X_test)
    out["Isolation Forest"] = evaluate(if_val, if_test, d)
    if_normal = IForest(seed).fit(X_train)
    out["IF (normal-only)"] = evaluate(if_normal.score(X_val), if_normal.score(X_test), d)
    out["Autoencoder"] = evaluate(ae_val, ae_test, d)
    out["XHAD (alpha=0.5)"] = evaluate(0.5 * if_val + 0.5 * ae_val, 0.5 * if_test + 0.5 * ae_test, d)

    val_f1 = xhad.tune(X_val, d["y_val"])
    test_scores, score_s = timed(xhad.score, X_test)
    out["XHAD"] = metrics(test_scores, d["y_test"], xhad.threshold)
    out["XHAD"].update(fit_seconds=fit_s, infer_ms_per_row=score_s * per_row_ms)

    for name, model in (("PCA", PCADetector(dataset, seed)), ("One-Class SVM", OCSVM(seed))):
        _, fit_s = timed(model.fit, X_train)
        test_scores, score_s = timed(model.score, X_test)
        out[name] = evaluate(model.score(X_val), test_scores, d)
        out[name].update(fit_seconds=fit_s, infer_ms_per_row=score_s * per_row_ms)

    return out, xhad.alpha, val_f1


def summarise(runs):
    """{model: {metric: [per-seed]}} -> {model: {metric: {mean, std, runs}}}"""
    summary = {}
    for model in runs[0]:
        summary[model] = {}
        for key in runs[0][model]:
            values = [r[model][key] for r in runs]
            summary[model][key] = {"mean": float(np.mean(values)),
                                   "std": float(np.std(values)),
                                   "runs": [float(v) for v in values]}
    return summary


def print_table(dataset, summary):
    print(f"\n{dataset.upper()} - test split, mean +/- std over {len(config.SEEDS)} seeds\n")
    header = f"{'Model':<18}" + "".join(f"{m:>17}" for m in METRICS)
    print(header)
    print("-" * len(header))
    for model, s in summary.items():
        cells = "".join(f"   {s[m]['mean']:.3f} +/- {s[m]['std']:.3f}" for m in METRICS)
        print(f"{model:<18}{cells}")


def main(dataset, shuffle):
    d = prepare(dataset, shuffle)
    stats = describe(d)
    print(f"\n{dataset.upper()}: {stats['n_features']} features")
    for split in ("train", "val", "test"):
        print(f"  {split:<5} {stats[split + '_rows']:>8,} rows, "
              f"{100 * stats[split + '_anomaly_rate']:.2f}% anomalous")
    print(f"  test rows identical to a training row: {100 * stats['test_rows_seen_in_train']:.1f}%")

    runs, alphas, val_curves = [], [], []
    for seed in config.SEEDS:
        print(f"\nseed {seed} ...", flush=True)
        result, alpha, val_f1 = run_seed(dataset, seed, d)
        runs.append(result)
        alphas.append(alpha)
        val_curves.append(val_f1)
        print(f"  alpha={alpha}  XHAD F1={result['XHAD']['f1']:.3f}  "
              f"IF={result['Isolation Forest']['f1']:.3f}  AE={result['Autoencoder']['f1']:.3f}")

    summary = summarise(runs)
    print_table(dataset, summary)
    print(f"\nalpha chosen per seed: {alphas}")

    config.RESULTS_DIR.mkdir(exist_ok=True)
    out = config.RESULTS_DIR / f"detection_{dataset}{'_random' if shuffle else ''}.json"
    out.write_text(json.dumps({
        "dataset": dataset,
        "split": "random" if shuffle else "chronological",
        "threshold_rule": config.THRESHOLD_RULE,
        "seeds": config.SEEDS,
        "data": stats,
        "alpha_per_seed": alphas,
        "val_f1_by_alpha": {str(a): float(np.mean([c[a] for c in val_curves])) for a in config.ALPHAS},
        "results": summary,
    }, indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    dataset = sys.argv[1].lower() if len(sys.argv) > 1 else ""
    if dataset not in config.DATASETS:
        raise SystemExit("usage: python run_experiments.py hdfs|bgl [--random-split]")
    main(dataset, "--random-split" in sys.argv)
