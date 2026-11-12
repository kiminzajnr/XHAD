# XHAD
Explainable Hybrid Anomaly Detection for Distributed System Logs

XHAD combines an Isolation Forest and a dense autoencoder into one anomaly
score, `S(x) = alpha * S_IF(x) + (1 - alpha) * S_AE(x)`, and explains each
alarm with KernelSHAP: which log templates pushed the score up.

## Pipeline

```
raw log ──► features.py ──► run_experiments.py ──► results/detection_<ds>.json
 (HDFS.log,   Drain templates,     IF, AE, XHAD, PCA, OC-SVM
  BGL.log)    count per block/     5 seeds, chronological split
              window, time order
                              └──► explain.py ──────► results/explain_<ds>.json
                                   KernelSHAP on XHAD   results/expert_review_<ds>.csv
                                                        figures/shap_*_<ds>.png
```

| File | What it does |
|---|---|
| `code/config.py` | Every setting (split, seeds, model sizes, SHAP options). |
| `code/log_parser.py` | Drain parser: turns each log line into a template id. |
| `code/features.py` | Builds one row per HDFS block / BGL 20-line window: template counts, kept in time order. |
| `code/evaluation.py` | Chronological 60/20/20 split, scaling, threshold choice, metrics. |
| `code/models.py` | Isolation Forest, autoencoder, PCA, One-Class SVM and the XHAD ensemble. |
| `code/run_experiments.py` | Detection experiment and ablations. |
| `code/explain.py` | SHAP explanations and their quality measures. |

## Protocol

1. Rows are split by time: oldest 60% train, next 20% validation, newest 20% test.
2. Feature scaling and every detector are fitted on **normal** training rows only.
3. Alarm thresholds and XHAD's `alpha` are chosen on validation.
4. The frozen threshold is applied once to test. Results are mean ± std over 5 seeds.

A random split would put near-copies of test rows into training
(99.8% of HDFS test rows), which inflates scores; see Le & Zhang, ICSE 2022.

## Running

Put `HDFS.log`, `preprocessed/anomaly_label.csv` and `BGL.log` (from
[Loghub](https://github.com/logpai/loghub)) in `data/`, then:

```bash
cd code
python features.py hdfs          # once, ~5 min
python features.py bgl           # once, ~1 min
python run_experiments.py hdfs   # detection results
python run_experiments.py bgl
python explain.py hdfs           # SHAP explanations
python explain.py bgl
```
