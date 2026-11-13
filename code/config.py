"""
Every setting used by the XHAD experiments, in one place.

Change a value here and every script picks it up, so the numbers in the
thesis and paper always come from one known configuration.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FEATURE_DIR = DATA_DIR / "features"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

DATASETS = ("hdfs", "bgl")

# --- Log parsing (Drain) --------------------------------------------------
DRAIN_DEPTH = 4
DRAIN_SIMILARITY = 0.5
BGL_WINDOW = 20             # consecutive BGL lines grouped into one sample

# --- Chronological split: oldest 60% train, next 20% val, newest 20% test --
TRAIN_FRAC = 0.6
VAL_FRAC = 0.2

SEEDS = [42, 123, 456, 789, 999]

# --- Detectors --------------------------------------------------------------
IF_PARAMS = dict(n_estimators=100, max_samples=256)

# Input width differs a lot (HDFS ~30 templates, BGL ~300), so the
# autoencoder is sized per dataset.
AE_PARAMS = {
    "hdfs": dict(hidden=[32, 16], bottleneck=8),
    "bgl": dict(hidden=[128, 64], bottleneck=32),
}
AE_TRAIN = dict(epochs=50, batch_size=256, lr=1e-3, val_split=0.2, patience=5)

PCA_COMPONENTS = {"hdfs": 8, "bgl": 32}
OCSVM_PARAMS = dict(nu=0.05, n_components=200)

# XHAD: score = alpha * IF + (1 - alpha) * AE   (alpha=1 is pure IF)
ALPHAS = [round(0.1 * i, 1) for i in range(11)]

# How the alarm threshold is picked on the validation split:
#   "f1"     - maximise F1 (common in log anomaly detection papers)
#   "youden" - maximise TPR - FPR (what the proposal describes)
THRESHOLD_RULE = "f1"

# --- Drift experiment (run_drift.py) ----------------------------------------
# The test period (newest 20%) is cut into equal chunks; before each chunk
# the detectors are retrained on earlier rows, and the chunk just before it
# picks alpha and the threshold. Several retraining frequencies are reported.
DRIFT_CHUNKS = [5, 10, 20]
DRIFT_WINDOW_FRAC = 0.2     # sliding scheme: training rows kept

# --- SHAP explanations ------------------------------------------------------
SHAP_BACKGROUND = 50        # k-means summary of normal training rows
SHAP_NSAMPLES = "auto"      # KernelSHAP coalitions per explanation
SHAP_N_EXPLAIN = 50         # detected anomalies explained (also the expert-review set)
SHAP_TOP_K = 5              # templates shown per explanation
SHAP_PERTURB = 0.01         # stability test: scale each feature by up to +/-1%
