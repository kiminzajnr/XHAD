"""
Turn raw HDFS and BGL logs into template-count matrices, in time order.

    HDFS: one row per block (all lines mentioning that block id);
          labelled from anomaly_label.csv.
    BGL:  one row per window of BGL_WINDOW consecutive lines;
          anomalous if any line in it is.

Each row counts how often each Drain template occurs in it. Rows stay in
the order they first appear in the log, which is what makes the
chronological train/val/test split in evaluation.py meaningful.

Usage:
    python features.py hdfs      # full HDFS.log, ~10 min, run once
    python features.py bgl       # full BGL.log,  ~5 min, run once
Writes data/features/<dataset>.npz
"""

import re
import sys
import time
from collections import Counter

import numpy as np

import config
from log_parser import DrainLogParser

# Replace high-cardinality values so lines collapse onto their template.
# Order matters: most specific first.
_MASKS = [
    (re.compile(r'blk_-?\d+'), '<BLK>'),
    (re.compile(r'0x[0-9a-fA-F]+'), '<HEX>'),
    (re.compile(r'\b[0-9a-fA-F]{2}:[0-9a-fA-F]{6,}\b'), '<REG>'),
    (re.compile(r'\b\d+:[0-9a-fA-F]{6,}\b'), '<REG>'),
    # an address needs at least one hex letter; pure digits are numbers
    (re.compile(r'\b(?=[0-9a-fA-F]*[a-fA-F])[0-9a-fA-F]{8,}\b'), '<ADDR>'),
    (re.compile(r'\b(?:[Rr]\d+-){1,}[A-Za-z0-9\-:]+\b'), '<NODE>'),
    (re.compile(r'/?\b\d+\.\d+\.\d+\.\d+(:\d+)?\b'), '<IP>'),
    (re.compile(r'/[\w./\-]{4,}'), '<PATH>'),
    (re.compile(r'\b\d+\b'), '<NUM>'),
]
_BLK = re.compile(r'blk_-?\d+')


def mask(message):
    for pattern, placeholder in _MASKS:
        message = pattern.sub(placeholder, message)
    return message


def _progress(i, start):
    if i % 1_000_000 == 0 and i:
        print(f"  {i:,} lines  ({time.time() - start:.0f}s)", flush=True)


def _to_matrix(rows, parser):
    n = parser.template_counter
    X = np.zeros((len(rows), n), dtype=np.float32)
    for r, counts in enumerate(rows):
        for t, c in counts.items():
            X[r, t] = c
    templates = np.array([parser.id_to_template[t] for t in range(n)])
    return X, templates


def build_hdfs():
    labels = {}
    with open(config.DATA_DIR / "preprocessed" / "anomaly_label.csv") as f:
        next(f)
        for line in f:
            block, label = line.strip().split(',')
            labels[block] = int(label == "Anomaly")

    parser = DrainLogParser(config.DRAIN_DEPTH, config.DRAIN_SIMILARITY)
    blocks = {}  # dicts keep insertion order = order of first appearance
    start = time.time()
    with open(config.DATA_DIR / "HDFS.log", errors="ignore") as f:
        for i, line in enumerate(f):
            _progress(i, start)
            ids = set(_BLK.findall(line))
            parts = line.split(maxsplit=3)  # date, time, pid, message
            if not ids or len(parts) < 4:
                continue
            tid, _ = parser.parse_log(mask(parts[3]))
            for block in ids:
                if block in labels:
                    blocks.setdefault(block, Counter())[tid] += 1

    X, templates = _to_matrix(list(blocks.values()), parser)
    y = np.array([labels[b] for b in blocks], dtype=np.int8)
    return X, y, templates


def build_bgl():
    parser = DrainLogParser(config.DRAIN_DEPTH, config.DRAIN_SIMILARITY)
    rows, y = [], []
    window, anomalous, n = Counter(), 0, 0
    start = time.time()
    with open(config.DATA_DIR / "BGL.log", errors="ignore") as f:
        for i, line in enumerate(f):
            _progress(i, start)
            # label, unix ts, date, node, timestamp, node, facility (always RAS),
            # message. Dropping the constant facility lets Drain's prefix tree
            # branch on the first word of the actual message.
            parts = line.split(maxsplit=7)
            if len(parts) < 8:
                continue
            tid, _ = parser.parse_log(mask(parts[7]))
            window[tid] += 1
            anomalous |= parts[0] != "-"
            n += 1
            if n == config.BGL_WINDOW:
                rows.append(window)
                y.append(anomalous)
                window, anomalous, n = Counter(), 0, 0

    X, templates = _to_matrix(rows, parser)
    return X, np.array(y, dtype=np.int8), templates


BUILDERS = {"hdfs": build_hdfs, "bgl": build_bgl}


def load(dataset):
    """Return X (rows x templates, time-ordered), y (0/1), template strings."""
    path = config.FEATURE_DIR / f"{dataset}.npz"
    if not path.exists():
        raise SystemExit(f"{path} not found - run: python features.py {dataset}")
    d = np.load(path)
    return d["X"], d["y"].astype(int), d["templates"]


if __name__ == "__main__":
    dataset = sys.argv[1].lower() if len(sys.argv) > 1 else ""
    if dataset not in BUILDERS:
        raise SystemExit("usage: python features.py hdfs|bgl")

    print(f"Building {dataset.upper()} features from the raw log...")
    X, y, templates = BUILDERS[dataset]()

    config.FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    out = config.FEATURE_DIR / f"{dataset}.npz"
    np.savez_compressed(out, X=X, y=y, templates=templates)

    print(f"\nSaved {out}")
    print(f"  rows: {X.shape[0]:,}   templates: {X.shape[1]}")
    print(f"  anomalous: {y.sum():,} ({100 * y.mean():.2f}%)")
