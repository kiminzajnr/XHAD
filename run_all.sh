#!/usr/bin/env bash
# Reproduce every result in the paper, from raw logs to results/ and figures/.
# Takes a few hours on a laptop CPU. Logs go to logs/.
set -euo pipefail
cd "$(dirname "$0")/code"
PY=${PYTHON:-../xhad_env/bin/python}
mkdir -p ../logs

run() { echo "== $*"; "$PY" -u "$@" > "../logs/$(echo "$*" | tr ' -' '__').log" 2>&1; }

run features.py hdfs
run features.py bgl
for d in hdfs bgl; do
    run run_experiments.py "$d"
    run run_experiments.py "$d" --random-split
    run explain.py "$d"
done
run run_drift.py bgl
echo "done"
