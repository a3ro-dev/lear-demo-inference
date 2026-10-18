#!/usr/bin/env bash
# Runs on the GPU pod for every commit: install, deploy, benchmark.
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; . deploy/pod.env; set +a
python -m venv .venv
. .venv/bin/activate
pip install -q --disable-pip-version-check -r requirements.txt
ln -sfn "$PWD" /workspace/app
python bench.py
