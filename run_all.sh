#!/usr/bin/env bash
# Regenerates the synthetic OSN data, checks the certificate implementation, and writes every table and figure to results/.
set -euo pipefail
cd "$(dirname "$0")"
python3 scripts/make_synth_osn.py
python3 -m pytest -q tests
python3 reproduce.py --config config/default.json --out results "$@"
