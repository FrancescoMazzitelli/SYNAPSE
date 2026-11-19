#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python costs/estimate_costs.py --refresh "$@"
