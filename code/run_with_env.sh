#!/bin/bash
# Usage:
#   ASCAT_CRPS_ROOT=/path/to/research_checkout \
#     bash code/run_with_env.sh code/validation/score_ensemble_field.py --limit 2
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${HERE}/code${PYTHONPATH:+:$PYTHONPATH}"
if [[ -z "${ASCAT_CRPS_ROOT:-}" ]]; then
  echo "WARNING: ASCAT_CRPS_ROOT unset; scripts will try to auto-detect." >&2
fi
exec python "$@"
