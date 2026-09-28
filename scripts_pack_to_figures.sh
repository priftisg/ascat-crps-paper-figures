#!/bin/bash
# Copy regenerated unprefixed outputs into figures/main with manuscript IDs.
# Edit SRC_* paths after running generators.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN="${REPO}/figures/main"
map_copy() { [[ -f "$1" ]] && cp -a "$1" "$MAIN/$2" && echo "→ $2"; }
# Examples (uncomment / adjust after regeneration):
# map_copy evaluation/garg_followup/ceiling_vs_d0.png 01_ceiling_vs_d0.png
# map_copy evaluation/garg_followup/ceiling_focus_table.png 02_ceiling_focus_table.png
# map_copy evaluation/crps_field_calibration/field_crps_vs_orbit.png 03_field_crps_vs_orbit.png
# map_copy evaluation/crps_field_calibration/spread_skill_histogram.png 04_spread_skill_histogram.png
# map_copy evaluation/crps_field_calibration/spread_skill_vs_crps.png 05_spread_skill_vs_crps.png
# map_copy evaluation/isotonic_oos/oos_bss_focus_table.png 06_oos_bss_focus_table.png
# map_copy evaluation/isotonic_oos/reliability_oos_W6_d075.png 07_pmax_reliability_W6_d075.png
# map_copy evaluation/isotonic_oos/reliability_oos_W3_d0100.png 08_pmax_reliability_W3_d0100.png
# map_copy evaluation/crps_field_calibration/rank_histogram.png 09_rank_histogram.png
echo "Edit $0 with your regeneration outdirs, then re-run."
