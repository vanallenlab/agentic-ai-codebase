#!/usr/bin/env bash
# Run all 3 HTAN MM tasks (config/htan/opus-4-6/mm/) on this machine, one after another, in
# tmux session htan_mm. Options (DRY_RUN=1, ONLY=, SKIP=, PRUNE_CHECKPOINTS=1, DATA_MNT=, RUNS=)
# are described in scripts/run_htan_cohort.sh.
exec bash "$(dirname "$0")/../run_htan_cohort.sh" mm "$@"
