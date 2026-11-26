#!/usr/bin/env bash
# Run all 3 HTAN GBM tasks (config/htan/<model>/gbm/, for every model set; MODELS= picks some) on this machine, one after another, in
# tmux session htan_gbm. Options (DRY_RUN=1, ONLY=, SKIP=, PRUNE_CHECKPOINTS=1, DATA_MNT=, RUNS=)
# are described in scripts/run_htan_cohort.sh.
exec bash "$(dirname "$0")/../run_htan_cohort.sh" gbm "$@"
