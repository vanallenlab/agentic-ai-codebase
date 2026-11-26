#!/usr/bin/env bash
# Run all 10 HTAN CESC tasks (config/htan/<model>/cesc/, for every model set; MODELS= picks some) on this machine, one after another, in
# tmux session htan_cesc. Options (DRY_RUN=1, ONLY=, SKIP=, PRUNE_CHECKPOINTS=1, DATA_MNT=, RUNS=)
# are described in scripts/run_htan_cohort.sh.
exec bash "$(dirname "$0")/../run_htan_cohort.sh" cesc "$@"
