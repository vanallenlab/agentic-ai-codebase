#!/usr/bin/env bash
# Run all 9 HTAN HGSOC tasks (config/htan/opus-5-5/hgsoc/, Claude Opus 5.5) on this machine, one after another, in
# tmux session htan_hgsoc. Options (DRY_RUN=1, ONLY=, SKIP=, PRUNE_CHECKPOINTS=1, DATA_MNT=, RUNS=)
# are described in scripts/run_htan_cohort.sh.
exec bash "$(dirname "$0")/../run_htan_cohort.sh" hgsoc "$@"
