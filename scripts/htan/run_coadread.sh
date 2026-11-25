#!/usr/bin/env bash
# Run all 12 HTAN COADREAD tasks (config/htan/opus-4-6/coadread/) on this machine, one after another, in
# tmux session htan_coadread. Options (DRY_RUN=1, ONLY=, SKIP=, PRUNE_CHECKPOINTS=1, DATA_MNT=, RUNS=)
# are described in scripts/run_htan_cohort.sh.
exec bash "$(dirname "$0")/../run_htan_cohort.sh" coadread "$@"
