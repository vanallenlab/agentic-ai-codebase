#!/usr/bin/env bash
# Machine 4 of 5: runs two cohorts at the same time, each in its own tmux session
# (htan_<cohort>), each worker with its own memory cap. Sized for a 16-vCPU / 128 GB VM.
# Each session runs every model set in turn (MODELS=, default: all; see scripts/run_htan_cohort.sh);
# finished tasks (output/<model>/<cohort>/<task>/DONE) are skipped, so re-running is safe.
# DRY_RUN=1 bash scripts/htan/machine_4.sh checks both cohorts without starting runs. See scripts/htan/README.md.
set -euo pipefail
D=$(dirname "$0")
# drop each finished run's checkpoints/ (~7 GB) so all models fit on the disk; PRUNE_CHECKPOINTS=0 keeps them
export PRUNE_CHECKPOINTS=${PRUNE_CHECKPOINTS:-1}
M3A_WORKER_MEMORY=24g M3A_WORKER_CPUS=7 bash "$D/run_hnsc.sh" || echo "!! $? from that cohort"
M3A_WORKER_MEMORY=24g M3A_WORKER_CPUS=7 bash "$D/run_skcm.sh" || echo "!! $? from that cohort"
