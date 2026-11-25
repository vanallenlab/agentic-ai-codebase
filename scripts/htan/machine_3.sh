#!/usr/bin/env bash
# Machine 3 of 5: runs two cohorts at the same time, each in its own tmux session
# (htan_<cohort>), each worker with its own memory cap. Sized for a 16-vCPU / 128 GB VM.
# DRY_RUN=1 bash scripts/htan/machine_3.sh checks both cohorts without starting runs. See scripts/htan/README.md.
set -euo pipefail
D=$(dirname "$0")
M3A_WORKER_MEMORY=56g M3A_WORKER_CPUS=7 bash "$D/run_paad.sh" || echo "!! $? from that cohort"
M3A_WORKER_MEMORY=24g M3A_WORKER_CPUS=7 bash "$D/run_cesc.sh" || echo "!! $? from that cohort"
