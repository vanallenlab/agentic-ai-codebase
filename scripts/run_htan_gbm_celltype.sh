#!/usr/bin/env bash
# GBM cell-type annotation run. All steps and checks are in scripts/run_htan_task.sh.
exec bash "$(dirname "$0")/run_htan_task.sh" config/htan/opus-5-5/gbm/celltype_annotation.yaml
