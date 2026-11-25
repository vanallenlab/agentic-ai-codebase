#!/usr/bin/env bash
# =============================================================================
# Run every HTAN task of one cancer type, one after another, in a tmux session.
#
#   bash scripts/run_htan_cohort.sh COHORT            # e.g. cesc; or scripts/htan/run_cesc.sh
#   DRY_RUN=1 bash scripts/run_htan_cohort.sh cesc     # pre-flight checks of every task, no runs
#   ONLY="celltype_annotation molecular_pik3ca" bash scripts/run_htan_cohort.sh cesc
#   SKIP="pancancer_clinical_survival" bash scripts/run_htan_cohort.sh cesc
#   MODEL_DIR=opus-4-6 (default) picks config/htan/<MODEL_DIR>/<COHORT>/*.yaml
#
# Each task goes through scripts/run_htan_task.sh (pre-flight checks, leakage guard, one
# isolated worker container). Runs are sequential because each worker may use up to 24 GB.
#
# Re-running the script is safe: a task that finished is marked with
# output/<cohort>/<task>/DONE and skipped. A task that failed is retried on the next start.
# Ctrl-c in the session stops the current task cleanly (results are saved) and does not
# start the next one.
#
# Per machine you need: this repo on branch harness-hardening, uv, Docker (the image is pulled
# on first use; `docker login ghcr.io` first), the cohort's data under DATA_MNT/htan_datasets,
# and ANTHROPIC_API_KEY (plus NCBI_API_KEY, WANDB_API_KEY if used) exported in ~/.bashrc.
# Memory: the worker container is capped at 24 GB (leakage_guard.yaml). BRCA (~41 GB of data),
# COADREAD (~33 GB) and PAAD (~28 GB) need a bigger machine: M3A_WORKER_MEMORY=<~1.6x data>g.
# Disk: ~20 GB per finished run in RUNS (checkpoints/ ~7 GB of it). PRUNE_CHECKPOINTS=1 deletes
# a run's checkpoints/ folder after it finishes successfully.
# =============================================================================
set -euo pipefail

COHORT=${1:?usage: run_htan_cohort.sh COHORT  (brca cesc coadread gbm hgsoc hnsc mm paad skcm ucec)}
REPO=$(cd "$(dirname "$0")/.." && pwd)
MODEL_DIR=${MODEL_DIR:-opus-4-6}
CFG_DIR=config/htan/$MODEL_DIR/$COHORT
cd "$REPO"
[ -d "$CFG_DIR" ] || { echo "no configs in $CFG_DIR"; ls "config/htan/$MODEL_DIR"; exit 1; }

# Which tasks: every config of the cohort, filtered by ONLY / SKIP (task names without .yaml).
TASKS=()
for f in "$CFG_DIR"/*.yaml; do
    t=$(basename "$f" .yaml)
    if [ -n "${ONLY:-}" ] && [[ " $ONLY " != *" $t "* ]]; then continue; fi
    if [[ " ${SKIP:-} " == *" $t "* ]]; then continue; fi
    TASKS+=("$t")
done
[ ${#TASKS[@]} -gt 0 ] || { echo "no tasks selected"; exit 1; }

done_marker() { echo "output/$COHORT/$1/DONE"; }

if [ "${DRY_RUN:-0}" = 1 ]; then
    fail=0
    for t in "${TASKS[@]}"; do
        if [ -f "$(done_marker "$t")" ]; then echo "== $t: already done"; continue; fi
        DRY_RUN=1 bash scripts/run_htan_task.sh "$CFG_DIR/$t.yaml" || { echo "!! $t: pre-flight FAILED"; fail=1; }
    done
    exit $fail
fi

# --- inside the tmux session: the loop itself -------------------------------------
if [ "${_IN_SESSION:-0}" = 1 ]; then
    eval "$(grep -E '^export (ANTHROPIC_API_KEY|NCBI_API_KEY|WANDB_API_KEY)=' ~/.bashrc || true)"
    export PATH="$HOME/.local/bin:$PATH"
    STOP=0
    trap 'STOP=1' INT          # Ctrl-c reaches the run (it saves and exits); we then stop the loop
    SUMMARY=logs/cohort_${COHORT}_$(date +%Y%m%d_%H%M%S).tsv
    mkdir -p logs; printf "task\tstatus\tminutes\n" > "$SUMMARY"
    for t in "${TASKS[@]}"; do
        if [ -f "$(done_marker "$t")" ]; then echo "== $t: already done, skipping"; continue; fi
        echo; echo "================ $COHORT / $t  ($(date '+%F %T')) ================"
        start=$(date +%s); rc=0
        FOREGROUND=1 bash scripts/run_htan_task.sh "$CFG_DIR/$t.yaml" || rc=$?
        mins=$(( ($(date +%s) - start) / 60 ))
        if [ "$STOP" = 1 ]; then
            printf "%s\tstopped\t%s\n" "$t" "$mins" >> "$SUMMARY"; echo "stopped by Ctrl-c"; break
        fi
        if [ "$rc" = 0 ]; then
            mkdir -p "output/$COHORT/$t"
            ptr=$(ls -t "output/$COHORT/$t"/run_*.json 2>/dev/null | head -1 || true)
            echo "${ptr:-}" > "$(done_marker "$t")"
            if [ "${PRUNE_CHECKPOINTS:-0}" = 1 ] && [ -n "$ptr" ]; then
                rd=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['run_dir'])" "$ptr")
                [ -d "$rd/checkpoints" ] && rm -rf "$rd/checkpoints" && echo "pruned $rd/checkpoints"
            fi
            printf "%s\tdone\t%s\n" "$t" "$mins" >> "$SUMMARY"
        else
            printf "%s\tFAILED(rc=%s)\t%s\n" "$t" "$rc" "$mins" >> "$SUMMARY"
            echo "!! $t failed (exit $rc), continuing with the next task"
        fi
    done
    echo; echo "== $COHORT finished. Summary ($SUMMARY):"; column -t "$SUMMARY"
    echo "press Enter to close"; read -r
    exit 0
fi

# --- start the tmux session -----------------------------------------------------------
SESSION=htan_$COHORT
tmux has-session -t "$SESSION" 2>/dev/null && { echo "tmux session $SESSION already exists (tmux attach -t $SESSION)"; exit 1; }
command -v tmux >/dev/null || { echo "tmux is not installed"; exit 1; }
echo "$COHORT: ${#TASKS[@]} task(s): ${TASKS[*]}"
tmux new-session -d -s "$SESSION" \
    "cd '$REPO' && _IN_SESSION=1 MODEL_DIR='$MODEL_DIR' ONLY='${ONLY:-}' SKIP='${SKIP:-}' PRUNE_CHECKPOINTS='${PRUNE_CHECKPOINTS:-0}' M3A_WORKER_MEMORY='${M3A_WORKER_MEMORY:-}' M3A_WORKER_CPUS='${M3A_WORKER_CPUS:-}' ALLOW_LOW_MEMORY='${ALLOW_LOW_MEMORY:-0}' DATA_MNT='${DATA_MNT:-}' RUNS='${RUNS:-}' DATA_DEV='${DATA_DEV:-}' bash scripts/run_htan_cohort.sh '$COHORT'"
cat <<EOF
Started tmux session '$SESSION'.
  tmux attach -t $SESSION     # watch; detach with Ctrl-b then d (Ctrl-c stops the current task and the loop)
  ls logs/ | grep -i $COHORT  # per-task logs; logs/cohort_${COHORT}_*.tsv is the summary
  ls output/$COHORT/*/DONE    # finished tasks
EOF
