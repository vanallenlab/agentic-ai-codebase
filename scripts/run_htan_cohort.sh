#!/usr/bin/env bash
# =============================================================================
# Run every HTAN task of one cancer type, for every model, one after another, in a tmux session.
#
#   bash scripts/run_htan_cohort.sh COHORT            # e.g. cesc; or scripts/htan/run_cesc.sh
#   DRY_RUN=1 bash scripts/run_htan_cohort.sh cesc     # pre-flight checks of every task, no runs
#   ONLY="celltype_annotation molecular_pik3ca" bash scripts/run_htan_cohort.sh cesc
#   SKIP="pancancer_clinical_survival" bash scripts/run_htan_cohort.sh cesc
#   MODELS="gpt-6-1-terra glm-5-3-flash" bash scripts/run_htan_cohort.sh cesc   # a subset of the models
#
# MODELS picks the config sets config/htan/<model>/<COHORT>/*.yaml, run in the order given; the
# default is every set below (DEFAULT_MODELS). MODEL_DIR=<one set> still works and means MODELS=<that set>.
# Make more sets with scripts/make_model_configs.py.
#
# Each task goes through scripts/run_htan_task.sh (pre-flight checks, leakage guard, one
# isolated worker container). Runs are sequential because each worker may use up to 24 GB.
#
# Re-running the script is safe: a task that finished is marked with
# <output_base_dir>/DONE (output/<model>/<cohort>/<task>/DONE) and skipped. A task that failed is
# retried on the next start. A model whose API key / model check fails is skipped (the other
# models still run). The loop stops (instead of failing every task) when the runs disk has less
# than MIN_FREE_GB (default 60) free.
# Ctrl-c in the session stops the current task cleanly (results are saved) and does not
# start the next one.
#
# Per machine you need: this repo, uv, Docker (the image is pulled
# on first use; `docker login ghcr.io` first), the cohort's data under DATA_MNT/htan_datasets,
# and the model providers' keys exported in ~/.bashrc: ANTHROPIC_API_KEY (Claude), OPENAI_API_KEY
# (GPT) and OPENROUTER_API_KEY (open-weight models), plus NCBI_API_KEY / WANDB_API_KEY if used.
# Memory: the worker container is capped at 24 GB (leakage_guard.yaml). BRCA (~41 GB of data),
# COADREAD (~33 GB) and PAAD (~28 GB) need a bigger machine: M3A_WORKER_MEMORY=<~1.6x data>g.
# Disk: ~20 GB per finished run in RUNS (checkpoints/ ~7 GB of it). PRUNE_CHECKPOINTS=1 deletes
# a run's checkpoints/ folder after it finishes successfully.
# UPLOAD_TO=gs://<bucket>/<prefix> (env or `export UPLOAD_TO=...` in ~/.bashrc) uploads each run
# folder when its task ends (done, failed or stopped) with scripts/htan/upload_run.sh and deletes
# the local copy once the upload is verified, so the disk never fills up.
# =============================================================================
set -euo pipefail

# Claude first (already done on the 5 machines, so skipped), then the cheaper / faster models.
DEFAULT_MODELS="opus-5-5 gpt-6-1-terra glm-5-3-flash muse-spark-open muse-glimmer-30b gpt-6-2-sol kimi-k3-1-pro"

COHORT=${1:?usage: run_htan_cohort.sh COHORT  (brca cesc coadread gbm hgsoc hnsc mm paad skcm ucec)}
REPO=$(cd "$(dirname "$0")/.." && pwd)
MODELS=${MODELS:-${MODEL_DIR:-$DEFAULT_MODELS}}
RUNS_DIR=${RUNS:-$HOME/agent-runs}
MIN_FREE_GB=${MIN_FREE_GB:-60}
cd "$REPO"

cfg_dir() { echo "config/htan/$1/$COHORT"; }
# Tasks of one model: every config of the cohort, filtered by ONLY / SKIP (task names without .yaml).
tasks_of() {
    local f t
    for f in "$(cfg_dir "$1")"/*.yaml; do
        [ -e "$f" ] || continue
        t=$(basename "$f" .yaml)
        if [ -n "${ONLY:-}" ] && [[ " $ONLY " != *" $t "* ]]; then continue; fi
        if [[ " ${SKIP:-} " == *" $t "* ]]; then continue; fi
        echo "$t"
    done
}
# A task's output folder is its config's output_base_dir (output/<model>/<cohort>/<task>/).
out_dir() { sed -n 's/^output_base_dir: *//p' "$(cfg_dir "$1")/$2.yaml" | sed 's:/*$::'; }
done_marker() { echo "$(out_dir "$1" "$2")/DONE"; }
free_gb() { mkdir -p "$RUNS_DIR"; df -BG --output=avail "$RUNS_DIR" | tail -1 | tr -dc 0-9; }
[ -n "${UPLOAD_TO:-}" ] || UPLOAD_TO=$(sed -n 's/^export UPLOAD_TO=//p' ~/.bashrc 2>/dev/null | tail -1 | tr -d "\"'")
UPLOAD_TO=${UPLOAD_TO%/}
# The bucket must be writable before hours of runs depend on it.
check_upload() {
    [ -n "$UPLOAD_TO" ] || return 0
    [[ "$UPLOAD_TO" == gs://* ]] || { echo "UPLOAD_TO must look like gs://<bucket>/<prefix>"; return 1; }
    command -v gcloud >/dev/null || { echo "UPLOAD_TO is set but gcloud is not installed"; return 1; }
    local probe; probe="$UPLOAD_TO/.write_test_$(hostname)_$$"
    if echo ok | gcloud storage cp -q - "$probe" >/dev/null 2>&1 && gcloud storage rm -q "$probe" >/dev/null 2>&1; then
        echo "upload OK: finished runs go to $UPLOAD_TO/ and are deleted locally"
    else
        echo "cannot write to $UPLOAD_TO (does the bucket exist? does this VM's service account have write access?)"; return 1
    fi
}
# Upload the run that just ended (the task's newest pointer file) and free its local disk.
upload_last() {
    [ -n "$UPLOAD_TO" ] || return 0
    local p; p=$(ls -t "$(out_dir "$1" "$2")"/run_*.json 2>/dev/null | head -1 || true)
    [ -n "$p" ] || return 0
    bash scripts/htan/upload_run.sh "$p" "$UPLOAD_TO" || echo "!! upload failed for $1 / $2; local copy kept"
}

# Check the model sets up front: a typo in MODELS should fail now, not hours later.
total=0; todo=0
for m in $MODELS; do
    [ -d "$(cfg_dir "$m")" ] || { echo "no configs in $(cfg_dir "$m") (sets: $(ls -d config/htan/*/ | xargs -n1 basename | tr '\n' ' '))"; exit 1; }
    for t in $(tasks_of "$m"); do
        total=$((total + 1)); [ -f "$(done_marker "$m" "$t")" ] || todo=$((todo + 1))
    done
done
[ "$total" -gt 0 ] || { echo "no tasks selected"; exit 1; }

if [ "${DRY_RUN:-0}" = 1 ]; then
    fail=0
    check_upload || fail=1
    for m in $MODELS; do
        for t in $(tasks_of "$m"); do
            if [ -f "$(done_marker "$m" "$t")" ]; then echo "== $m / $t: already done"; continue; fi
            DRY_RUN=1 bash scripts/run_htan_task.sh "$(cfg_dir "$m")/$t.yaml" || { echo "!! $m / $t: pre-flight FAILED"; fail=1; }
        done
    done
    exit $fail
fi

# --- inside the tmux session: the loop itself -------------------------------------
if [ "${_IN_SESSION:-0}" = 1 ]; then
    eval "$(grep -E '^export (ANTHROPIC_API_KEY|OPENAI_API_KEY|OPENROUTER_API_KEY|NCBI_API_KEY|WANDB_API_KEY)=' ~/.bashrc || true)"
    export PATH="$HOME/.local/bin:$PATH"
    STOP=0
    trap 'STOP=1' INT          # Ctrl-c reaches the run (it saves and exits); we then stop the loop
    SUMMARY=logs/cohort_${COHORT}_$(date +%Y%m%d_%H%M%S).tsv
    mkdir -p logs; printf "model\ttask\tstatus\tminutes\n" > "$SUMMARY"
    if ! check_upload; then echo "fix UPLOAD_TO (or unset it) and start again"; echo "press Enter to close"; read -r; exit 1; fi
    for m in $MODELS; do
        mapfile -t TASKS < <(tasks_of "$m")
        pending=()
        for t in "${TASKS[@]}"; do [ -f "$(done_marker "$m" "$t")" ] || pending+=("$t"); done
        if [ ${#pending[@]} = 0 ]; then echo "== $m: all ${#TASKS[@]} task(s) already done, skipping"; continue; fi
        echo; echo "######## $COHORT / $m: ${#pending[@]} of ${#TASKS[@]} task(s) to run ########"
        # one key / model check per model; a failing provider skips this model, not the whole cohort
        if ! uv run --frozen python -m src.provider_check "$(cfg_dir "$m")/${pending[0]}.yaml"; then
            echo "!! $m: provider check failed, skipping this model"
            printf "%s\t*\tSKIPPED(provider check)\t0\n" "$m" >> "$SUMMARY"; continue
        fi
        for t in "${pending[@]}"; do
            if [ -f "$(done_marker "$m" "$t")" ]; then echo "== $m / $t: already done, skipping"; continue; fi
            if [ "$(free_gb)" -lt "$MIN_FREE_GB" ]; then
                echo "!! only $(free_gb) GB free on $RUNS_DIR (need $MIN_FREE_GB): stopping. Free space and re-run; finished tasks are skipped."
                printf "%s\t%s\tSTOPPED(disk)\t0\n" "$m" "$t" >> "$SUMMARY"; STOP=1; break
            fi
            echo; echo "================ $COHORT / $m / $t  ($(date '+%F %T')) ================"
            start=$(date +%s); rc=0
            FOREGROUND=1 bash scripts/run_htan_task.sh "$(cfg_dir "$m")/$t.yaml" || rc=$?
            mins=$(( ($(date +%s) - start) / 60 ))
            if [ "$STOP" = 1 ]; then
                printf "%s\t%s\tstopped\t%s\n" "$m" "$t" "$mins" >> "$SUMMARY"; echo "stopped by Ctrl-c"
                upload_last "$m" "$t"; break
            fi
            if [ "$rc" = 0 ]; then
                od=$(out_dir "$m" "$t"); mkdir -p "$od"
                ptr=$(ls -t "$od"/run_*.json 2>/dev/null | head -1 || true)
                echo "${ptr:-}" > "$od/DONE"
                if [ "${PRUNE_CHECKPOINTS:-0}" = 1 ] && [ -n "$ptr" ]; then
                    rd=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['run_dir'])" "$ptr")
                    [ -d "$rd/checkpoints" ] && rm -rf "$rd/checkpoints" && echo "pruned $rd/checkpoints"
                fi
                printf "%s\t%s\tdone\t%s\n" "$m" "$t" "$mins" >> "$SUMMARY"
                upload_last "$m" "$t"
            else
                printf "%s\t%s\tFAILED(rc=%s)\t%s\n" "$m" "$t" "$rc" "$mins" >> "$SUMMARY"
                echo "!! $m / $t failed (exit $rc), continuing with the next task"
                upload_last "$m" "$t"
            fi
        done
        [ "$STOP" = 1 ] && break
    done
    echo; echo "== $COHORT finished. Summary ($SUMMARY):"; column -t "$SUMMARY"
    echo "press Enter to close"; read -r
    exit 0
fi

# --- start the tmux session -----------------------------------------------------------
SESSION=htan_$COHORT
command -v tmux >/dev/null || { echo "tmux is not installed"; exit 1; }
tmux has-session -t "$SESSION" 2>/dev/null && { echo "tmux session $SESSION already exists (tmux attach -t $SESSION)"; exit 1; }
echo "$COHORT: $todo of $total task(s) to run, models: $MODELS"
tmux new-session -d -s "$SESSION" \
    "cd '$REPO' && _IN_SESSION=1 MODELS='$MODELS' ONLY='${ONLY:-}' SKIP='${SKIP:-}' PRUNE_CHECKPOINTS='${PRUNE_CHECKPOINTS:-0}' UPLOAD_TO='$UPLOAD_TO' MIN_FREE_GB='$MIN_FREE_GB' M3A_WORKER_MEMORY='${M3A_WORKER_MEMORY:-}' M3A_WORKER_CPUS='${M3A_WORKER_CPUS:-}' ALLOW_LOW_MEMORY='${ALLOW_LOW_MEMORY:-0}' DATA_MNT='${DATA_MNT:-}' RUNS='${RUNS:-}' DATA_DEV='${DATA_DEV:-}' bash scripts/run_htan_cohort.sh '$COHORT'"
cat <<EOF
Started tmux session '$SESSION'.
  tmux attach -t $SESSION     # watch; detach with Ctrl-b then d (Ctrl-c stops the current task and the loop)
  ls logs/ | grep -i $COHORT  # per-task logs; logs/cohort_${COHORT}_*.tsv is the summary
  ls output/*/$COHORT/*/DONE  # finished tasks, per model
EOF
