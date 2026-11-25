#!/usr/bin/env bash
# =============================================================================
# Run one HTAN task end to end, with pre-flight checks.
#
#   bash scripts/run_htan_task.sh [CONFIG]
#   bash scripts/run_htan_task.sh config/htan/opus-4-6/cesc/molecular_pik3ca.yaml   # default
#   bash scripts/run_htan_task.sh config/htan/opus-4-6/gbm/celltype_annotation.yaml
#   DRY_RUN=1 bash scripts/run_htan_task.sh [CONFIG]      # all checks, no run
#   FOREGROUND=1 bash scripts/run_htan_task.sh CONFIG     # run here, no tmux (used by run_htan_cohort.sh)
#
# Paths can be overridden for another machine (defaults are this VM's layout):
#   DATA_MNT  folder that holds htan_datasets/ (data/datasets-mt must point to it)  [~/htan-disk-mt]
#   RUNS      where run folders go (output/runs must point to it; needs ~20 GB/run) [~/agent-runs]
#   DATA_DEV  block device to mount at DATA_MNT if it is not mounted               [/dev/sdb]
#   M3A_WORKER_MEMORY / M3A_WORKER_CPUS   worker container limits [24g / 8 from leakage_guard.yaml];
#             set to ~85% of the machine's RAM on a bigger machine (e.g. M3A_WORKER_MEMORY=100g)
#
# You do NOT start Docker yourself: the harness (run_agent_htan.py) runs on this VM and
# starts one isolated worker container per run (only src/, the run folder and the task's
# input files are mounted; no network). The settings come from config/htan/leakage_guard.yaml,
# and the run refuses to start if any protection is missing.
#
# Storage: the data disk (/dev/sdb) is mounted read-write at ~/htan-disk-mt. Run folders live on
# it (htan-disk-mt/agent_runs), bind-mounted at the neutral path ~/agent-runs so the agent's
# working directory does not mention HTAN; output/runs is a symlink to ~/agent-runs. The root
# disk is too small for run outputs (checkpoints + results can reach 25-35 GB for a cohort).
# =============================================================================
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
CONFIG=${1:-config/htan/opus-4-6/cesc/molecular_pik3ca.yaml}
IMAGE=ghcr.io/vanallenlab/agentic-sc-runtime:v3
DATA_MNT=${DATA_MNT:-$HOME/htan-disk-mt}
RUNS=${RUNS:-$HOME/agent-runs}
DATA_DEV=${DATA_DEV:-/dev/sdb}
cd "$REPO"
[ -f "$CONFIG" ] || { echo "config not found: $CONFIG"; exit 1; }
NAME=$(echo "$CONFIG" | sed -E 's#.*/([^/]+)/([^/]+)\.yaml$#\1_\2#')     # e.g. cesc_molecular_pik3ca
echo "== $CONFIG  (tmux session: $NAME)"

# --- 1. Mounts (they do not survive a reboot; this re-creates them) -----------
# On another machine without a separate disk, DATA_MNT and RUNS can be plain folders.
if [ ! -d "$DATA_MNT/htan_datasets" ] && ! findmnt -n "$DATA_MNT" >/dev/null && [ -b "$DATA_DEV" ]; then
    sudo mount -o rw "$DATA_DEV" "$DATA_MNT"
fi
[ -d "$DATA_MNT/htan_datasets" ] || { echo "no htan_datasets/ in $DATA_MNT (set DATA_MNT=...)"; exit 1; }
if [ -d "$DATA_MNT/agent_runs" ] && [ "$RUNS" != "$DATA_MNT/agent_runs" ] && ! findmnt -n "$RUNS" >/dev/null; then
    mkdir -p "$RUNS"; sudo mount --bind "$DATA_MNT/agent_runs" "$RUNS"    # neutral path for the agent
fi
mkdir -p "$RUNS" output
[ -e data/datasets-mt ] || ln -s "$DATA_MNT" data/datasets-mt
[ -e output/runs ] || ln -s "$RUNS" output/runs
[ "$(readlink -f output/runs)" = "$(readlink -f "$RUNS")" ] || { echo "output/runs must be a symlink to $RUNS"; exit 1; }
[ "$(readlink -f data/datasets-mt)" = "$(readlink -f "$DATA_MNT")" ] || { echo "data/datasets-mt must point to $DATA_MNT"; exit 1; }
echo "paths OK: data $DATA_MNT, runs $RUNS"

# --- 2. Free disk space ---------------------------------------------------------
free_gb() { df -BG --output=avail "$1" | tail -1 | tr -dc 0-9; }
echo "free: runs disk $(free_gb $RUNS) GB, root disk $(free_gb /) GB"
[ "$(free_gb $RUNS)" -ge 60 ] || { echo "need >= 60 GB free on the runs disk"; exit 1; }
[ "$(free_gb /)" -ge 5 ] || { echo "need >= 5 GB free on the root disk (W&B cache, logs)"; exit 1; }

# --- 3. Branch, environment, image ----------------------------------------------
git switch -q harness-hardening
export PATH="$HOME/.local/bin:$PATH"
uv sync --frozen -q
docker image inspect "$IMAGE" >/dev/null 2>&1 || docker pull "$IMAGE"   # new machine: docker login ghcr.io first

# --- 4. Keys (read from ~/.bashrc, which returns early in non-interactive shells) ----
eval "$(grep -E '^export (ANTHROPIC_API_KEY|NCBI_API_KEY|WANDB_API_KEY)=' ~/.bashrc || true)"
code=$(curl -sS -o /dev/null -w "%{http_code}" https://api.anthropic.com/v1/models \
       -H "x-api-key: ${ANTHROPIC_API_KEY:-}" -H "anthropic-version: 2023-06-01" || true)
[ "$code" = 200 ] || { echo "Anthropic API key check failed (HTTP $code)"; exit 1; }
echo "Anthropic key OK"
[ -n "${NCBI_API_KEY:-}" ] || echo "note: no NCBI_API_KEY in ~/.bashrc -> PubMed/GEO lookups may hit rate limits (HTTP 429)"
if [ -n "${WANDB_API_KEY:-}" ] || grep -qs api.wandb.ai ~/.netrc; then echo "W&B: logged in"; else echo "W&B: not logged in -> logging disabled"; fi

# --- 5. Leakage guard, inputs and expected memory ---------------------------------
uv run python - "$CONFIG" <<'EOF'
import sys, h5py
sys.path.insert(0, ".")
from pathlib import Path
from src.config import load_config, resolve_literature_blacklist, check_leakage_guard
cfg = load_config(sys.argv[1]); check_leakage_guard(cfg, resolve_literature_blacklist(cfg))
c = cfg["execution"]["container"]
print(f"leakage guard OK | model {cfg['llm']['model']} | container {c['image']}, memory {c.get('memory')}, network {c['network']}")
total = 0
for k in ("scrna_data_path", "scatac_data_path", "scatac_snap_path", "cnmf_usage_path",
          "cnmf_spectra_scores_path", "cnmf_top_genes_path"):
    if not cfg.get(k):
        continue
    p = Path(cfg[k])
    if not p.exists():
        sys.exit(f"missing input {k}: {p}")
    size = p.stat().st_size
    if p.suffix == ".h5ad" and k != "scatac_snap_path":       # in-memory size of the loaded matrices
        with h5py.File(p) as f:
            size = 0
            def add(n, o):
                global size
                if isinstance(o, h5py.Dataset) and n.split("/")[0] in ("X", "layers", "raw", "obsm", "obsp"):
                    size += o.size * o.dtype.itemsize
            f.visititems(add)
        total += size
    print(f"  {k:26s} {size/1e9:6.2f} GB  {p.name}")
import os
limit = float(str(os.environ.get("M3A_WORKER_MEMORY") or c.get("memory", "24g")).rstrip("gG"))
print(f"  data loaded in the worker: ~{total/1e9:.1f} GB of the {limit:.0f} GB container limit"
      + ("  <-- WARNING: little room for the agent's copies" if total / 1e9 > 0.6 * limit else ""))
if total / 1e9 > 0.9 * limit and os.environ.get("ALLOW_LOW_MEMORY") != "1":
    sys.exit(f"the data alone needs ~{total/1e9:.0f} GB: set M3A_WORKER_MEMORY to at least "
             f"{int(total/1e9*1.6)}g on a machine with that much RAM (ALLOW_LOW_MEMORY=1 to try anyway)")
EOF
echo "deliverables:"; sed -n '/^### Deliverables/,/^### /p' "$(uv run python -c "
import sys; sys.path.insert(0,'.')
from src.config import load_config; print(load_config('$CONFIG')['task_file'])")" | grep '^- ' | cut -c1-110

[ "${DRY_RUN:-0}" = 1 ] && { echo "DRY_RUN=1: all checks passed, not starting the run"; exit 0; }

mkdir -p logs "$RUNS/_wandb"
LOG=logs/${NAME}_$(date +%Y%m%d_%H%M%S).log
if [ "${FOREGROUND:-0}" = 1 ]; then          # keys are already in this shell's environment
    [ -n "${WANDB_API_KEY:-}" ] || grep -qs api.wandb.ai ~/.netrc || export WANDB_MODE=disabled
    export WANDB_DIR=$RUNS/_wandb
    echo "log: $LOG"
    uv run run_agent_htan.py --config "$CONFIG" 2>&1 | tee -i "$LOG"
    exit "${PIPESTATUS[0]}"
fi

# --- 6. Start the run in tmux (survives SSH disconnects) ---------------------------
# Keys are loaded inside the session from ~/.bashrc, never put on a command line (ps).
tmux has-session -t "$NAME" 2>/dev/null && { echo "tmux session $NAME already exists (tmux attach -t $NAME)"; exit 1; }
tmux new-session -d -s "$NAME" "bash -c '
  cd $REPO
  eval \"\$(grep -E \"^export (ANTHROPIC_API_KEY|NCBI_API_KEY|WANDB_API_KEY)=\" ~/.bashrc)\"
  [ -n \"\${WANDB_API_KEY:-}\" ] || grep -qs api.wandb.ai ~/.netrc || export WANDB_MODE=disabled
  export WANDB_DIR=$RUNS/_wandb PATH=\"\$HOME/.local/bin:\$PATH\"
  uv run run_agent_htan.py --config $CONFIG 2>&1 | tee -i $LOG
  echo \"run finished (exit \${PIPESTATUS[0]}); press Enter to close\"; read'"
echo
echo "Started: tmux session '$NAME', log $LOG"
cat <<EOF

Watch it:
  tmux attach -t $NAME           # live; detach with Ctrl-b then d  (Ctrl-c STOPS the run)
  tail -f $LOG
  docker stats --no-stream        # worker memory
Results: the run folder has a random name; its pointer file is in the config's output_base_dir:
  cat \$(ls -t output/*/*/run_*.json | head -1)       # -> "run_dir": "$RUNS/<id>"
Stop early: tmux attach -t $NAME, then Ctrl-c once (saves everything, stops the container).
EOF
