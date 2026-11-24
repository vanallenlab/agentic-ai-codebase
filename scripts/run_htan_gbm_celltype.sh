#!/usr/bin/env bash
# =============================================================================
# GBM cell-type annotation run (HTAN), step by step.
# Copy and paste the steps one at a time, or run the whole file:
#   bash scripts/run_htan_gbm_celltype.sh
#
# You do NOT start the Docker container yourself. The harness (run_agent_htan.py)
# runs on the VM and starts one worker container per run, from the lab runtime
# image, with only these mounts:
#   - src/ (read-only)
#   - this run's own folder, output/runs/<random id>/ (writable)
#   - the task's input files, at /inputs/... (read-only)
#   - an empty /app (hides build files and credentials baked into the image)
# The container has no network; literature and database tools run in the harness.
# All of this comes from config/htan/leakage_guard.yaml (pulled in by the task
# config with `extends:`), and the run refuses to start if any of it is missing.
# =============================================================================
set -euo pipefail

REPO=/home/sjohri/agentic-ai-codebase
CONFIG=config/htan/opus-4-6/gbm/celltype_annotation.yaml
IMAGE=ghcr.io/vanallenlab/agentic-sc-runtime:v2
cd "$REPO"

# --- 1. Data disk: mounted read-only at ~/htan-disk-mt, outside the repo -------
# data/datasets-mt in the repo is a symlink to it. The mount does not survive a reboot.
if ! findmnt -n /home/sjohri/htan-disk-mt >/dev/null; then
  sudo mount -o ro /dev/sdb /home/sjohri/htan-disk-mt
fi
findmnt -no TARGET,OPTIONS /home/sjohri/htan-disk-mt      # expect: ro
ls -l data/datasets-mt                                     # expect: -> /home/sjohri/htan-disk-mt

# --- 2. Branch and Python environment ----------------------------------------
git switch harness-hardening
git pull --ff-only || true      # nothing to pull if you haven't pushed the branch
export PATH="$HOME/.local/bin:$PATH"
uv sync --frozen

# --- 3. Runtime image (already pulled on this VM) -----------------------------
# Only needed on a new machine: docker login ghcr.io (GitHub classic token with read:packages)
docker image inspect "$IMAGE" >/dev/null 2>&1 || docker pull "$IMAGE"

# --- 4. API keys ---------------------------------------------------------------
# ~/.bashrc returns early in non-interactive shells, so load the key line directly.
eval "$(grep -E '^export ANTHROPIC_API_KEY=' ~/.bashrc | tail -1)"
curl -sS -o /dev/null -w "Anthropic API key check: HTTP %{http_code} (expect 200)\n" \
  https://api.anthropic.com/v1/models -H "x-api-key: $ANTHROPIC_API_KEY" -H "anthropic-version: 2023-06-01"
# Recommended: an NCBI key avoids HTTP 429 rate limits on PubMed / GEO lookups.
# export NCBI_API_KEY=...
# W&B: set WANDB_API_KEY to log to vanallenlab-agentic-ai/agentic-ai-htan,
# otherwise logging is skipped (the run still writes everything to its folder).
[ -n "${WANDB_API_KEY:-}" ] || export WANDB_MODE=disabled

# --- 5. Check the leakage guard before spending anything -----------------------
uv run python -c "
import sys; sys.path.insert(0, '.')
from src.config import load_config, resolve_literature_blacklist, check_leakage_guard
c = load_config('$CONFIG'); check_leakage_guard(c, resolve_literature_blacklist(c))
print('leakage guard OK:', c['llm']['model'], '| container', c['execution']['container']['image'])"

# --- 6. Start the run in tmux, so it survives an SSH disconnect ----------------
# Model and settings come from the config (claude-opus-4-6, reasoning_effort medium).
# Keys are read from ~/.bashrc inside the session, never put on a command line (visible in ps).
# Put `export NCBI_API_KEY=...` / `export WANDB_API_KEY=...` in ~/.bashrc to have them picked up.
mkdir -p logs
LOG=logs/gbm_celltype_$(date +%Y%m%d_%H%M%S).log
tmux new-session -d -s gbm_celltype "bash -c '
  cd $REPO
  eval \"\$(grep -E \"^export (ANTHROPIC_API_KEY|NCBI_API_KEY|WANDB_API_KEY)=\" ~/.bashrc)\"
  [ -n \"\${WANDB_API_KEY:-}\" ] || export WANDB_MODE=disabled
  export PATH=\"\$HOME/.local/bin:\$PATH\"
  uv run run_agent_htan.py --config $CONFIG 2>&1 | tee $LOG
  echo \"run finished (exit \${PIPESTATUS[0]}); press Enter to close\"; read'"
echo "Started. Log: $LOG"

# --- 7. Watch it ----------------------------------------------------------------
#   tmux attach -t gbm_celltype            # live view (detach: Ctrl-b then d)
#   tail -f logs/gbm_celltype_*.log        # or follow the log
#   docker ps --filter name=m3a-worker     # the worker container
#   docker stats --no-stream               # its memory (capped at 24g)
#
# Where the results are: the run folder has a random name. Its pointer file is in
# the task's output folder:
#   cat output/gbm/celltype_annotation/run_*.json      # -> "run_dir": ".../output/runs/<id>"
# The run folder holds agent.log, steps.jsonl (every tool call), conversation.jsonl,
# celltype_annotation.csv (the deliverable), plots and result_*.h5ad.
#
# Evaluation: cell IDs are now P-<n>_<barcode>, so join celltype_annotation.csv with
# the eval object on its `anon_cell_id` column, not on the index:
#   results/scrna/GBM/eval_obj_combined_adata_GBM_with_cancertype_sitetype_celltype.h5ad
#
# Stop early: tmux attach -t gbm_celltype, then Ctrl-c once. The harness shuts down
# cleanly and saves everything; it also stops the worker container.
