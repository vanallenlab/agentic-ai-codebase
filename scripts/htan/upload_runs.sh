#!/usr/bin/env bash
# Upload every run folder that is still on this machine to the bucket, and delete each local copy
# once its upload is verified. For runs that finished before UPLOAD_TO was set (the cohort script
# uploads new runs by itself).
#
#   bash scripts/htan/upload_runs.sh gs://<bucket>/<prefix>      # or with UPLOAD_TO in ~/.bashrc
#   DRY_RUN=1 bash scripts/htan/upload_runs.sh gs://...          # list what would be uploaded
#
# Safe while cohorts are running: the newest run of a task that has no DONE marker may still be
# in progress, so it is left alone while any htan_* tmux session exists.
set -uo pipefail
cd "$(dirname "$0")/../.."
DEST=${1:-${UPLOAD_TO:-$(sed -n 's/^export UPLOAD_TO=//p' ~/.bashrc 2>/dev/null | tail -1 | tr -d "\"'")}}
[ -n "$DEST" ] || { echo "usage: upload_runs.sh gs://<bucket>/<prefix>"; exit 1; }
running=0; tmux ls 2>/dev/null | grep -q '^htan_' && running=1

n=0; failed=0; skipped=0
for task_dir in output/*/*/*/; do
    task_dir=${task_dir%/}
    newest=$(ls -t "$task_dir"/run_*.json 2>/dev/null | head -1 || true)
    for p in "$task_dir"/run_*.json; do
        [ -e "$p" ] || continue
        rd=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['run_dir'])" "$p" 2>/dev/null) || continue
        [ -d "$rd" ] || continue
        if [ "$running" = 1 ] && [ "$p" = "$newest" ] && [ ! -f "$task_dir/DONE" ]; then
            echo "-- $task_dir: newest run may still be running, left alone"; skipped=$((skipped + 1)); continue
        fi
        if [ "${DRY_RUN:-0}" = 1 ]; then echo "would upload $rd ($(du -sh "$rd" | cut -f1)) for $task_dir"; n=$((n + 1)); continue; fi
        if bash scripts/htan/upload_run.sh "$p" "$DEST"; then n=$((n + 1)); else failed=$((failed + 1)); fi
    done
done
verb=uploaded; [ "${DRY_RUN:-0}" = 1 ] && verb="to upload"
echo "== $n run(s) $verb, $failed failed, $skipped left alone; free: $(df -h --output=avail "$HOME" | tail -1 | tr -d ' ')"
[ "$failed" = 0 ]
