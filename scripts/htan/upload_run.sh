#!/usr/bin/env bash
# Upload one finished run to a Cloud Storage bucket and (by default) delete the local copy.
#
#   bash scripts/htan/upload_run.sh POINTER_JSON GS_PREFIX
#   bash scripts/htan/upload_run.sh output/gpt-6-1-terra/cesc/molecular_pik3ca/run_20261126_101500_ab12cd34ef56.json gs://my-bucket/htan
#
# POINTER_JSON is the file the harness writes in the task's output folder
# (output/<model>/<cohort>/<task>/run_<stamp>_<id>.json); it names the run folder (~/agent-runs/<id>).
# checkpoints/ is deleted first (PRUNE_CHECKPOINTS=0 keeps and uploads it). Uploads:
#   <run folder>            -> GS_PREFIX/agent-runs/<id>/
#   the task's output folder -> GS_PREFIX/output/<model>/<cohort>/<task>/   (pointer files, DONE)
# The local run folder is deleted only after the upload is verified (same number of files and
# bytes in the bucket); KEEP_LOCAL=1 keeps it. The pointer files and DONE stay on the machine, so
# finished tasks are still skipped. A record is appended to <output folder>/UPLOADED.
# Exit status: 0 uploaded (or already uploaded), 1 upload failed (the local copy is kept).
set -uo pipefail
PTR=${1:?usage: upload_run.sh POINTER_JSON GS_PREFIX}
DEST=${2:?usage: upload_run.sh POINTER_JSON GS_PREFIX}
DEST=${DEST%/}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

read -r RUN_DIR RUN_ID < <(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['run_dir'], d['run_id'])" "$PTR") \
    || { echo "upload: cannot read $PTR"; exit 1; }
OUT=$(dirname "$PTR")                       # output/<model>/<cohort>/<task>
REL_OUT=${OUT#output/}
if [ ! -d "$RUN_DIR" ]; then
    grep -qs " $RUN_ID " "$OUT/UPLOADED" && exit 0     # uploaded and deleted earlier
    echo "upload: run folder $RUN_DIR is missing"; exit 1
fi

# symlinks are not uploaded (they point at inputs outside the run), so only regular files count
local_count() { find "$RUN_DIR" -type f | wc -l; }
local_bytes() { find "$RUN_DIR" -type f -printf '%s\n' | awk '{s+=$1} END {print s+0}'; }
remote_stats() {   # "<files> <bytes>" under the run's prefix in the bucket
    gcloud storage ls -l "$DEST/agent-runs/$RUN_ID/**" 2>/dev/null \
        | awk '$1 ~ /^[0-9]+$/ && $3 ~ /^gs:/ {n++; s+=$1} END {print n+0, s+0}'
}

# checkpoints/ (~7 GB: data snapshots a run's own worker restarts from) are not needed once the run
# has ended, so they are deleted before uploading; PRUNE_CHECKPOINTS=0 uploads them too.
if [ "${PRUNE_CHECKPOINTS:-1}" = 1 ] && [ -d "$RUN_DIR/checkpoints" ]; then
    echo "pruning $RUN_DIR/checkpoints ($(du -sh "$RUN_DIR/checkpoints" | cut -f1)) before the upload"
    rm -rf "$RUN_DIR/checkpoints"
fi
size=$(du -sh "$RUN_DIR" | cut -f1)
echo "⬆️  uploading $RUN_DIR ($size) -> $DEST/agent-runs/$RUN_ID/"
mkdir -p logs; LOG=logs/upload_$RUN_ID.log    # gcloud prints a line per file; kept out of the tmux window
ok=0
for attempt in 1 2 3; do
    if gcloud storage rsync -r -q --ignore-symlinks "$RUN_DIR" "$DEST/agent-runs/$RUN_ID" >>"$LOG" 2>&1 \
        && gcloud storage rsync -r -q "$OUT" "$DEST/output/$REL_OUT" >>"$LOG" 2>&1; then
        read -r rn rb < <(remote_stats)
        ln=$(local_count); lb=$(local_bytes)
        if [ "$rn" = "$ln" ] && [ "$rb" = "$lb" ]; then ok=1; break; fi
        echo "upload: bucket has $rn files / $rb bytes, local $ln / $lb (attempt $attempt)"
    fi
    sleep ${UPLOAD_RETRY_S:-$((attempt * 30))}
done
if [ "$ok" != 1 ]; then
    tail -5 "$LOG" | sed 's/^/   gcloud: /'
    echo "!! upload of $RUN_DIR failed; the local copy is kept (run this script again to retry; log: $LOG)"
    exit 1
fi

echo "$(date '+%F %T') $RUN_ID $DEST/agent-runs/$RUN_ID/ $ln files" >> "$OUT/UPLOADED"
gcloud storage cp -q "$OUT/UPLOADED" "$DEST/output/$REL_OUT/UPLOADED" >/dev/null 2>&1 || true
if [ "${KEEP_LOCAL:-0}" = 1 ]; then
    echo "✅ uploaded ($ln files); local copy kept"
else
    rm -rf "$RUN_DIR" && echo "✅ uploaded ($ln files) and deleted the local copy ($size freed)"
fi
