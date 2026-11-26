#!/usr/bin/env bash
# One-time setup of a fresh VM (Debian/Ubuntu) for HTAN runs. Run from the cloned repo:
#   bash scripts/htan/setup_machine.sh [DATA_DISK_DEVICE] [RUNS_DIR]
#   bash scripts/htan/setup_machine.sh /dev/disk/by-id/google-htan-data ~/agent-runs
# The HTAN data disk is expected to be attached READ-ONLY (it is mounted ro,noload at
# ~/htan-disk-mt). RUNS_DIR must be on a local disk with ~20 GB per run.
# Not done here (secrets): put `export ANTHROPIC_API_KEY=...` (and NCBI_API_KEY, WANDB_API_KEY)
# in ~/.bashrc, and `docker login ghcr.io` if the image is private.
set -euo pipefail
DEV=${1:-/dev/sdb}
RUNS=${2:-$HOME/agent-runs}
DATA_MNT=$HOME/htan-disk-mt
cd "$(dirname "$0")/../.."

command -v docker >/dev/null || { sudo apt-get update -q && sudo apt-get install -y -q docker.io; }
command -v tmux >/dev/null || sudo apt-get install -y -q tmux
# the host env (uv sync) builds a few packages from source (macs3 -> cykhash needs a C compiler, zlib)
dpkg -s build-essential zlib1g-dev >/dev/null 2>&1 || sudo apt-get install -y -q build-essential zlib1g-dev
command -v uv >/dev/null || [ -x "$HOME/.local/bin/uv" ] || curl -LsSf https://astral.sh/uv/install.sh | sh
id -nG | grep -qw docker || { sudo usermod -aG docker "$USER"; echo "added $USER to the docker group: log out and back in, then re-run"; }

mkdir -p "$DATA_MNT" "$RUNS"
if ! findmnt -n "$DATA_MNT" >/dev/null; then
    if [ "$(sudo blockdev --getro "$DEV")" = 1 ]; then sudo mount -o ro,noload "$DEV" "$DATA_MNT"
    else echo "warning: $DEV is attached read-write; mounting it read-only anyway"; sudo mount -o ro,noload "$DEV" "$DATA_MNT"; fi
fi
[ -d "$DATA_MNT/htan_datasets" ] || { echo "no htan_datasets/ on $DEV"; exit 1; }
grep -qs " $DATA_MNT " /etc/fstab || echo "$DEV $DATA_MNT ext4 ro,noload,nofail 0 0" | sudo tee -a /etc/fstab >/dev/null

git pull -q --ff-only   # the branch you cloned/switched to (BRANCH=<name> to switch first)
[ -z "${BRANCH:-}" ] || { git switch -q "$BRANCH" && git pull -q --ff-only; }
mkdir -p data output
[ -e data/datasets-mt ] || ln -s "$DATA_MNT" data/datasets-mt
[ -e output/runs ] || ln -s "$RUNS" output/runs
export PATH="$HOME/.local/bin:$PATH"
echo "installing the harness environment (uv sync, a few minutes)..."; uv sync --frozen
if id -nG | grep -qw docker; then echo "pulling the worker image (~9.5 GB)..."; docker pull ghcr.io/vanallenlab/agentic-sc-runtime:v3; fi
echo "setup done: data $DATA_MNT (read-only), runs $RUNS, free $(df -h --output=avail "$RUNS" | tail -1)"
echo "next: DRY_RUN=1 RUNS=$RUNS bash scripts/htan/machine_<N>.sh"
