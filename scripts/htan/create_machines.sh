#!/usr/bin/env bash
# =============================================================================
# Create the HTAN run VMs (default: sjohri-machine-1 .. sjohri-machine-5), each 16 vCPU / 128 GB
# (n2-highmem-16), Ubuntu 22.04, 550 GB boot disk for run folders, and the existing disk
# `htan-datasets` attached READ-ONLY (device /dev/disk/by-id/google-htan-datasets).
# Run it on your own computer (needs gcloud, logged in: `gcloud auth login`).
#
#   bash scripts/htan/create_machines.sh                     # project from `gcloud config`
#   PROJECT=my-project ZONE=us-central1-a N=5 bash scripts/htan/create_machines.sh
#   DRY_RUN=1 bash scripts/htan/create_machines.sh           # print the commands only
#
# Detach htan-datasets from any VM that has it read-write before running this: a disk can be
# attached read-only to several VMs only if no VM has it read-write (and only in its own zone).
# 5 x n2-highmem-16 needs 80 N2 CPUs of quota in the region.
#
# Afterwards, on each machine: see scripts/htan/README.md (clone, keys, docker login,
# setup_machine.sh /dev/disk/by-id/google-htan-datasets, machine_<N>.sh).
# Delete when done:  gcloud compute instances delete sjohri-machine-{1..5} --zone ZONE
# =============================================================================
set -euo pipefail
PROJECT=${PROJECT:-$(gcloud config get-value project 2>/dev/null)}
ZONE=${ZONE:-us-central1-a}
DISK=${DISK:-htan-datasets}
PREFIX=${PREFIX:-sjohri-machine}
N=${N:-5}
MACHINE_TYPE=${MACHINE_TYPE:-n2-highmem-16}       # 16 vCPU, 128 GB
BOOT_GB=${BOOT_GB:-550}
IMAGE_FAMILY=${IMAGE_FAMILY:-ubuntu-2204-lts}
[ -n "$PROJECT" ] || { echo "set PROJECT=... (or gcloud config set project ...)"; exit 1; }
G="gcloud --project $PROJECT"
run() { echo "+ $*"; [ "${DRY_RUN:-0}" = 1 ] || "$@"; }

echo "project $PROJECT, zone $ZONE, disk $DISK, $N x $MACHINE_TYPE"

# --- Create the VMs ---------------------------------------------------------------------
for i in $(seq 1 "$N"); do
    name=$PREFIX-$i
    if $G compute instances describe "$name" --zone "$ZONE" >/dev/null 2>&1; then
        echo "$name exists, skipping"; continue
    fi
    run $G compute instances create "$name" --zone "$ZONE" \
        --machine-type "$MACHINE_TYPE" \
        --image-family "$IMAGE_FAMILY" --image-project ubuntu-os-cloud \
        --boot-disk-size "${BOOT_GB}GB" --boot-disk-type pd-balanced \
        --disk "name=$DISK,mode=ro,device-name=$DISK,boot=no,auto-delete=no" \
        --scopes cloud-platform
done

echo
echo "Done. Connect with:  gcloud compute ssh $PREFIX-1 --zone $ZONE --project $PROJECT"
echo "Then follow scripts/htan/README.md; the data disk is /dev/disk/by-id/google-$DISK"
