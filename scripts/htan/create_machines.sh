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
# A disk can be attached read-only to several VMs only if NO VM has it read-write, and only
# in the disk's own zone. If it is still attached read-write (e.g. to the old VM), the script
# offers to stop that VM and detach the disk first.
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

# --- 1. The data disk: exists here, type supports read-only multi-attach, not attached rw ----
info=$($G compute disks describe "$DISK" --zone "$ZONE" --format='value(type.basename(),sizeGb)') \
    || { echo "disk $DISK not found in $ZONE (it must be in the same zone as the VMs)"; exit 1; }
echo "disk: $info"
users=$($G compute disks describe "$DISK" --zone "$ZONE" --format='value(users.basename())' | tr ';' ' ')
for vm in $users; do
    mode=$($G compute instances describe "$vm" --zone "$ZONE" --format=json \
           | python3 -c "import json,sys; d=json.load(sys.stdin); print(next(x['mode'] for x in d['disks'] if x['source'].endswith('/$DISK')))")
    [ "$mode" = READ_ONLY ] && continue
    echo "$DISK is attached READ_WRITE to $vm; it must be detached before read-only attachment."
    read -r -p "Stop $vm and detach $DISK from it now? [y/N] " a
    [ "$a" = y ] || { echo "aborted"; exit 1; }
    run $G compute instances stop "$vm" --zone "$ZONE"
    run $G compute instances detach-disk "$vm" --disk "$DISK" --zone "$ZONE"
done

# --- 2. CPU quota for the region -----------------------------------------------------------
REGION=${ZONE%-*}
need=$((16 * N))
$G compute regions describe "$REGION" --format=json | python3 -c "
import json, sys
need = $need
for q in json.load(sys.stdin)['quotas']:
    if q['metric'] in ('N2_CPUS', 'CPUS'):
        free = q['limit'] - q['usage']
        print(f\"quota {q['metric']}: {free:.0f} free of {q['limit']:.0f} (need {need})\")
        if free < need: print('  -> not enough: request more in IAM & Admin > Quotas, or use N=fewer')
"

# --- 3. Create the VMs ---------------------------------------------------------------------
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
