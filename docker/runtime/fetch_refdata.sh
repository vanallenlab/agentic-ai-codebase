#!/usr/bin/env bash
# Download the reference files SnapATAC2 needs (hg38 genome + GENCODE v41 annotation + CIS-BP
# motifs, ~4 GB) into OUT_DIR, with the file names and checksums SnapATAC2 expects, and build
# the FASTA index. The image build copies them in (see Dockerfile / README.md).
#
#   bash docker/runtime/fetch_refdata.sh /path/on/a/big/disk/snapatac2_data
set -euo pipefail
OUT=${1:?usage: fetch_refdata.sh OUT_DIR}
mkdir -p "$OUT"
cd "$(dirname "$0")/../.."
SNAP_DATA_DIR="$OUT" uv run python - <<'EOF'
import pyfaidx, snapatac2 as snap
snap.genome.hg38.annotation                 # gencode_v41_GRCh38.gff3.gz
fa = snap.genome.hg38.fasta                 # gencode_v41_GRCh38.fa.gz (+ .decomp)
pyfaidx.Fasta(str(fa))                      # writes the .fai index next to it
print(len(snap.datasets.cis_bp(unique=True)), "motifs")   # cisBP_human.meme
EOF
chmod a+r "$OUT"/*
ls -la "$OUT"
