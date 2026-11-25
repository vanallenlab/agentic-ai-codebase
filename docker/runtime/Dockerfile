# syntax=docker/dockerfile:1.7
# =============================================================================
# agentic-sc-runtime:v3 — the tool-worker image for HTAN runs (execution.mode: container).
#
# The harness starts one container per run from this image and mounts only src/, the
# task's input files and the run folder (see src/worker_client.py). The image therefore
# holds just the Python environment the worker tools need, plus the reference data that
# SnapATAC2 would otherwise download at run time (the worker has no network).
#
# Build (the reference files come from a local folder, see docker/runtime/README.md):
#   docker build -t ghcr.io/vanallenlab/agentic-sc-runtime:v3 \
#       --build-context refdata=<folder with the SnapATAC2 files> docker/runtime
#
# Differences from v2 (all found by running every tool in both images):
#   - packages pinned to the repo's uv.lock (snapatac2 2.8.0 needs anndata<0.11, pandas<2.1.2):
#     v2 had snapatac2 2.5.3 (no metrics.summary_by_chrom) and no multiprocess (macs3 peak calling)
#   - kaleido 0.2.1 (bundles its renderer); v2's kaleido 1.x needs Google Chrome, so every
#     static plotly/SnapATAC2 figure failed
#   - scanorama and gseapy added (scanorama is a batch-correction option of the tools;
#     agents use gseapy for enrichment)
#   - CPU-only torch (needed by harmonypy); no CUDA libraries
#   - hg38 genome, GENCODE v41 annotation and CIS-BP motifs baked in (SNAP_DATA_DIR)
#   - no build files, registry credentials or shell history in the image
# =============================================================================

ARG PYTHON_VERSION=3.10.19
ARG UV_VERSION=0.13.9

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

# ---- build stage: Python + packages (some need a compiler) -------------------------
FROM debian:trixie-slim AS build
ARG PYTHON_VERSION
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential zlib1g-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_PYTHON_INSTALL_DIR=/opt/python UV_PYTHON_PREFERENCE=only-managed \
    UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_NO_CACHE=1
RUN uv python install ${PYTHON_VERSION} && uv venv --python ${PYTHON_VERSION} /opt/venv
COPY requirements.txt /tmp/requirements.txt
RUN VIRTUAL_ENV=/opt/venv uv pip install --no-deps -r /tmp/requirements.txt \
        --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cpu \
    && VIRTUAL_ENV=/opt/venv uv pip check

# ---- reference data -------------------------------------------------------------------
FROM debian:trixie-slim AS ref
# Names and checksums as in SnapATAC2's own cache (pooch). The .gz is kept because pooch
# checks it before using the decompressed copy; the .fai index is pre-built because the
# folder is read-only at run time.
COPY --from=refdata gencode_v41_GRCh38.gff3.gz gencode_v41_GRCh38.fa.gz \
     gencode_v41_GRCh38.fa.gz.decomp gencode_v41_GRCh38.fa.gz.decomp.fai cisBP_human.meme \
     /opt/snapatac2_data/

# ---- final image -------------------------------------------------------------------------
FROM debian:trixie-slim
LABEL org.opencontainers.image.source="https://github.com/vanallenlab/agentic-ai-codebase" \
      org.opencontainers.image.description="Tool-worker runtime for M3A HTAN runs (v3)" \
      org.opencontainers.image.version="v3"
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates libgomp1 procps \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 worker
COPY --from=build /opt/python /opt/python
COPY --from=build /opt/venv /opt/venv
COPY --from=ref --chmod=a+rX /opt/snapatac2_data /opt/snapatac2_data
ENV PATH=/opt/venv/bin:$PATH \
    VIRTUAL_ENV=/opt/venv \
    SNAP_DATA_DIR=/opt/snapatac2_data \
    MPLBACKEND=Agg \
    PYTHONUNBUFFERED=1
USER worker
WORKDIR /home/worker
CMD ["python"]
