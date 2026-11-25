# Worker runtime image (`ghcr.io/vanallenlab/agentic-sc-runtime:v3`)

The image the harness starts for the tool worker when `execution.mode: container`
(set for all HTAN configs in `config/htan/leakage_guard.yaml`). It only holds the Python
environment and reference data; the harness mounts `src/`, the task inputs and the run folder.

| File | What it is |
|---|---|
| `requirements.in` | the packages the worker tools need (direct dependencies) |
| `requirements.txt` | the exact, tested package set installed in the image (CPU torch) |
| `Dockerfile` | multi-stage build: uv-managed Python 3.10.19 + venv, reference data, non-root user |
| `fetch_refdata.sh` | downloads the hg38 / GENCODE v41 / CIS-BP files that SnapATAC2 needs offline |

## Build and push

```bash
bash docker/runtime/fetch_refdata.sh ~/htan-disk-mt/env_test/snapdata      # once, ~4 GB
docker buildx build --load -t ghcr.io/vanallenlab/agentic-sc-runtime:v3 \
    --build-context refdata=$HOME/htan-disk-mt/env_test/snapdata docker/runtime
docker push ghcr.io/vanallenlab/agentic-sc-runtime:v3
```

Changing a package: edit `requirements.in`, re-resolve `requirements.txt` with
`uv pip compile` against the repo's `uv.lock` versions, rebuild, and run the tool test
before pushing. `snapatac2==2.8.0` requires `anndata<0.11`, `pandas<2.1.2`, `numpy<2`, `plotly<6`;
keep `kaleido==0.2.1` (1.x needs Google Chrome).

## Why not v2

Running every HTAN tool in both images showed v2 breaks several of them: snapatac2 2.5.3
(no `metrics.summary_by_chrom` → scATAC QC fails), no `multiprocess` (peak calling fails),
kaleido 1.x without Chrome (every static plotly/SnapATAC2 figure fails), no offline genome /
motif files. v2 also contains registry credentials under `/app`.
