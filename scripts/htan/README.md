# Running the HTAN cohorts on several machines

Each `machine_<N>.sh` starts two cohorts at once (each cohort: its tasks one after another, in
tmux session `htan_<cohort>`). All five are sized for a **16 vCPU / 128 GB** VM (e.g. n2-highmem-16).

| Machine | Cohorts (tasks) | Worker memory caps | Data loaded |
|---|---|---|---|
| 1 | brca (12) + gbm (3) | 80g + 24g | 41 + 2 GB |
| 2 | coadread (12) + mm (3) | 64g + 24g | 33 + 14 GB |
| 3 | paad (11) + cesc (10) | 56g + 24g | 28 + 12 GB |
| 4 | hnsc (13) + skcm (10) | 24g + 24g | 15 + 14 GB |
| 5 | ucec (12) + hgsoc (9) | 24g + 24g | 11 + 10 GB |

The big cohorts get fewer tasks because each of their steps is slower.

## Per machine

1. Create the VMs from your computer: `bash scripts/htan/create_machines.sh` (5 x n2-highmem-16, 550 GB boot disk,
   `htan-datasets` attached read-only). Detach the disk from the old VM's read-write attachment first.
   The boot disk holds the run folders (~20 GB per run; ~13 GB with `PRUNE_CHECKPOINTS=1`).
2. `git clone https://github.com/vanallenlab/agentic-ai-codebase && cd agentic-ai-codebase`
3. Put the `export ANTHROPIC_API_KEY=...` (and `NCBI_API_KEY`, `WANDB_API_KEY`) lines in `~/.bashrc`;
   `docker login ghcr.io` (the image is private: a GitHub token with `read:packages`).
4. `bash scripts/htan/setup_machine.sh /dev/disk/by-id/google-htan-datasets`
   (installs docker/tmux/uv if missing, mounts the data disk ro at `~/htan-disk-mt`, pulls the image).
5. `DRY_RUN=1 bash scripts/htan/machine_<N>.sh`, then `bash scripts/htan/machine_<N>.sh`.
6. Watch: `tmux ls`, `tmux attach -t htan_<cohort>`; finished tasks: `ls output/opus-5-5/*/*/DONE`.

Other models: `MODEL_DIR=<set> bash scripts/htan/machine_<N>.sh` with a set from `config/htan/`
(`gpt-6-2-sol`, `gpt-6-1-terra`, `qwen3-8-27b`, `muse-spark-open`, or one made with
`scripts/make_model_configs.py`). The pre-flight checks the matching key (`OPENAI_API_KEY`,
`OPENROUTER_API_KEY`) in `~/.bashrc`. Results go to `output/<set>/`.
   Re-running the machine script skips finished tasks and retries failed ones.

Results stay on each machine (`output/opus-5-5/<cohort>/` pointer files and the run folders in `~/agent-runs`);
copy both back, e.g. `rsync -a vm:agentic-ai-codebase/output/opus-5-5/ output/opus-5-5/` and
`rsync -a vm:agent-runs/ ~/agent-runs/` (run folder ids are random, so they do not collide).
