# Evaluating agentic AI for biological discovery in autonomous and copilot settings

Code for the **Multistep Multimodal Multiomic Agentic (M3A)** framework — an evaluation framework for measuring how LLMs perform data-driven reasoning over complex computational biology workflows.

M3A enforces constraints across five core dimensions to enable rigorous tracing of agent behavior: (i) a standardized execution environment for reproducible baseline conditions; (ii) a unified tool suite spanning literature retrieval, programmatic analysis, terminal access, and domain-specific single-cell pipelines; (iii) recursive multimodal context integration across steps; (iv) persistent data state maintained across tool invocations; and (v) step-level telemetry capturing intent, tool selection, and outcomes at each decision point.

We applied M3A to evaluate a Claude Opus 4.6 (Feb 6, 2026) based agent across multi-step, multi-omic reasoning tasks with predefined evaluation endpoints spanning molecular profiling data from multiple cancer types.

## Environment

The environment is managed with [uv](https://docs.astral.sh/uv/) and fully pinned:
`.python-version` fixes the interpreter (a uv-managed CPython 3.10.19 build),
`pyproject.toml` pins direct dependencies, and `uv.lock` pins every transitive
package with hashes. Every machine that runs `uv sync --frozen` gets the same
environment.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # once per machine (uv >= 0.13.9)
uv sync --frozen                                  # creates .venv exactly from uv.lock
uv sync --frozen --extra scvi                     # optional: scvi-tools based tools
```

Do not run `uv lock` / `uv add` unless you intend to change the environment; commit
the updated `uv.lock` when you do. System libraries (glibc, GPU driver) are outside
uv's scope.

## Quick start

```bash
# HAI experiment (interactive)
uv run run_agent.py --config config/hai/jyates_eac.yaml --name your-name

# HTAN experiment (batch)
uv run run_agent_htan.py --config config/htan/opus-4-6/brca/celltype_annotation.yaml
```

Interactive mode: add `--interactive`. List available tools: `--list-tools`.

## Config

Each run is driven by a YAML config. Key fields:

```yaml
task_file: tasks/hai/jyates_eac.md        # analysis task prompt
scrna_data_path: data/.../scrna.h5ad       # scRNA-seq input
scatac_data_path: data/.../scatac.h5ad     # scATAC-seq input
output_base_dir: output/                   # run outputs written here
tool_category: scselected_scrna, scselected_scatac, external_database
reasoning_effort: medium                   # low / medium / high
interactive: true                          # prompt user after each tool call
llm:
  provider: anthropic
  model: claude-opus-4-6
logging:
  enabled: true
  project: my-wandb-project
  entity: my-wandb-entity
```

**Env:** set `ANTHROPIC_API_KEY`. For W&B logging also set `WANDB_API_KEY`.

### Harness robustness options (all optional)

```yaml
tool_timeout_s: 7200            # wall-clock limit per tool call (0/null = none)
tool_output_max_chars: 100000   # larger outputs: head+tail to the model, full text in output/tool_outputs/
track_data_changes: true        # detect changes to shared data per call; warn the model if a FAILED call modified data
checkpoint_data: true           # after each step, write modified data to output/checkpoints/ (atomic)
seed: 0                         # python/numpy seed (null = don't seed)
image_max_edge: 1568            # plots sent to the model are downscaled to the model's native resolution (disk copies untouched)
llm:
  max_tokens: 16384             # per-response output cap (default by reasoning_effort: 8192/16384/32000)
  thinking: default             # default | adaptive | disabled  (default: adaptive only when reasoning_effort=high)
  effort: null                  # Anthropic output_config.effort (null = API default)
  max_retries: 8                # SDK retries for 429/5xx
  retry_total_seconds: 1800     # additional backoff window for transient API errors
  max_images_per_request: 600   # stop cleanly (everything saved) instead of sending a request the API rejects
logging:
  required: false               # if true, a wandb init failure aborts the run
```

Every run writes, as it goes: `agent.log`, `steps.jsonl` (every tool execution in full) and `conversation.jsonl` (every request/response; images referenced by file path). These are also streamed to W&B. SIGTERM/SIGHUP trigger a clean shutdown with all outputs saved, and the process exits non-zero if the run ended early.

## Layout

- `run_agent.py` — main entry point (HAI / interactive experiments)
- `run_agent_htan.py` — entry point for HTAN batch experiments (adds cNMF data loading)
- `src/` — agent loop, LLM client, tool registry, hooks, observability (W&B, Weave)
- `config/` — YAML configs (`hai/` for HAI datasets, `htan/opus-4-6/` for HTAN cancer types)
- `tasks/` — task prompt files (`.md` / `.txt`) referenced by configs
- `data/` — input datasets (see `data/README.md` for expected structure)
- `output/` — agent run results (see `output/README.md` for structure)
- `scripts/` — utility scripts for tool registry and MCP server generation
