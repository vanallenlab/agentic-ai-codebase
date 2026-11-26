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
uv run run_agent_htan.py --config config/htan/opus-5-5/brca/celltype_annotation.yaml
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

**Env:** set the key of the model provider. For W&B logging also set `WANDB_API_KEY`.

### Model providers

| `llm.provider` | Models | Key | Notes |
|---|---|---|---|
| `anthropic` | Claude | `ANTHROPIC_API_KEY` | explicit prompt caching + keep-alive pings during long tools |
| `openai` | GPT | `OPENAI_API_KEY` | Responses API; `store: false` keeps the conversation locally (nothing stored at OpenAI) |
| `openrouter` | open-weight models (Qwen, Muse, ...) | `OPENROUTER_API_KEY` | chat completions; host routing pinned per model; billed cost read from each response |
| `azure_openai` | GPT on Azure | `AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_ENDPOINT` | |

The agent loop, tools, worker and leakage guard are the same for all of them.
`python scripts/make_model_configs.py --provider <p> --model <id>` writes the 95 HTAN configs
for a model (config/htan/<model>/); `MODEL_DIR=<model> bash scripts/htan/machine_<N>.sh` runs them.

### Harness robustness options (all optional)

```yaml
tool_timeout_s: 7200            # wall-clock limit per tool call (0/null = none)
tool_output_max_chars: 100000   # larger outputs: head+tail to the model, full text in output/tool_outputs/
track_data_changes: true        # detect changes to shared data per call
checkpoint_data: true           # after each step, write modified data to output/checkpoints/ (atomic)
rollback_on_failure: true       # a FAILED call's changes to shared data are undone (reloaded from the last
                                # checkpoint) and the model is told; needs the two options above
seed: 0                         # python/numpy seed (null = don't seed)
image_max_edge: 1568            # plots sent to the model are downscaled to the model's native resolution (disk copies untouched)
send_images: true               # false = text-only ablation: plots are saved to disk but never shown to the model,
                                # and PIL / matplotlib image readers raise inside the REPL
llm:
  max_tokens: 16384             # per-response output cap (default by reasoning_effort: 8192/16384/32000)
  thinking: default             # default | adaptive | disabled  (default: adaptive only when reasoning_effort=high)
  effort: null                  # Anthropic output_config.effort (null = API default)
  thinking_display: null        # summarized = readable thinking in model_thinking (Opus 5.x returns it empty by default)
  max_retries: 8                # SDK retries for 429/5xx
  retry_total_seconds: 1800     # additional backoff window for transient API errors
  max_images_per_request: 600   # stop cleanly (everything saved) instead of sending a request the API rejects
  prompt_caching: true          # Anthropic: cache the conversation prefix (cuts input cost ~80%; outputs unchanged)
  cache_keepalive_minutes: 30   # Anthropic: keep the cache alive through long tool calls (0 = off)
  store: true                   # OpenAI: true = conversation kept at OpenAI; false = kept locally, nothing stored
  reasoning_summary: auto       # OpenAI: reasoning summaries in the step logs
  routing: {}                   # OpenRouter `provider` object (order/only/allow_fallbacks/...); defaults:
                                # require_parameters: true, data_collection: deny
logging:
  required: false               # if true, a wandb init failure aborts the run
```

### Tool execution isolation

Tools, the Python REPL and the datasets live in a separate **worker** process; the
harness (model loop, validation, logging, checkpoints) never runs agent code.
If the worker crashes, is OOM-killed or hangs past the hard timeout, the harness
kills it, starts a new one from the latest per-step checkpoints and reports the
failure to the model as the result of that call — the run continues.

```yaml
execution:
  mode: subprocess            # subprocess (default) | container | inprocess (debug only, no isolation)
  hard_timeout_grace_s: 120   # worker is killed if a call exceeds tool_timeout_s + this
  memory_limit_gb: null       # subprocess mode: address-space cap (MemoryError instead of OOM kill)
  container:                  # used when mode: container
    runtime: docker           # docker | podman
    image: ghcr.io/vanallenlab/agentic-sc-runtime:v2   # lab runtime image (see onboarding/README.md)
    memory: 48g               # cgroup memory limit (no swap)
    cpus: 8
    network: true             # false = no network (literature/database tools need it)
    gpus: null                # e.g. "all"
  agent_code_llm_keys: false  # LLM API keys are never visible to agent-written code (REPL/terminal);
                              # harness tools that need them (literature search) get them per call
```

Agent code is audited: each step in `steps.jsonl` carries `audit_flags`, `image_read` (code that opens or
decodes image files) and/or `llm_access` (code that references an LLM API or API keys). Flags never block a call.

In container mode the repo is mounted read-only, input data read-only, and only the
run's output directory is writable; paths are identical inside and outside the
container, and files are written as your user. Pull the image first:

```bash
docker login ghcr.io          # GitHub classic token with read:packages
docker pull ghcr.io/vanallenlab/agentic-sc-runtime:v2
```

Every run writes, as it goes: `agent.log`, `steps.jsonl` (every tool execution in full) and `conversation.jsonl` (every request/response; images referenced by file path). These are also streamed to W&B. SIGTERM/SIGHUP trigger a clean shutdown with all outputs saved, and the process exits non-zero if the run ended early.

## Layout

- `run_agent.py` — main entry point (HAI / interactive experiments)
- `run_agent_htan.py` — entry point for HTAN batch experiments (adds cNMF data loading)
- `src/` — agent loop, LLM client, tool registry, hooks, observability (W&B, Weave)
- `config/` — YAML configs (`hai/` for HAI datasets, `htan/<model>/` for HTAN cancer types: `opus-5-5`, `gpt-6-2-sol`,
  `gpt-6-1-terra`, `qwen3-8-27b`, `muse-spark-open`, `muse-glimmer-30b`, `kimi-k3-1-pro`, `glm-5-3-flash`; `htan/opus-4-6/` is the older Opus 4.6 set and the source of the others)
- `tasks/` — task prompt files (`.md` / `.txt`) referenced by configs
- `data/` — input datasets (see `data/README.md` for expected structure)
- `output/` — agent run results (see `output/README.md` for structure)
- `scripts/` — utility scripts for tool registry and MCP server generation
