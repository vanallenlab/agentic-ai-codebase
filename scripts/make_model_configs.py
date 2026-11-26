#!/usr/bin/env python3
"""Generate the 95 HTAN task configs for another model, from the reference set.

Same tasks, inputs, tools, leakage guard and worker image as config/htan/opus-4-6/; only the
`llm:` block, the output folder and (for text-only models) send_images change.

    python scripts/make_model_configs.py --provider anthropic  --model claude-opus-5-5
    python scripts/make_model_configs.py --provider openai     --model gpt-6.2-sol
    python scripts/make_model_configs.py --provider openai     --model gpt-6.1-terra
    python scripts/make_model_configs.py --provider openrouter --model qwen/qwen3.8-27b
    python scripts/make_model_configs.py ... --dir NAME   # folder name (default: from the model id)
    python scripts/make_model_configs.py ... --check      # print the llm block, write nothing

Writes config/htan/<dir>/<cohort>/<task>.yaml with output_base_dir output/<dir>/<cohort>/<task>/,
so `MODEL_DIR=<dir> bash scripts/htan/machine_<N>.sh` runs the set.

Per provider (the settings the harness supports are documented in src/llm_client.py):
  anthropic   prompt caching + keep-alive pings (30 min), effort medium, summarized thinking
  openai      store: false (the conversation is kept locally and nothing is stored at OpenAI;
              billed tokens are the same), reasoning summaries, caching is automatic
  openrouter  routing pinned to up to 3 hosts, picked from OpenRouter's endpoint list:
              hosts that support tools (and reasoning), not 4-bit quantized, the model maker's
              own host first, then by uptime and price; fallbacks stay within that list, so
              the prompt cache and the model's behaviour stay consistent within a run.
              Models without image input get send_images: false.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SOURCE = REPO / "config/htan/opus-4-6"
SOURCE_LLM = "llm:\n  provider: anthropic\n  model: claude-opus-4-6\n"
MAX_TOKENS = 64000
# OpenRouter model namespace -> the maker's own host tag (preferred when it serves the model)
MAKER_HOST = {"qwen": "alibaba", "meta": "meta", "meta-llama": "meta", "mistralai": "mistral",
              "deepseek": "deepseek", "google": "google-ai-studio", "moonshotai": "moonshotai",
              "z-ai": "z-ai", "minimax": "minimax"}
LOW_PRECISION = {"fp4", "int4", "fp6", "int6"}


def default_dir(model: str) -> str:
    name = model.split("/")[-1]
    name = re.sub(r"^claude-", "", name)
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def _openrouter(path: str) -> dict:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        sys.exit("OPENROUTER_API_KEY is not set (export it from ~/.bashrc)")
    req = urllib.request.Request(f"https://openrouter.ai/api/v1{path}", headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=60) as f:
        return json.load(f)


def openrouter_settings(model: str) -> tuple[dict, bool, list[str]]:
    """(routing, accepts_images, notes) for an OpenRouter model."""
    entry = next((m for m in _openrouter("/models")["data"] if m["id"] == model), None)
    if entry is None:
        sys.exit(f"OpenRouter has no model {model!r}")
    params = set(entry.get("supported_parameters") or [])
    if "tools" not in params:
        sys.exit(f"{model} does not support tool calling on OpenRouter")
    wants_reasoning = "reasoning" in params
    images = "image" in ((entry.get("architecture") or {}).get("input_modalities") or [])
    eps = _openrouter(f"/models/{model}/endpoints")["data"]["endpoints"]
    usable = []
    for ep in eps:
        sp = set(ep.get("supported_parameters") or [])
        if "tools" not in sp or (wants_reasoning and "reasoning" not in sp):
            continue
        if (ep.get("quantization") or "").lower() in LOW_PRECISION or ep.get("status", 0) not in (0, None):
            continue
        usable.append(ep)
    if not usable:
        sys.exit(f"no OpenRouter host serves {model} with tools{' + reasoning' if wants_reasoning else ''} "
                 "at 8-bit precision or better")
    maker = MAKER_HOST.get(model.split("/")[0])
    price = lambda ep: float((ep.get("pricing") or {}).get("prompt") or 0) + float((ep.get("pricing") or {}).get("completion") or 0)
    usable.sort(key=lambda ep: (
        0 if maker and (ep.get("tag") or "").split("/")[0] == maker else 1,
        -(ep.get("uptime_last_30m") or 0),
        0 if (ep.get("pricing") or {}).get("input_cache_read") else 1,
        price(ep),
    ))
    chosen = usable[:3]
    tags = [ep["tag"] for ep in chosen]
    notes = [f"{ep['tag']} ({ep.get('quantization') or 'unknown'} precision, ctx {ep.get('context_length')}, "
             f"cache {'yes' if (ep.get('pricing') or {}).get('input_cache_read') else 'no'}, "
             f"${float((ep.get('pricing') or {}).get('prompt') or 0) * 1e6:.2f}/${float((ep.get('pricing') or {}).get('completion') or 0) * 1e6:.2f} per M)"
             for ep in chosen]
    routing = {"order": tags, "only": tags, "allow_fallbacks": True}
    return routing, images, notes


def llm_block(provider: str, model: str, routing: dict | None) -> str:
    lines = ["llm:", f"  provider: {provider}", f"  model: {model}"]
    if provider == "anthropic":
        lines += ["  prompt_caching: true", f"  max_tokens: {MAX_TOKENS}", "  effort: medium",
                  "  thinking_display: summarized",
                  "  cache_keepalive_minutes: 30   # keep the prompt cache alive through tool calls up to 30 min"]
    elif provider == "openai":
        lines += [f"  max_tokens: {MAX_TOKENS}",
                  "  store: false                 # conversation kept locally; nothing stored at OpenAI",
                  "  reasoning_summary: auto"]
    elif provider == "openrouter":
        lines += [f"  max_tokens: {MAX_TOKENS}         # capped at the model's own limit",
                  "  routing:                      # OpenRouter hosts (see scripts/make_model_configs.py)",
                  f"    order: {json.dumps(routing['order'])}",
                  f"    only: {json.dumps(routing['only'])}",
                  f"    allow_fallbacks: {str(routing['allow_fallbacks']).lower()}",
                  "    data_collection: deny       # only hosts that do not store or train on prompts",
                  "    require_parameters: true"]
    else:
        sys.exit(f"unsupported provider {provider}")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", required=True, choices=["anthropic", "openai", "openrouter"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--dir", help="folder name under config/htan/ (default: from the model id)")
    ap.add_argument("--check", action="store_true", help="print the llm block only")
    a = ap.parse_args()

    routing, images, notes = None, True, []
    if a.provider == "openrouter":
        routing, images, notes = openrouter_settings(a.model)
    block = llm_block(a.provider, a.model, routing)
    out_name = a.dir or default_dir(a.model)
    print(f"{a.provider} {a.model} -> config/htan/{out_name}/  (images {'on' if images else 'OFF: model has no image input'})")
    for n in notes:
        print("  host:", n)
    print(block, end="")
    if a.check:
        return

    n = 0
    for src in sorted(SOURCE.glob("*/*.yaml")):
        text = src.read_text()
        if SOURCE_LLM not in text:
            sys.exit(f"{src}: llm block is not the expected reference block")
        cohort, task = src.parent.name, src.stem
        ob = f"output_base_dir: output/{cohort}/{task}/"
        if ob not in text:
            sys.exit(f"{src}: unexpected output_base_dir")
        text = text.replace(SOURCE_LLM, block).replace(ob, f"output_base_dir: output/{out_name}/{cohort}/{task}/")
        if not images:
            if re.search(r"^send_images:", text, re.M):
                text = re.sub(r"^send_images:.*$", "send_images: false   # model has no image input", text, flags=re.M)
            else:
                text = text.replace("interactive: false\n", "interactive: false\nsend_images: false   # model has no image input\n")
        header = (f"# Generated by scripts/make_model_configs.py from {src.relative_to(REPO)}: same task and inputs,\n"
                  f"# model {a.model} via {a.provider}.\n")
        dst = REPO / "config/htan" / out_name / cohort / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(header + text)
        n += 1
    print(f"wrote {n} configs to config/htan/{out_name}/")


if __name__ == "__main__":
    main()
