"""Configuration loading and model constants."""

from pathlib import Path
from typing import Optional

import yaml

# Combined context window (input + output) limits (in tokens).
# Source: OpenAI API docs and Anthropic docs
# https://platform.openai.com/docs/models
# https://docs.anthropic.com/en/docs/about-claude/models
MODEL_CONTEXT_LIMITS = {
    # Direct OpenAI (platform.openai.com)
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4o-2024-08-06": 128_000,
    "o1": 200_000,
    "o1-mini": 128_000,
    # Azure OpenAI / OpenAI Responses API
    "gpt-5": 400_000,
    "o4-mini-0416": 128_000,
    # Anthropic Claude models
    "claude-3-5-sonnet-20241022": 200_000,
    "claude-3-5-haiku-20241022": 200_000,
    "claude-3-opus-20240229": 200_000,
    "claude-3-sonnet-20240229": 200_000,
    "claude-3-haiku-20240307": 200_000,
    "claude-4-sonnet-latest": 200_000,
    "claude-opus-4-6": 200_000,
    "claude-sonnet-4-6": 200_000,
}

# Max output tokens per response (subset of context window)
MODEL_OUTPUT_LIMITS = {
    # Direct OpenAI
    "gpt-4o": 16_384,
    "gpt-4o-mini": 16_384,
    "gpt-4o-2024-08-06": 16_384,
    "o1": 100_000,
    "o1-mini": 65_536,
    # Azure OpenAI / OpenAI
    "gpt-5": 128_000,
    "o4-mini-0416": 16_384,
    # Anthropic Claude models (all support up to 8K standard, can request higher)
    "claude-3-5-sonnet-20241022": 8192,
    "claude-3-5-haiku-20241022": 8192,
    "claude-3-opus-20240229": 4096,
    "claude-3-sonnet-20240229": 4096,
    "claude-3-haiku-20240307": 4096,
    "claude-4-sonnet-latest": 8192,
    "claude-opus-4-6": 16384,
    "claude-sonnet-4-6": 16384,
}

# Estimated tokens per image at different detail levels
# These remain heuristic, not official guarantees.
# USD per million tokens: (input, 5-min cache write, cache read, output).
# Source: platform.claude.com/docs/en/about-claude/pricing (checked 2026-11-24).
# Used only for the run's running cost estimate.
MODEL_PRICES = {
    "claude-opus-4-6": (5.0, 6.25, 0.50, 25.0),
    "claude-opus-4-7": (5.0, 6.25, 0.50, 25.0),
    "claude-opus-4-8": (5.0, 6.25, 0.50, 25.0),
    "claude-opus-5": (5.0, 6.25, 0.50, 25.0),
    "claude-opus-5-5": (4.0, 5.0, 0.20, 20.0),
    "claude-sonnet-4-6": (3.0, 3.75, 0.30, 15.0),
    "claude-sonnet-5-5": (2.0, 2.50, 0.10, 10.0),
}


def estimate_cost_usd(model: str, uncached: int, cache_write: int, cache_read: int, output: int) -> float:
    prices = MODEL_PRICES.get(model or "")
    if prices is None:
        return 0.0
    p_in, p_write, p_read, p_out = prices
    return (uncached * p_in + cache_write * p_write + cache_read * p_read + output * p_out) / 1e6


IMAGE_TOKENS_ESTIMATE = {
    "low": 85,
    "high": 765,  # Conservative upper bound for typical image inputs
    "auto": 765,
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(config_path: str, _seen: tuple = ()) -> dict:
    """Load configuration from YAML file.

    `extends: <path or list of paths>` (relative to the working directory) loads
    those files first; keys in this file override them (dicts are merged).
    """
    p = Path(config_path).resolve()
    if p in _seen:
        raise ValueError(f"config extends loop: {' -> '.join(map(str, _seen + (p,)))}")
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}
    parents = cfg.pop("extends", None) or []
    merged: dict = {}
    for parent in [parents] if isinstance(parents, str) else parents:
        merged = _deep_merge(merged, load_config(parent, _seen + (p,)))
    return _deep_merge(merged, cfg)


BLACKLIST_KEYS = ("pmids", "pmcids", "dois", "urls", "title_contains")


def resolve_literature_blacklist(config: dict) -> Optional[dict]:
    """Inline `literature_blacklist` plus `literature_blacklist_file`(s), merged (lists unioned)."""
    parts = []
    if config.get("literature_blacklist"):
        parts.append(config["literature_blacklist"])
    files = config.get("literature_blacklist_file") or []
    for fp in [files] if isinstance(files, str) else files:
        if not Path(fp).is_file():
            raise FileNotFoundError(f"literature_blacklist_file not found: {fp} "
                                    f"(generate it with scripts/make_literature_blacklist.py)")
        with open(fp) as f:
            parts.append(yaml.safe_load(f) or {})
    if not parts:
        return None
    merged: dict = {}
    for part in parts:
        if isinstance(part, list):  # legacy: plain list of URL/title substrings
            part = {"urls": part}
        for k, v in part.items():
            key = "pmcids" if k == "pmcid" else k
            merged.setdefault(key, [])
            merged[key] += [x for x in (v or []) if x not in merged[key]]
    return merged


def check_leakage_guard(config: dict, literature_blacklist: Optional[dict]) -> None:
    """`leakage_guard: true` refuses to run unless the agent is isolated from the answers:
    container worker mounting only code + inputs, no network, anonymous run directory,
    a non-empty literature blacklist."""
    if not config.get("leakage_guard"):
        return
    ex = config.get("execution") or {}
    c = ex.get("container") or {}
    problems = []
    if ex.get("mode") != "container":
        problems.append("execution.mode must be 'container'")
    if c.get("mount", "code") != "code":
        problems.append("execution.container.mount must be 'code'")
    if c.get("network", True) is not False:
        problems.append("execution.container.network must be false")
    if ex.get("agent_code_llm_keys"):
        problems.append("execution.agent_code_llm_keys must be false")
    if not config.get("anonymous_run_dir"):
        problems.append("anonymous_run_dir must be true")
    if config.get("output_dir"):
        problems.append("output_dir must not be set (it names the run directory)")
    if not literature_blacklist or not any(literature_blacklist.get(k) for k in BLACKLIST_KEYS):
        problems.append("a non-empty literature blacklist is required (literature_blacklist[_file])")
    if problems:
        raise ValueError("leakage_guard: refusing to run:\n  - " + "\n  - ".join(problems))
