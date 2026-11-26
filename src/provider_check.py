"""Pre-flight check of the LLM provider a config uses: the right API key is set and works,
and the model exists. Used by scripts/run_htan_task.sh; keys are read from the environment
(never from the command line), and no key is ever printed.

    python -m src.provider_check CONFIG
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "azure_openai": "AZURE_OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


def _get(url: str, headers: dict) -> tuple[int, dict]:
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as f:
            return f.status, json.load(f)
    except urllib.error.HTTPError as e:
        try:
            body = json.load(e)
        except Exception:
            body = {}
        return e.code, body


def check(llm: dict) -> str:
    """Raise SystemExit with a clear message if the provider/key/model is not usable;
    return a one-line OK message otherwise."""
    provider = (llm.get("provider") or "azure_openai").lower()
    model = llm.get("model")
    env = KEY_ENV.get(provider)
    if env is None:
        sys.exit(f"unknown llm.provider {provider!r} (use one of {', '.join(KEY_ENV)})")
    key = llm.get("api_key") or os.environ.get(env)
    if not key:
        sys.exit(f"{env} is not set: add `export {env}=...` to ~/.bashrc")
    quoted = urllib.parse.quote(model or "", safe="")

    if provider == "anthropic":
        code, body = _get(f"https://api.anthropic.com/v1/models/{quoted}",
                          {"x-api-key": key, "anthropic-version": "2023-06-01"})
    elif provider == "openai":
        code, body = _get(f"https://api.openai.com/v1/models/{quoted}", {"Authorization": f"Bearer {key}"})
    elif provider == "azure_openai":
        return f"{env} set (Azure deployment {model} not checked)"
    else:  # openrouter
        code, body = _get("https://openrouter.ai/api/v1/key", {"Authorization": f"Bearer {key}"})
        if code == 200:
            data = body.get("data") or {}
            limit, usage = data.get("limit"), data.get("usage") or 0
            if limit is not None and usage >= limit:
                sys.exit(f"OpenRouter key has used its whole credit limit (${usage:.2f} of ${limit:.2f})")
            code2, cat = _get("https://openrouter.ai/api/v1/models", {"Authorization": f"Bearer {key}"})
            entry = next((m for m in (cat.get("data") or []) if m.get("id") == model), None)
            if entry is None:
                sys.exit(f"OpenRouter has no model {model!r}")
            if "tools" not in (entry.get("supported_parameters") or []):
                sys.exit(f"OpenRouter model {model} does not support tool calling")
            left = "" if limit is None else f", ${limit - usage:.2f} credit left"
            return f"{env} OK, model {model} available on OpenRouter{left}"

    if code == 401 or code == 403:
        sys.exit(f"{env} was rejected by {provider} (HTTP {code}): replace it in ~/.bashrc")
    if code == 404:
        sys.exit(f"{provider} has no model {model!r} for this key (HTTP 404)")
    if code != 200:
        msg = (body.get("error") or {}).get("message") if isinstance(body.get("error"), dict) else body.get("error")
        sys.exit(f"{provider} key/model check failed (HTTP {code}): {msg or body}")
    return f"{env} OK, model {model} available"


if __name__ == "__main__":
    sys.path.insert(0, ".")
    from src.config import load_config
    print(check(load_config(sys.argv[1]).get("llm") or {}))
