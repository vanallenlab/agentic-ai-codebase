"""
Unified LLM client abstraction for OpenAI, Azure OpenAI, and Anthropic Claude.

This module provides a single interface for calling different LLM providers,
handling message format conversion, API differences, and response parsing.
"""

import copy
import json
import os
import time
from typing import Any, Optional
from dataclasses import dataclass


class UsageInfo:
    """Unified usage object that exposes attributes like the Azure OpenAI usage object."""
    def __init__(self, input_tokens: int = 0, output_tokens: int = 0,
                 input_tokens_details=None, output_tokens_details=None):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.input_tokens_details = input_tokens_details
        self.output_tokens_details = output_tokens_details
    
    def __repr__(self):
        return f"UsageInfo(input={self.input_tokens}, output={self.output_tokens})"


@dataclass
class LLMResponse:
    """Unified response object across providers."""
    id: str
    provider: str  # "openai", "azure_openai", or "anthropic"
    output: list  # Unified output format
    usage: Any  # Token usage object (attribute access: .input_tokens, .output_tokens, etc.)
    raw_response: Any  # Original response object
    # Normalized stop reason: "end_turn" | "tool_use" | "max_tokens" | "refusal" |
    # "pause_turn" | "stop_sequence" | None (unknown)
    stop_reason: Optional[str] = None


class RequestTooLargeError(Exception):
    """The conversation can no longer be sent without dropping history
    (e.g. more images than the API accepts in one request). Raised *before*
    the request so the run can stop cleanly with everything saved."""


# Models that predate adaptive thinking (use budget_tokens if thinking is requested)
_PRE_ADAPTIVE_MARKERS = ("claude-3", "claude-4-", "opus-4-0", "sonnet-4-0", "opus-4-1",
                         "opus-4-5", "sonnet-4-5", "haiku-4-5", "opus-4-2025", "sonnet-4-2025")


# Models on which thinking is on by default (omitting `thinking` runs adaptive)
_THINKING_DEFAULT_ON_MARKERS = ("opus-5", "sonnet-5", "haiku-5", "fable", "mythos")
# ...and on which `{"type": "disabled"}` is rejected outright
_THINKING_NOT_DISABLEABLE_MARKERS = ("opus-5-5", "sonnet-5-5", "fable", "mythos")


def _thinking_default_on(model: str) -> bool:
    m = (model or "").lower()
    return any(x in m for x in _THINKING_DEFAULT_ON_MARKERS)


def _supports_adaptive_thinking(model: str) -> bool:
    m = (model or "").lower()
    return not any(x in m for x in _PRE_ADAPTIVE_MARKERS)


class LLMClient:
    """Unified interface for LLM providers (OpenAI, Azure OpenAI, Anthropic)."""

    # Overridable from config (see create_llm_client_from_config)
    max_images_per_request: int = 600
    prompt_caching: bool = True   # Anthropic: cache the conversation prefix (billing only; outputs unchanged)
    retry_total_seconds: float = 1800.0  # extra patience on 429/5xx/overloaded beyond SDK retries
    # Anthropic: while a tool runs, keep the 5-minute prompt cache alive for up to this many
    # minutes with max_tokens=0 re-sends of the last request (cache read only; 0 = off).
    cache_keepalive_minutes: float = 30.0

    def __init__(self, provider: str, model: str, **kwargs):
        self.provider = provider.lower()
        self.model = model
        self.max_tokens_override = kwargs.pop("max_tokens", None)
        self.thinking_override = kwargs.pop("thinking", None)   # "adaptive" | "disabled" | None
        self.thinking_display = kwargs.pop("thinking_display", None)  # "summarized" | "omitted" | None
        self.effort_override = kwargs.pop("effort", None)       # output_config.effort, or None
        self.max_retries = int(kwargs.pop("max_retries", 8))

        if self.provider == "openai":
            self._init_openai(**kwargs)
        elif self.provider == "azure_openai":
            self._init_azure_openai(**kwargs)
        elif self.provider == "anthropic":
            self._init_anthropic(**kwargs)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def _init_openai(self, **kwargs):
        """Direct OpenAI API (platform.openai.com), not Azure."""
        from openai import OpenAI
        self.client = OpenAI(
            api_key=kwargs.get("api_key") or os.environ.get("OPENAI_API_KEY"),
            max_retries=self.max_retries,
        )
        self._conversation_id = None
    
    def _init_azure_openai(self, **kwargs):
        from openai import AzureOpenAI
        self.client = AzureOpenAI(
            api_key=kwargs.get("api_key") or os.environ["AZURE_OPENAI_API_KEY"],
            api_version=kwargs.get("api_version") or os.environ.get("AZURE_OPENAI_API_VERSION", "2025-03-01-preview"),
            azure_endpoint=kwargs.get("azure_endpoint") or os.environ["AZURE_OPENAI_ENDPOINT"],
            max_retries=self.max_retries,
        )
        self._conversation_id = None
    
    def _init_anthropic(self, **kwargs):
        import anthropic
        self.client = anthropic.Anthropic(
            api_key=kwargs.get("api_key") or os.environ.get("ANTHROPIC_API_KEY"),
            max_retries=self.max_retries,
        )
        # Anthropic conversation state: we maintain the full message list ourselves.
        # This is a list of dicts in Anthropic's native format:
        #   {"role": "user"|"assistant", "content": [...]}
        # System prompt is stored separately (Anthropic uses a top-level `system` param).
        self._conversation_history: list[dict] = []
        self._system_prompt: Optional[str] = None
        # Last main-conversation request (for cache keep-alive pings) and when it started:
        # the cache entry's 5-minute lifetime is counted from the start of a request.
        self._last_cached_request: Optional[dict] = None
        self._last_cache_touch: Optional[float] = None
    
    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def create(
        self,
        messages: list,
        tools: Optional[list] = None,
        reasoning_effort: str = "medium",
        store: bool = True,
        parallel_tool_calls: bool = False,
        tool_choice: Optional[dict] = None,
        previous_response_id: Optional[str] = None,
        isolated: bool = False,
    ) -> LLMResponse:
        if self.provider in ("openai", "azure_openai"):
            return self._create_openai_responses(
                messages, tools, reasoning_effort, store=store and not isolated,
                parallel_tool_calls=parallel_tool_calls, tool_choice=tool_choice,
                previous_response_id=previous_response_id
            )
        elif self.provider == "anthropic":
            if isolated:
                # Swap in a fresh list; the isolated call only ever appends to the
                # new list, so the main history object is untouched (no deep copy
                # of the full image-laden history needed).
                saved_history = self._conversation_history
                saved_system = self._system_prompt
                try:
                    self._conversation_history = []
                    return self._create_anthropic(
                        messages, tools, reasoning_effort, tool_choice,
                        parallel_tool_calls=parallel_tool_calls,
                        _isolated_mode=True  # Signal to filter problematic messages
                    )
                finally:
                    self._conversation_history = saved_history
                    self._system_prompt = saved_system
            else:
                return self._create_anthropic(messages, tools, reasoning_effort, tool_choice,
                                              parallel_tool_calls=parallel_tool_calls)

    # -------------------------------------------------------------------------
    # Prompt-cache keep-alive (Anthropic)
    # -------------------------------------------------------------------------
    def seconds_since_cache_touch(self) -> Optional[float]:
        if self.provider != "anthropic" or self._last_cached_request is None:
            return None
        return time.monotonic() - self._last_cache_touch

    def cache_keepalive(self):
        """Re-send the last main request with max_tokens=0: the API only reads the cached
        prefix (refreshing its 5-minute timer) and generates nothing. Returns the usage.
        Safe to call from another thread while the main thread waits on a tool."""
        kw = self._last_cached_request
        if kw is None:
            return None
        started = time.monotonic()
        resp = self.client.messages.create(**{**kw, "max_tokens": 0})
        self._last_cache_touch = started
        return resp.usage

    # -------------------------------------------------------------------------
    # Retry wrapper (on top of the SDK's own retries)
    # -------------------------------------------------------------------------
    def _with_retries(self, fn):
        """Call fn(); on transient errors (429, 5xx incl. overloaded, connection,
        timeout) keep retrying with exponential backoff for up to
        retry_total_seconds. Non-retryable errors (400, 401, 403, 404, 413) raise
        immediately."""
        import random
        import time as _time

        transient = []
        try:
            import anthropic
            transient += [anthropic.RateLimitError, anthropic.InternalServerError,
                          anthropic.APIConnectionError, anthropic.APITimeoutError]
        except ImportError:
            pass
        try:
            import openai
            transient += [openai.RateLimitError, openai.InternalServerError,
                          openai.APIConnectionError, openai.APITimeoutError]
        except ImportError:
            pass
        transient = tuple(transient)

        deadline = _time.monotonic() + float(self.retry_total_seconds or 0)
        attempt = 0
        while True:
            try:
                return fn()
            except transient as e:
                attempt += 1
                remaining = deadline - _time.monotonic()
                if remaining <= 0:
                    raise
                delay = min(300.0, 5.0 * (2 ** min(attempt, 6))) + random.uniform(0, 3)
                delay = min(delay, max(1.0, remaining))
                print(f"⚠️ LLM API transient error ({type(e).__name__}: {e}); "
                      f"retrying in {delay:.0f}s (attempt {attempt})")
                _time.sleep(delay)
    
    @property
    def conversation_id(self) -> Optional[str]:
        if self.provider in ("openai", "azure_openai"):
            return self._conversation_id
        return None
    
    def reset_conversation_history(self):
        if self.provider == "anthropic":
            self._conversation_history = []
    
    # -------------------------------------------------------------------------
    # OpenAI Responses API (direct OpenAI and Azure OpenAI share the same API)
    # -------------------------------------------------------------------------
    def _create_openai_responses(self, messages, tools, reasoning_effort, store,
                                  parallel_tool_calls, tool_choice, previous_response_id):
        """Call OpenAI Responses API (same for direct OpenAI and Azure OpenAI)."""
        messages = _strip_local_keys(messages)
        extra = {}
        if self.max_tokens_override:
            extra["max_output_tokens"] = int(self.max_tokens_override)
        response = self._with_retries(lambda: self.client.responses.create(
            model=self.model,
            input=messages,
            tools=tools or [],
            reasoning={"effort": reasoning_effort},
            store=store,
            parallel_tool_calls=parallel_tool_calls,
            **({"tool_choice": tool_choice} if tool_choice else {}),
            **({"previous_response_id": previous_response_id} if previous_response_id else {}),
            **extra,
        ))
        self._conversation_id = response.id

        stop_reason = None
        status = getattr(response, "status", None)
        if status == "incomplete":
            reason = getattr(getattr(response, "incomplete_details", None), "reason", None)
            stop_reason = "max_tokens" if reason == "max_output_tokens" else "refusal" \
                if reason == "content_filter" else str(reason)
        elif status == "completed":
            has_call = any(getattr(o, "type", None) == "function_call" for o in (response.output or []))
            stop_reason = "tool_use" if has_call else "end_turn"
        return LLMResponse(
            id=response.id, provider=self.provider,
            output=response.output, usage=response.usage, raw_response=response,
            stop_reason=stop_reason,
        )
    
    # -------------------------------------------------------------------------
    # Anthropic
    # -------------------------------------------------------------------------
    def _create_anthropic(self, messages, tools, reasoning_effort, tool_choice,
                           parallel_tool_calls: bool = False,
                           _isolated_mode: bool = False):
        """Call Anthropic Messages API.

        Anthropic multi-turn rules:
        1. Messages must strictly alternate: user, assistant, user, assistant, ...
        2. The first message must be role=user.
        3. tool_result blocks go in a user message and MUST reference a tool_use
           block in the immediately preceding assistant message.
        4. The system prompt is a separate top-level parameter, not a message.

        Strategy:
        - We maintain `_conversation_history` as the canonical Anthropic message list.
        - On each call, we convert the incoming OpenAI-format `messages` to Anthropic
          content blocks and append them to the history.
        - After the API call, we append the assistant's response to the history.
        - We never try to "merge" messages. Instead, we append content blocks to the
          last message if the role matches, or start a new message if it doesn't.
          This is safe because the caller always sends either:
            (a) Initial messages: [system, user] → system extracted, user appended
            (b) Tool results + images: [function_call_output, user(image)] → all user role
            (c) User feedback: [user message]

        Isolated mode (_isolated_mode=True):
        - History has been cleared by the caller
        - We filter out function_call_output messages since they require matching
          tool_use blocks in the history (which doesn't exist in isolated mode)
        - Only user/system messages are processed
        """
        # --- Step 1: Convert incoming messages and append to history ---
        # In isolated mode, filter out function_call_output to avoid tool_result errors
        if _isolated_mode:
            messages = [m for m in messages if m.get("type") != "function_call_output"]
        new_blocks = self._convert_incoming_messages(messages)
        
        for role, content_blocks in new_blocks:
            if not content_blocks:
                continue
            # Append to last message if same role, otherwise start new message
            if self._conversation_history and self._conversation_history[-1]["role"] == role:
                self._conversation_history[-1]["content"].extend(content_blocks)
            else:
                self._conversation_history.append({"role": role, "content": content_blocks})

        # The API requires every tool_result in a user turn to come before any
        # other content (images, text). With several tool results each followed by
        # its plots, the naive append order would be [result A, imgs A, result B]
        # → 400. Stable-sort the pending user turn: tool_results first.
        if self._conversation_history and self._conversation_history[-1]["role"] == "user":
            content = self._conversation_history[-1]["content"]
            if isinstance(content, list) and any(b.get("type") == "tool_result" for b in content):
                self._conversation_history[-1]["content"] = (
                    [b for b in content if b.get("type") == "tool_result"]
                    + [b for b in content if b.get("type") != "tool_result"]
                )

        # Pre-flight: refuse (cleanly) rather than send a request the API will reject.
        n_images = sum(
            1 for m in self._conversation_history
            for b in (m["content"] if isinstance(m["content"], list) else [])
            if isinstance(b, dict) and b.get("type") == "image"
        )
        if n_images > self.max_images_per_request:
            raise RequestTooLargeError(
                f"Conversation holds {n_images} images; the API accepts at most "
                f"{self.max_images_per_request} per request. Full history is required by "
                f"configuration, so the run is stopping here (everything is saved)."
            )

        # --- Step 2: Build API kwargs ---
        # Thinking blocks (with their signatures) are kept verbatim in history:
        # the API requires them to be passed back unchanged on tool-use
        # continuations when thinking is on.
        max_tokens_map = {"low": 8192, "medium": 16384, "high": 32000}
        max_tokens = int(self.max_tokens_override or max_tokens_map.get(reasoning_effort, 16384))
        messages = self._conversation_history
        if self.prompt_caching and not _isolated_mode and messages:
            # One breakpoint on the last block of the newest turn: each request
            # reads the previous turn's prefix (tools + system + history) from
            # cache and writes the new suffix. Applied to a per-request copy so
            # breakpoints never accumulate in the stored history (max 4).
            last = messages[-1]
            content = last["content"]
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            if content and isinstance(content[-1], dict):
                content = list(content[:-1]) + [{**content[-1], "cache_control": {"type": "ephemeral"}}]
                messages = list(messages[:-1]) + [{**last, "content": content}]
        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        thinking = self._thinking_config(reasoning_effort, _isolated_mode)
        if thinking is not None:
            kwargs["thinking"] = thinking
            if thinking.get("type") == "enabled":
                kwargs["max_tokens"] = max(max_tokens, thinking["budget_tokens"] + 4096)
        if self.effort_override and not _isolated_mode:
            kwargs["output_config"] = {"effort": self.effort_override}
        if self._system_prompt:
            kwargs["system"] = self._system_prompt
        if tools:
            kwargs["tools"] = self._convert_tools_to_anthropic(tools)
            tc = self._convert_tool_choice_to_anthropic(tool_choice) if tool_choice else {"type": "auto"}
            if not parallel_tool_calls and tc.get("type") in ("auto", "any", "tool"):
                tc = {**tc, "disable_parallel_tool_use": True}
            kwargs["tool_choice"] = tc
        elif tool_choice:
            kwargs["tool_choice"] = self._convert_tool_choice_to_anthropic(tool_choice)

        # --- Step 3: Call API ---
        # Streaming lets max_tokens be large without hitting HTTP timeouts.
        def _call(kw):
            with self.client.messages.stream(**kw) as stream:
                return stream.get_final_message()

        import anthropic
        started = time.monotonic()
        try:
            response = self._with_retries(lambda: _call(kwargs))
        except anthropic.BadRequestError as e:
            # Forced tool_choice is rejected by newer models (and with thinking on).
            # Fall back to auto + the instruction already in the prompt.
            tc = kwargs.get("tool_choice") or {}
            if tc.get("type") in ("any", "tool") and "tool_choice" in str(e):
                kwargs["tool_choice"] = {"type": "auto", "disable_parallel_tool_use": True}
                response = self._with_retries(lambda: _call(kwargs))
            else:
                raise
        if self.prompt_caching and not _isolated_mode:
            # Snapshot of exactly what was sent, so a keep-alive re-send matches the cached prefix.
            self._last_cached_request = {**kwargs, "messages": list(kwargs["messages"])}
            self._last_cache_touch = started

        # --- Step 4: Append assistant response to history ---
        # Serialize the response content blocks to plain dicts so they can be
        # re-sent in future requests. This preserves tool_use ids and thinking
        # signatures.
        assistant_content = self._serialize_response_content(response.content)
        if not assistant_content:
            # An empty assistant turn would make the next request invalid.
            assistant_content = [{"type": "text", "text": "(no content)"}]
        self._conversation_history.append({
            "role": "assistant",
            "content": assistant_content,
        })
        
        # --- Step 5: Convert to unified output format ---
        unified_output = self._convert_response_to_unified(response)
        
        # With caching, usage.input_tokens is only the UNCACHED part. Report the
        # full prompt size as input_tokens (context saturation, comparability with
        # earlier runs) and the cache split separately (cost).
        cache_read = getattr(response.usage, 'cache_read_input_tokens', 0) or 0
        cache_write = getattr(response.usage, 'cache_creation_input_tokens', 0) or 0
        uncached = response.usage.input_tokens or 0
        usage = UsageInfo(
            input_tokens=uncached + cache_read + cache_write,
            output_tokens=response.usage.output_tokens,
        )
        usage.input_tokens_details = type('Details', (), {
            'cached_tokens': cache_read, 'cache_write_tokens': cache_write,
            'uncached_tokens': uncached})()
        
        return LLMResponse(
            id=response.id, provider="anthropic",
            output=unified_output, usage=usage, raw_response=response,
            stop_reason=getattr(response, "stop_reason", None),
        )

    def _thinking_config(self, reasoning_effort: str, isolated: bool) -> Optional[dict]:
        """Thinking parameter for this request (None = omit).

        Default behaviour is unchanged from before for low/medium effort (no
        thinking parameter sent). For high effort, models from 4.6 on use
        adaptive thinking (budget_tokens is deprecated on 4.6 and rejected on
        later models); older models keep the fixed budget. `llm.thinking` in the
        config overrides this: "adaptive" | "disabled" | "default".

        Models with thinking on by default (Opus 5.x etc.) always think;
        `llm.thinking_display: summarized` makes them return a readable summary
        (the default display returns empty thinking text), which is what the
        step telemetry (`model_thinking`) records.
        """
        override = (self.thinking_override or "default").lower()
        if isolated:
            return None  # side-channel calls (hook, user questions) stay cheap
        m = (self.model or "").lower()
        if override == "disabled":
            if any(x in m for x in _THINKING_NOT_DISABLEABLE_MARKERS):
                raise ValueError(f"llm.thinking: disabled is not supported by {self.model} (thinking is "
                                 "always on; use llm.effort to control it)")
            return {"type": "disabled"}
        want = (override == "adaptive" or (override == "default" and reasoning_effort == "high")
                or _thinking_default_on(m))
        if not want:
            return None
        if _supports_adaptive_thinking(self.model):
            cfg = {"type": "adaptive"}
            if self.thinking_display:
                cfg["display"] = self.thinking_display
            return cfg
        return {"type": "enabled", "budget_tokens": 10000}

    def _convert_incoming_messages(self, messages: list) -> list[tuple[str, list[dict]]]:
        """Convert incoming OpenAI Responses-format messages to (role, content_blocks) pairs.
        
        Each pair represents content that should be appended to the conversation.
        System messages are extracted to self._system_prompt and not returned.
        
        Returns a list of (role, content_blocks) tuples. Content blocks are in
        Anthropic's native format (text, image, tool_result).
        """
        result = []
        
        for msg in messages:
            msg_type = msg.get("type")
            
            if msg_type == "message":
                role = msg.get("role")
                content = msg.get("content")
                
                # System prompt → store separately, not a message
                if role == "system":
                    if isinstance(content, str):
                        self._system_prompt = content
                    elif isinstance(content, list):
                        self._system_prompt = "\n".join(
                            c.get("text", "") for c in content
                            if isinstance(c, dict) and c.get("type") in ("text", "input_text")
                        )
                    continue
                
                if role not in ("user", "assistant"):
                    continue
                
                # Convert content to Anthropic blocks
                blocks = self._convert_content_blocks(content)
                if blocks:
                    result.append((role, blocks))
            
            elif msg_type == "function_call_output":
                # Tool result → user message with tool_result block.
                # Anthropic requires this to be in a user message immediately
                # after an assistant message containing the matching tool_use.
                result.append(("user", [{
                    "type": "tool_result",
                    "tool_use_id": msg.get("call_id"),
                    "content": msg.get("output", ""),
                }]))
        
        return result
    
    def _convert_content_blocks(self, content) -> list[dict]:
        """Convert a single message's content (string or list) to Anthropic content blocks."""
        if isinstance(content, str):
            return [{"type": "text", "text": content}]
        
        if not isinstance(content, list):
            return []
        
        blocks = []
        for c in content:
            if not isinstance(c, dict):
                continue
            
            c_type = c.get("type")
            
            if c_type in ("text", "input_text"):
                text = c.get("text", "")
                if text:
                    blocks.append({"type": "text", "text": text})
            
            elif c_type == "input_image":
                image_block = self._convert_image_block(c)
                if image_block:
                    blocks.append(image_block)
        
        return blocks
    
    def _convert_image_block(self, c: dict) -> Optional[dict]:
        """Convert an input_image block to Anthropic image format."""
        image_url = c.get("image_url")
        source = c.get("source")
        
        if image_url and isinstance(image_url, str) and image_url.startswith("data:"):
            # Parse data URL: "data:image/png;base64,<data>"
            header, _, b64data = image_url.partition(",")
            media_type = "image/png"
            if ":" in header and ";" in header:
                media_type = header.split(":")[1].split(";")[0]
            return {
                "type": "image",
                "source": {"type": "base64", "media_type": media_type, "data": b64data}
            }
        elif source and isinstance(source, dict):
            return {
                "type": "image",
                "source": {
                    "type": source.get("type", "base64"),
                    "media_type": source.get("media_type", "image/png"),
                    "data": source.get("data", ""),
                }
            }
        return None
    
    def _serialize_response_content(self, content) -> list[dict]:
        """Serialize Anthropic SDK response content blocks to plain dicts that
        are appended to _conversation_history and re-sent on every later call.

        - text blocks: type, text
        - tool_use blocks: type, id, name, input (id is needed by tool_result)
        - thinking / redacted_thinking blocks: passed back VERBATIM, including
          the signature. When thinking is on, the API requires them on tool-use
          continuations; dropping them breaks the request (or silently degrades
          reasoning on models that preserve thinking across turns).
        Empty text blocks are dropped (the API rejects them).
        """
        out = []
        for block in content:
            bt = self._get_block_type(block)

            if bt == "text":
                text = getattr(block, "text", "")
                if text:
                    out.append({"type": "text", "text": text})
            elif bt == "tool_use":
                out.append({
                    "type": "tool_use",
                    "id": getattr(block, "id", ""),
                    "name": getattr(block, "name", ""),
                    "input": getattr(block, "input", {}),
                })
            elif bt == "thinking":
                out.append({
                    "type": "thinking",
                    "thinking": getattr(block, "thinking", "") or "",
                    "signature": getattr(block, "signature", "") or "",
                })
            elif bt == "redacted_thinking":
                out.append({"type": "redacted_thinking", "data": getattr(block, "data", "")})
            elif hasattr(block, "model_dump"):
                # Unknown/new block type: pass back exactly as received.
                out.append(block.model_dump(mode="json", exclude_none=True))
        return out
    
    def _get_block_type(self, block) -> str:
        """Get content block type as a lowercase string, handling SDK enums."""
        if isinstance(block, dict):
            return block.get("type", "")
        t = getattr(block, "type", None)
        if t is None:
            return ""
        if isinstance(t, str):
            return t
        # Handle enums (e.g. ContentBlockType.tool_use)
        if hasattr(t, "value"):
            return str(t.value)
        return str(t).rsplit(".", 1)[-1].lower()

    def _convert_response_to_unified(self, response) -> list:
        """Convert Anthropic response to the unified output format used by the agent.

        Converts Anthropic content blocks to unified format:
        - thinking blocks → reasoning (with summary for logging)
        - text blocks → message with content
        - tool_use blocks → function_call

        Multiple text blocks are combined into a single message for cleaner output.
        """
        output = []
        text_parts = []
        thinking_parts = []

        for block in response.content:
            bt = self._get_block_type(block)
            if bt == "thinking":
                thinking_text = getattr(block, "thinking", "")
                if thinking_text:
                    thinking_parts.append(thinking_text)
            elif bt == "text":
                text_parts.append(block.text)
            elif bt == "tool_use":
                output.append({
                    "type": "function_call",
                    "name": block.name,
                    "call_id": block.id,
                    "arguments": json.dumps(block.input),
                })

        # Emit thinking blocks as unified "reasoning" items so parse_response can log them
        if thinking_parts:
            output.insert(0, {
                "type": "reasoning",
                "summary": "\n".join(thinking_parts),
            })

        # Combine all text into a single message (cleaner than multiple messages)
        if text_parts:
            # Insert after reasoning (if present) but before tool calls
            insert_idx = 1 if thinking_parts else 0
            output.insert(insert_idx, {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "\n".join(text_parts)}]
            })

        return output
    
    # -------------------------------------------------------------------------
    # Tool format conversion
    # -------------------------------------------------------------------------
    def _convert_tools_to_anthropic(self, tools: list) -> list:
        claude_tools = []
        for tool in tools:
            if "input_schema" in tool:
                claude_tools.append(tool)
            elif tool.get("type") == "function":
                claude_tools.append({
                    "name": tool.get("name"),
                    "description": tool.get("description"),
                    "input_schema": tool.get("parameters", {}),
                })
        return claude_tools
    
    def _convert_tool_choice_to_anthropic(self, tool_choice: dict) -> dict:
        if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
            return {"type": "tool", "name": tool_choice.get("name")}
        return tool_choice


def _strip_local_keys(messages: list) -> list:
    """Remove harness-only keys (e.g. an image's local 'path') before sending
    messages to an API that validates unknown fields."""
    out = []
    for m in messages:
        if isinstance(m, dict) and isinstance(m.get("content"), list):
            m = {**m, "content": [
                {k: v for k, v in c.items() if k != "path"} if isinstance(c, dict) else c
                for c in m["content"]
            ]}
        out.append(m)
    return out


def create_llm_client_from_config(config: dict) -> tuple[LLMClient, str]:
    """Create LLM client from configuration.
    
    Config under `llm`:
        provider: "openai" | "azure_openai" | "anthropic"
        model: model name or deployment name (required for provider selection)
        api_key: optional; otherwise uses OPENAI_API_KEY, AZURE_OPENAI_API_KEY, or ANTHROPIC_API_KEY
        For Azure: api_version, azure_endpoint (or env AZURE_OPENAI_API_VERSION, AZURE_OPENAI_ENDPOINT)
    """
    llm_config = config.get("llm", {}) or {}
    provider = (llm_config.get("provider") or "azure_openai").strip().lower()
    common = {
        "api_key": llm_config.get("api_key"),
        "max_tokens": llm_config.get("max_tokens"),
        "thinking": llm_config.get("thinking"),
        "effort": llm_config.get("effort"),
        "thinking_display": llm_config.get("thinking_display"),
        "max_retries": llm_config.get("max_retries", 8),
    }

    if provider == "openai":
        model = llm_config.get("model") or os.environ.get("OPENAI_MODEL", "gpt-4o")
        client = LLMClient(provider="openai", model=model, **common)
    elif provider == "azure_openai":
        model = llm_config.get("model") or os.environ.get("AZURE_OPENAI_DEPLOYMENT", "o4-mini-0416")
        client = LLMClient(
            provider="azure_openai", model=model,
            api_version=llm_config.get("api_version"),
            azure_endpoint=llm_config.get("azure_endpoint"),
            **common,
        )
    elif provider == "anthropic":
        model = llm_config.get("model", "claude-3-5-sonnet-20241022")
        client = LLMClient(provider="anthropic", model=model, **common)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}. Use one of: openai, azure_openai, anthropic")

    if llm_config.get("retry_total_seconds") is not None:
        client.retry_total_seconds = float(llm_config["retry_total_seconds"])
    if llm_config.get("max_images_per_request") is not None:
        client.max_images_per_request = int(llm_config["max_images_per_request"])
    if llm_config.get("prompt_caching") is not None:
        client.prompt_caching = bool(llm_config["prompt_caching"])
    if llm_config.get("cache_keepalive_minutes") is not None:
        client.cache_keepalive_minutes = float(llm_config["cache_keepalive_minutes"])

    return client, model
