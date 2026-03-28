"""
llm_router.py — Reusable Free LLM Router

Rotates across multiple provider/model slots, tracks per-slot cooldowns,
and waits when all slots are rate-limited.

Config path resolution (in order):
1. Explicit path passed to constructor
2. ~/.llm_router_config.json
3. ./llm_router_config.json
"""

import json
import re
import time
import uuid
import logging
from datetime import datetime, timedelta
from pathlib import Path

import litellm
from litellm.exceptions import (
    RateLimitError,
    ServiceUnavailableError,
    APIConnectionError,
    AuthenticationError,
    BadRequestError,
    ContextWindowExceededError,
    NotFoundError,
)

logger = logging.getLogger(__name__)

# Suppress litellm's verbose logging
litellm.suppress_debug_info = True
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


class LLMRouter:
    """
    Routes LLM calls across multiple provider/model slots with:
    - Round-robin rotation
    - Per-slot cooldown tracking
    - Automatic retry with backoff
    - Permanent slot disabling on auth errors
    """

    def __init__(self, config_path=None):
        self._config = self._load_config(config_path)
        self._slots = self._expand_slots(self._config["providers"])
        self._slot_index = 0
        self._cooldown_until: dict[str, datetime] = {}
        self._disabled: set[str] = set()
        self._failure_count: dict[str, int] = {}  # for exponential backoff

        if not self._slots:
            raise ValueError("No provider slots found in config. Check your llm_router_config.json.")

        logger.info(f"LLMRouter initialized with {len(self._slots)} slots")

    def _load_config(self, config_path) -> dict:
        candidates = []
        if config_path:
            candidates.append(Path(config_path))
        candidates.append(Path.home() / ".llm_router_config.json")
        candidates.append(Path("llm_router_config.json"))

        for path in candidates:
            if path.exists():
                logger.info(f"Loading LLM router config from: {path}")
                with open(path) as f:
                    return json.load(f)

        raise FileNotFoundError(
            "No llm_router_config.json found. Checked:\n"
            + "\n".join(f"  {p}" for p in candidates)
            + "\n\nCopy llm_router_config.json.example to ~/.llm_router_config.json and fill in your API keys."
        )

    def _expand_slots(self, providers: list) -> list:
        """Expand each provider × model into a flat slot list.

        Supports "enabled": false at the provider level (skips all models)
        or per-model as {"model": "...", "enabled": false}.
        """
        slots = []
        for provider in providers:
            if not provider.get("enabled", True):
                logger.info(f"Provider {provider['name']} disabled via enabled:false")
                continue
            for entry in provider["models"]:
                if isinstance(entry, dict):
                    if not entry.get("enabled", True):
                        logger.info(f"Model {entry['model']} in {provider['name']} disabled via enabled:false")
                        continue
                    model = entry["model"]
                else:
                    model = entry
                slot = {
                    "name": f"{provider['name']}::{model}",
                    "type": provider["type"],
                    "api_key": provider["api_key"],
                    "model": model,
                }
                slots.append(slot)
        return slots

    def _slot_key(self, slot: dict) -> str:
        return slot["name"]

    def _is_available(self, slot: dict) -> bool:
        key = self._slot_key(slot)
        if key in self._disabled:
            return False
        cooldown = self._cooldown_until.get(key)
        if cooldown and datetime.now() < cooldown:
            return False
        return True

    def _next_available_slot(self) -> dict | None:
        """
        Find the next available slot using round-robin.
        Returns None if all slots are disabled.
        Returns the slot and updates the index.
        """
        n = len(self._slots)
        for _ in range(n):
            slot = self._slots[self._slot_index % n]
            self._slot_index = (self._slot_index + 1) % n
            if self._is_available(slot):
                return slot
        return None

    def _wait_for_slot(self) -> dict:
        """Wait until at least one slot becomes available, then return it."""
        while True:
            slot = self._next_available_slot()
            if slot:
                return slot

            # All slots cooling down — find soonest expiry
            active_cooldowns = [
                cd for key, cd in self._cooldown_until.items()
                if key not in self._disabled and datetime.now() < cd
            ]
            if not active_cooldowns:
                # All slots permanently disabled
                raise RuntimeError("All LLM slots are permanently disabled (auth errors). Check your API keys.")

            soonest = min(active_cooldowns)
            wait_secs = (soonest - datetime.now()).total_seconds() + 0.5
            print(f"  [router] All slots cooling down. Waiting {wait_secs:.1f}s...")
            time.sleep(max(0.5, wait_secs))

    def _set_cooldown(self, slot: dict, base_seconds: int, max_seconds: int = 600):
        """Set exponential backoff cooldown: base * 2^failures, capped at max_seconds."""
        key = self._slot_key(slot)
        failures = self._failure_count.get(key, 0)
        seconds = min(base_seconds * (2 ** failures), max_seconds)
        self._failure_count[key] = failures + 1
        self._cooldown_until[key] = datetime.now() + timedelta(seconds=seconds)
        logger.warning(f"Slot {key} on cooldown for {seconds}s (failure #{failures + 1})")
        print(f"    cooldown {seconds}s (attempt #{failures + 1})")

    def _disable_slot(self, slot: dict, reason: str):
        key = self._slot_key(slot)
        self._disabled.add(key)
        logger.error(f"Slot {key} permanently disabled: {reason}")
        print(f"  [router] Slot {slot['name']} disabled: {reason}")

    def _build_litellm_kwargs(self, slot: dict, **extra) -> tuple[str, dict]:
        """Return (model_string, kwargs_dict) for litellm call."""
        kwargs = dict(extra)
        provider_type = slot["type"]

        if provider_type == "openrouter":
            model_str = f"openrouter/{slot['model']}"
            kwargs["api_base"] = "https://openrouter.ai/api/v1"
            kwargs["api_key"] = slot["api_key"]
        elif provider_type == "google":
            model_str = slot["model"]  # already "gemini/gemini-2.0-flash-exp"
            kwargs["api_key"] = slot["api_key"]
        elif provider_type == "groq":
            model_str = slot["model"]  # already "groq/llama-3.3-70b-versatile"
            kwargs["api_key"] = slot["api_key"]
        else:
            # Generic fallback
            model_str = slot["model"]
            kwargs["api_key"] = slot["api_key"]

        return model_str, kwargs

    def _parse_xml_tool_calls(self, error_str: str, tools: list) -> object | None:
        """
        Some models (Kimi K2, older Qwen) emit XML-format tool calls instead of JSON:
            <function=read_file>journal.md</function>
        Groq rejects these with tool_use_failed before returning a response.
        This method extracts the failed_generation, parses the XML, and returns
        a synthetic response object that the harness can process normally.
        """
        # Extract failed_generation JSON blob from error string
        failed_gen = ""
        json_match = re.search(r'\{.*\}', error_str, re.DOTALL)
        if json_match:
            try:
                err_data = json.loads(json_match.group())
                failed_gen = err_data.get("error", {}).get("failed_generation", "")
            except (json.JSONDecodeError, AttributeError):
                pass

        if not failed_gen:
            return None

        # Parse <function=name>content</function> patterns
        calls = re.findall(r'<function=(\w+)>(.*?)</function>', failed_gen, re.DOTALL)
        if not calls:
            return None

        # Build map of tool_name → first required param (for plain-string args)
        first_param: dict[str, str | None] = {}
        for t in tools:
            fn = t["function"]
            req = fn["parameters"].get("required", [])
            first_param[fn["name"]] = req[0] if req else None

        # Build synthetic tool call objects
        class _Fn:
            def __init__(self, name, arguments):
                self.name = name
                self.arguments = arguments

        class _TC:
            def __init__(self, name, arguments):
                self.id = f"call_{uuid.uuid4().hex[:8]}"
                self.type = "function"
                self.function = _Fn(name, arguments)

        class _Msg:
            def __init__(self, tool_calls):
                self.content = ""
                self.tool_calls = tool_calls

        class _Choice:
            def __init__(self, message):
                self.message = message

        class _Response:
            def __init__(self, choices):
                self.choices = choices

        tool_calls = []
        for name, content in calls:
            content = content.strip()
            # Try to parse content as JSON args dict first
            try:
                args = json.loads(content)
            except json.JSONDecodeError:
                # Plain string — map to first required param
                param = first_param.get(name)
                args = {param: content} if param else {}
            tool_calls.append(_TC(name, json.dumps(args)))

        print(f"  [router] Parsed {len(tool_calls)} XML tool call(s): {[tc.function.name for tc in tool_calls]}")
        return _Response([_Choice(_Msg(tool_calls))])

    def _call_with_retry(self, build_kwargs_fn, tools=None) -> object:
        """
        Core retry loop with exponential backoff per slot.
        Exits only when: a call succeeds, a fatal error is raised, or all slots are permanently disabled.
        """
        while True:
            slot = self._wait_for_slot()  # raises if all slots permanently disabled
            model_str, kwargs = build_kwargs_fn(slot)

            try:
                logger.debug(f"Calling {model_str} (slot: {slot['name']})")
                response = litellm.completion(model=model_str, **kwargs)
                # Success — reset failure count for this slot
                self._failure_count.pop(self._slot_key(slot), None)
                return response

            except RateLimitError:
                print(f"  [router] Rate limit on {slot['name']}", end="")
                self._set_cooldown(slot, base_seconds=60, max_seconds=600)

            except ServiceUnavailableError:
                print(f"  [router] Service unavailable on {slot['name']}", end="")
                self._set_cooldown(slot, base_seconds=30, max_seconds=300)

            except APIConnectionError:
                print(f"  [router] Connection error on {slot['name']}", end="")
                self._set_cooldown(slot, base_seconds=10, max_seconds=120)

            except AuthenticationError as e:
                self._disable_slot(slot, f"AuthenticationError: {e}")

            except NotFoundError as e:
                err_str = str(e).lower()
                if "guardrail" in err_str or "data policy" in err_str or "privacy" in err_str:
                    self._disable_slot(slot, f"Account privacy restriction — fix at openrouter.ai/settings/privacy: {e}")
                elif "no endpoints" in err_str:
                    self._disable_slot(slot, f"No endpoints available: {e}")
                else:
                    print(f"  [router] Not found on {slot['name']}", end="")
                    self._set_cooldown(slot, base_seconds=60, max_seconds=600)

            except ContextWindowExceededError:
                raise  # propagate — harness handles trimming

            except BadRequestError as e:
                err_str = str(e).lower()
                if any(kw in err_str for kw in ("decommissioned", "not supported", "deprecated", "no longer")):
                    self._disable_slot(slot, f"Model decommissioned: {e}")
                elif "thought_signature" in err_str:
                    self._disable_slot(slot, f"Thinking model requires thought_signature (incompatible): {e}")
                elif "tool_use_failed" in err_str or "failed_generation" in err_str:
                    if tools:
                        synthetic = self._parse_xml_tool_calls(str(e), tools)
                        if synthetic:
                            return synthetic
                    print(f"  [router] Tool generation failed on {slot['name']}", end="")
                    self._set_cooldown(slot, base_seconds=15, max_seconds=120)
                else:
                    raise  # truly bad request — propagate

            except Exception as e:
                print(f"  [router] Unexpected error on {slot['name']}: {type(e).__name__}: {e}", end="")
                self._set_cooldown(slot, base_seconds=10, max_seconds=120)

    def chat_with_tools(self, messages: list, tools: list, tool_choice="auto", max_tokens=4096) -> object:
        """
        Call the LLM with tool/function calling support.
        Returns the raw litellm response.
        """
        def build_kwargs(slot):
            model_str, kwargs = self._build_litellm_kwargs(slot)
            kwargs["messages"] = messages
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice
            kwargs["max_tokens"] = max_tokens
            return model_str, kwargs

        return self._call_with_retry(build_kwargs, tools=tools)

    def chat(self, messages: list, max_tokens=4096) -> object:
        """
        Simple chat call without tools.
        Returns the raw litellm response.
        """
        def build_kwargs(slot):
            model_str, kwargs = self._build_litellm_kwargs(slot)
            kwargs["messages"] = messages
            kwargs["max_tokens"] = max_tokens
            return model_str, kwargs

        return self._call_with_retry(build_kwargs)
