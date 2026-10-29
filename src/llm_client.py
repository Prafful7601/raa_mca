"""Async, provider-agnostic LLM client with disk caching (keyed by
provider+model+prompt hash), retry/exponential backoff, and bounded
concurrency. Adding a new model to config.yaml's models.available list
requires no code changes here as long as its `provider` is one of the
providers implemented below (currently `anthropic`, `gemini`)."""
from __future__ import annotations
import asyncio
import hashlib
import json
import os
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
load_dotenv(CONFIG_PATH.parent / ".env")


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _cache_key(provider: str, model: str, prompt: str, temperature: float, system: Optional[str], nonce: int) -> str:
    h = hashlib.sha256()
    h.update(provider.encode())
    h.update(model.encode())
    h.update(str(temperature).encode())
    h.update((system or "").encode())
    h.update(prompt.encode())
    h.update(str(nonce).encode())
    return h.hexdigest()


class _AnthropicBackend:
    def __init__(self, api_key: Optional[str]):
        import anthropic
        self._mod = anthropic
        self.client = anthropic.AsyncAnthropic(api_key=api_key) if api_key else None

    def available(self) -> bool:
        return self.client is not None

    @property
    def retryable_exceptions(self):
        a = self._mod
        return (a.RateLimitError, a.APIStatusError, a.APIConnectionError, asyncio.TimeoutError)

    async def call(self, model: str, prompt: str, system: str, temperature: float, max_tokens: int, timeout: float) -> dict:
        a = self._mod
        resp = await asyncio.wait_for(
            self.client.messages.create(
                model=model, max_tokens=max_tokens, temperature=temperature,
                system=system or a.NOT_GIVEN,
                messages=[{"role": "user", "content": prompt}],
            ),
            timeout=timeout,
        )
        text = "".join(b.text for b in resp.content if hasattr(b, "text"))
        return {"text": text, "input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}


class _GeminiBackend:
    def __init__(self, api_key: Optional[str]):
        from google import genai
        from google.genai import errors, types
        import httpx
        self._genai = genai
        self._types = types
        self._errors = errors
        self._httpx = httpx
        self.client = genai.Client(api_key=api_key) if api_key else None

    def available(self) -> bool:
        return self.client is not None

    @property
    def retryable_exceptions(self):
        e, h = self._errors, self._httpx
        # ClientError/ServerError cover HTTP-level failures (429, 5xx); the SDK's
        # own request layer can also raise raw httpx transport errors (dropped
        # connections, read timeouts) that never reach an HTTP response at all --
        # those must be retried too, or a single network blip kills the whole run.
        return (e.ClientError, e.ServerError, h.TransportError, asyncio.TimeoutError)

    async def call(self, model: str, prompt: str, system: str, temperature: float, max_tokens: int, timeout: float) -> dict:
        t = self._types
        config = t.GenerateContentConfig(
            system_instruction=system or None,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        resp = await asyncio.wait_for(
            self.client.aio.models.generate_content(model=model, contents=prompt, config=config),
            timeout=timeout,
        )
        text = resp.text or ""
        usage = resp.usage_metadata
        input_tokens = (usage.prompt_token_count or 0) if usage else 0
        output_tokens = (usage.candidates_token_count or 0) if usage else 0
        return {"text": text, "input_tokens": input_tokens, "output_tokens": output_tokens}


class _OllamaBackend:
    """Local inference via Ollama -- no API key, no external rate limit.
    Throughput is bounded only by local hardware."""
    def __init__(self, api_key: Optional[str], host: str = "http://localhost:11434"):
        import ollama
        self._ollama = ollama
        self.client = ollama.AsyncClient(host=host)
        self._host = host

    def available(self) -> bool:
        return self.client is not None

    @property
    def retryable_exceptions(self):
        o = self._ollama
        return (o.ResponseError, ConnectionError, asyncio.TimeoutError)

    async def call(self, model: str, prompt: str, system: str, temperature: float, max_tokens: int, timeout: float) -> dict:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = await asyncio.wait_for(
            self.client.chat(model=model, messages=messages,
                              options={"temperature": temperature, "num_predict": max_tokens},
                              # Ollama's default keep_alive (5 min idle) unloads a 4.7GB model
                              # between calls during a long batch run, forcing a slow reload on
                              # every request; keep it resident for the whole run instead.
                              keep_alive="60m"),
            timeout=timeout,
        )
        text = resp.message.content or ""
        return {"text": text, "input_tokens": resp.prompt_eval_count or 0, "output_tokens": resp.eval_count or 0}


_BACKENDS = {"anthropic": _AnthropicBackend, "gemini": _GeminiBackend, "ollama": _OllamaBackend}


class LLMClient:
    def __init__(self, config: Optional[dict] = None):
        self.config = config or load_config()
        self.cache_dir = Path(self.config["paths"]["cache_dir"])
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._model_provider = {m["name"]: m["provider"] for m in self.config["models"]["available"]}
        self._backends = {}
        for provider, pconf in self.config["providers"].items():
            api_key = os.environ.get(pconf["api_key_env"]) if "api_key_env" in pconf else None
            extra_kwargs = {k: v for k, v in pconf.items() if k != "api_key_env"}
            backend_cls = _BACKENDS.get(provider)
            if backend_cls is None:
                continue
            try:
                self._backends[provider] = backend_cls(api_key, **extra_kwargs)
            except ImportError:
                self._backends[provider] = None

        self.max_concurrency = self.config["llm"]["max_concurrency"]
        self.max_retries = self.config["llm"]["max_retries"]
        self.base_backoff = self.config["llm"]["base_backoff_seconds"]
        self.max_backoff = self.config["llm"]["max_backoff_seconds"]
        self.timeout = self.config["llm"]["request_timeout_seconds"]
        concurrency_overrides = self.config["llm"].get("max_concurrency_per_provider", {})
        self._sems = {
            provider: asyncio.Semaphore(concurrency_overrides.get(provider, self.max_concurrency))
            for provider in self._backends
        }
        self._timeout_overrides = self.config["llm"].get("request_timeout_seconds_per_provider", {})
        self._min_interval = self.config["llm"].get("min_interval_seconds", {})
        self._rate_locks = {provider: asyncio.Lock() for provider in self._backends}
        self._last_call_at = {provider: 0.0 for provider in self._backends}
        self.stats = {"calls": 0, "cache_hits": 0, "input_tokens": 0, "output_tokens": 0}

    async def _pace(self, provider: str):
        interval = self._min_interval.get(provider, 0.0)
        if interval <= 0:
            return
        lock = self._rate_locks[provider]
        async with lock:
            now = asyncio.get_event_loop().time()
            wait = self._last_call_at[provider] + interval - now
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_call_at[provider] = asyncio.get_event_loop().time()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _provider_for(self, model: str) -> str:
        provider = self._model_provider.get(model)
        if provider is None:
            raise ValueError(f"model '{model}' is not listed in config.yaml models.available")
        return provider

    def available(self, model: Optional[str] = None) -> bool:
        model = model or self.config["models"]["default"]
        provider = self._provider_for(model)
        backend = self._backends.get(provider)
        return backend is not None and backend.available()

    async def complete(self, model: str, prompt: str, system: str = "",
                        temperature: float = 0.0, max_tokens: int = 1024, nonce: int = 0) -> dict:
        """Returns {"text": str, "cached": bool, "usage": {...}}. Raises on
        unrecoverable API failure after exhausting retries.

        `nonce` distinguishes multiple independently-sampled calls that would
        otherwise share an identical cache key (same model/prompt/temperature/
        system) -- e.g. the K sampled checks in src/verifier.py's run_ours.
        Without it, K "independent" samples collapse to one cached response on
        any rerun, silently destroying the sampling diversity C2 depends on.
        Leave at 0 for calls that are meant to be a single deterministic/cached
        result (B1, B2, B3's greedy check)."""
        provider = self._provider_for(model)
        key = _cache_key(provider, model, prompt, temperature, system, nonce)
        cache_path = self._cache_path(key)
        if cache_path.exists():
            self.stats["cache_hits"] += 1
            return json.loads(cache_path.read_text())

        backend = self._backends.get(provider)
        if backend is None or not backend.available():
            pconf = self.config["providers"][provider]
            reason = f"{pconf['api_key_env']} not set" if "api_key_env" in pconf else "server unreachable"
            raise RuntimeError(f"provider '{provider}' unavailable: {reason} (or SDK not installed)")

        timeout = self._timeout_overrides.get(provider, self.timeout)
        async with self._sems[provider]:
            attempt = 0
            while True:
                try:
                    await self._pace(provider)
                    out = await backend.call(model, prompt, system, temperature, max_tokens, timeout)
                    result = {
                        "text": out["text"], "cached": False,
                        "usage": {"input_tokens": out["input_tokens"], "output_tokens": out["output_tokens"]},
                    }
                    self.stats["calls"] += 1
                    self.stats["input_tokens"] += out["input_tokens"]
                    self.stats["output_tokens"] += out["output_tokens"]
                    cache_path.write_text(json.dumps(result))
                    return result
                except backend.retryable_exceptions:
                    attempt += 1
                    if attempt > self.max_retries:
                        raise
                    backoff = min(self.max_backoff, self.base_backoff * (2 ** (attempt - 1)))
                    backoff *= 0.5 + os.urandom(1)[0] / 512.0  # jitter
                    await asyncio.sleep(backoff)

    def cost_estimate(self, model_pricing: dict) -> dict:
        inp = self.stats["input_tokens"] / 1_000_000 * model_pricing.get("input_per_mtok", 0)
        out = self.stats["output_tokens"] / 1_000_000 * model_pricing.get("output_per_mtok", 0)
        return {"input_cost": inp, "output_cost": out, "total_cost": inp + out}
