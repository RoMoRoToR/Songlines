"""LLM backend — thin ollama HTTP client with deterministic caching.

We keep the dependency surface minimal: only `requests` (already in the
project's environment). Caching is keyed on (model, system, prompt,
temperature, seed) so reruns are reproducible without re-charging
inference cost or wall-clock.
"""

from __future__ import annotations

import dataclasses as dc
import hashlib
import json
import os
import time
from typing import Any, Dict, Optional

import requests


DEFAULT_OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("LLM_COLLECTIVE_MODEL", "llama3.1:latest")
DEFAULT_CACHE_DIR = os.environ.get(
    "LLM_COLLECTIVE_CACHE",
    os.path.join(os.getcwd(), "tmp", "llm_bridge_minimal", ".cache_llm"),
)


@dc.dataclass
class LLMCallStats:
    cache_hits: int = 0
    cache_misses: int = 0
    total_latency_s: float = 0.0
    total_calls: int = 0


class OllamaBackend:
    """Deterministic ollama client. One method: complete(...) -> str."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        url: str = DEFAULT_OLLAMA_URL,
        cache_dir: Optional[str] = DEFAULT_CACHE_DIR,
        temperature: float = 0.0,
        timeout_s: float = 60.0,
        raw_prompt: Optional[bool] = None,
    ) -> None:
        self.model = model
        self.url = url.rstrip("/")
        self.cache_dir = cache_dir
        self.temperature = float(temperature)
        self.timeout_s = float(timeout_s)
        self.raw_prompt = self._prefers_raw_prompt(model) if raw_prompt is None else bool(raw_prompt)
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
        self.stats = LLMCallStats()

    # --- public ----------------------------------------------------------

    def complete(
        self,
        prompt: str,
        *,
        system: str = "",
        seed: int = 0,
        max_tokens: int = 256,
        stop: Optional[list[str]] = None,
    ) -> str:
        key = self._cache_key(system, prompt, seed, max_tokens, stop)
        cached = self._cache_get(key)
        if cached is not None:
            self.stats.cache_hits += 1
            return cached

        t0 = time.time()
        body: Dict[str, Any] = {
            "model": self.model,
            "prompt": self._compose_prompt(system, prompt),
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "seed": int(seed),
                "num_predict": int(max_tokens),
            },
        }
        if self.raw_prompt:
            body["raw"] = True
        else:
            body["system"] = system or ""
        if stop:
            body["options"]["stop"] = list(stop)
        try:
            r = requests.post(
                f"{self.url}/api/generate",
                json=body,
                timeout=self.timeout_s,
            )
            r.raise_for_status()
            data = r.json()
            out = (data.get("response") or "").strip()
            if not out:
                out = (
                    "[LLM_EMPTY_RESPONSE] "
                    f"done_reason={data.get('done_reason', '')} "
                    f"thinking_chars={len(data.get('thinking') or '')}"
                )
        except Exception as e:
            out = f"[LLM_ERROR] {type(e).__name__}: {e}"

        self.stats.cache_misses += 1
        self.stats.total_calls += 1
        self.stats.total_latency_s += time.time() - t0
        self._cache_put(key, out)
        return out

    def summary(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "cache_hits": self.stats.cache_hits,
            "cache_misses": self.stats.cache_misses,
            "total_calls": self.stats.total_calls,
            "mean_latency_s": (
                self.stats.total_latency_s / max(1, self.stats.total_calls)
            ),
            "raw_prompt": self.raw_prompt,
        }

    # --- caching ---------------------------------------------------------

    @staticmethod
    def _prefers_raw_prompt(model: str) -> bool:
        # Ollama's Qwen3 chat template can spend the whole budget in the
        # separate thinking field for short structured completions. Raw
        # completion keeps the same model but makes the first-line contract
        # observable to the parser.
        return model.lower().startswith("qwen3")

    def _compose_prompt(self, system: str, prompt: str) -> str:
        if not self.raw_prompt or not system:
            return prompt
        return system.strip() + "\n\n" + prompt

    def _cache_key(
        self,
        system: str,
        prompt: str,
        seed: int,
        max_tokens: int,
        stop: Optional[list[str]],
    ) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "system": system,
                "prompt": prompt,
                "seed": int(seed),
                "max_tokens": int(max_tokens),
                "stop": list(stop or []),
                "temperature": self.temperature,
                "raw_prompt": self.raw_prompt,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> Optional[str]:
        if not self.cache_dir:
            return None
        return os.path.join(self.cache_dir, key + ".txt")

    def _cache_get(self, key: str) -> Optional[str]:
        p = self._cache_path(key)
        if p and os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return f.read()
        return None

    def _cache_put(self, key: str, value: str) -> None:
        p = self._cache_path(key)
        if p:
            with open(p, "w", encoding="utf-8") as f:
                f.write(value)
