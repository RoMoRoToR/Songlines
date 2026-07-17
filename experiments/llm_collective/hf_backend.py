"""
HuggingFace transformers backend, duck-type compatible with OllamaBackend
(complete(prompt, *, system, seed, max_tokens, stop) -> str; .summary()).

Greedy decoding (do_sample=False) for determinism; same on-disk cache scheme.
Intended for cluster GPU nodes (offline: export HF_HUB_OFFLINE=1 after weights
are pre-downloaded to $HF_HOME on shared scratch).
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class LLMCallStats:
    total_calls: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_latency_s: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)


class HFBackend:
    """Deterministic transformers client. One method: complete(...) -> str."""

    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-3B-Instruct",
        cache_dir: Optional[str] = "tmp/llm_cache_hf",
        temperature: float = 0.0,
        device: Optional[str] = None,
        dtype: str = "float16",
    ) -> None:
        self.model_name = model
        self.model = model  # summary-compat with OllamaBackend
        self.cache_dir = cache_dir
        self.temperature = float(temperature)
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
        self.stats = LLMCallStats()
        self._tok = None
        self._lm = None
        self._device = device
        self._dtype = dtype

    # --- lazy model load (so cache-only replays never touch the GPU) -----

    def _ensure_loaded(self):
        if self._lm is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dev = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        dt = {"float16": torch.float16, "bfloat16": torch.bfloat16,
              "float32": torch.float32}[self._dtype]
        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        try:  # transformers >=4.56 / v5 use dtype=; older use torch_dtype=
            self._lm = AutoModelForCausalLM.from_pretrained(
                self.model_name, dtype=dt, device_map=dev)
        except TypeError:
            self._lm = AutoModelForCausalLM.from_pretrained(
                self.model_name, torch_dtype=dt, device_map=dev)
        self._lm.eval()
        self._device = dev

    # --- public ----------------------------------------------------------

    def complete(
        self,
        prompt: str,
        *,
        system: str = "",
        seed: int = 0,
        max_tokens: int = 256,
        stop: Optional[list] = None,
    ) -> str:
        key = self._cache_key(system, prompt, seed, max_tokens, stop)
        cached = self._cache_get(key)
        if cached is not None:
            self.stats.cache_hits += 1
            return cached

        t0 = time.time()
        try:
            import torch
            self._ensure_loaded()
            msgs = []
            if system:
                msgs.append({"role": "system", "content": system})
            msgs.append({"role": "user", "content": prompt})
            text = self._tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
            inputs = self._tok(text, return_tensors="pt").to(self._device)
            with torch.no_grad():
                out_ids = self._lm.generate(
                    **inputs,
                    max_new_tokens=int(max_tokens),
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    pad_token_id=self._tok.eos_token_id,
                )
            out = self._tok.decode(
                out_ids[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True).strip()
            if stop:
                for s in stop:
                    idx = out.find(s)
                    if idx >= 0:
                        out = out[:idx].strip()
            if not out:
                out = "[LLM_EMPTY_RESPONSE]"
        except Exception as e:  # mirror OllamaBackend error contract
            out = f"[LLM_ERROR] {type(e).__name__}: {e}"

        self.stats.cache_misses += 1
        self.stats.total_calls += 1
        self.stats.total_latency_s += time.time() - t0
        self._cache_put(key, out)
        return out

    def summary(self) -> Dict[str, Any]:
        n = max(1, self.stats.total_calls)
        return {
            "model": self.model_name,
            "backend": "hf",
            "cache_hits": self.stats.cache_hits,
            "cache_misses": self.stats.cache_misses,
            "total_calls": self.stats.total_calls,
            "mean_latency_s": self.stats.total_latency_s / n,
        }

    # --- cache (same scheme as OllamaBackend) -----------------------------

    def _cache_key(self, system, prompt, seed, max_tokens, stop) -> str:
        payload = json.dumps({
            "backend": "hf", "model": self.model_name, "system": system,
            "prompt": prompt, "seed": seed, "max_tokens": max_tokens,
            "stop": stop or [], "temperature": self.temperature,
        }, sort_keys=True, ensure_ascii=False)
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
