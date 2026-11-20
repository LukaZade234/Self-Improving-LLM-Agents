"""Model registry and a thin LiteLLM wrapper that records usage."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "models.yaml"


@dataclass(frozen=True)
class ModelConfig:
    name: str
    model: str
    tier: str = ""
    params: dict[str, Any] = field(default_factory=dict)

    def identity(self) -> dict[str, Any]:
        """Everything that affects outputs; used for cache keys and run configs."""
        return {"model": self.model, "params": dict(sorted(self.params.items()))}


@dataclass
class Completion:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    cost_usd: float
    finish_reason: str | None


def load_models(path: Path = DEFAULT_CONFIG) -> dict[str, ModelConfig]:
    cfg = yaml.safe_load(path.read_text())
    defaults = cfg.get("defaults", {}).get("params", {})
    return {
        name: ModelConfig(
            name=name,
            model=m["model"],
            tier=m.get("tier", ""),
            params={**defaults, **(m.get("params") or {})},
        )
        for name, m in cfg["models"].items()
    }


def get_model(name: str, path: Path = DEFAULT_CONFIG) -> ModelConfig:
    models = load_models(path)
    if name not in models:
        raise KeyError(f"unknown model {name!r}; known: {', '.join(models)}")
    return models[name]


def unload(cfg: ModelConfig, host: str = "http://localhost:11434") -> bool:
    """Ask Ollama to evict a local model now, freeing VRAM *and* host RAM.

    On ROCm the Ollama runner keeps an anonymous host-RAM copy of most of the
    weights (2.5 GB for the 3.1 GB 4B model, measured 2026-11-20), so a large
    model plus sandbox scoring can exhaust system RAM. Returns False for
    non-Ollama models or if the server is unreachable.
    """
    import json
    import urllib.request

    prefix, _, tag = cfg.model.partition("/")
    if prefix not in ("ollama", "ollama_chat"):
        return False
    req = urllib.request.Request(
        f"{host}/api/generate",
        data=json.dumps({"model": tag, "keep_alive": 0}).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        urllib.request.urlopen(req, timeout=60).read()
        return True
    except OSError:
        return False


def complete(cfg: ModelConfig, messages: list[dict[str, str]], **overrides: Any) -> Completion:
    import litellm

    litellm.suppress_debug_info = True
    params = {**cfg.params, **overrides}
    start = time.perf_counter()
    resp = litellm.completion(model=cfg.model, messages=messages, **params)
    latency = time.perf_counter() - start
    try:
        cost = litellm.completion_cost(resp) or 0.0
    except Exception:  # local models have no price entry
        cost = 0.0
    choice = resp.choices[0]
    return Completion(
        text=choice.message.content or "",
        prompt_tokens=resp.usage.prompt_tokens,
        completion_tokens=resp.usage.completion_tokens,
        latency_s=latency,
        cost_usd=cost,
        finish_reason=choice.finish_reason,
    )
