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
