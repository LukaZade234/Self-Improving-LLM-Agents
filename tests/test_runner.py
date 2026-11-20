from pathlib import Path

import pytest

from sia.bench import sandbox
from sia.bench.models import DEFAULT_CONFIG, ModelConfig, load_models
from sia.bench.runner import cache_key
from sia.bench.tasks import Task

MODEL = ModelConfig(name="m", model="ollama_chat/x", tier="tiny", params={"temperature": 0.0, "max_tokens": 2048})
TASK = Task(task_id="HumanEval/0", dataset="humaneval", prompt="def f():\n    pass\n", entry_point="f")


def test_cache_key_stable():
    a = cache_key(MODEL, "sys", TASK, 0)
    b = cache_key(ModelConfig(**{**MODEL.__dict__, "params": dict(MODEL.params)}), "sys", TASK, 0)
    assert a == b
    assert len(a) == 32


def test_cache_key_ignores_param_order():
    reordered = ModelConfig(name="m", model="ollama_chat/x", tier="tiny", params={"max_tokens": 2048, "temperature": 0.0})
    assert cache_key(MODEL, "sys", TASK, 0) == cache_key(reordered, "sys", TASK, 0)


def test_cache_key_changes_with_system_prompt():
    assert cache_key(MODEL, "sys", TASK, 0) != cache_key(MODEL, "sys2", TASK, 0)


def test_cache_key_changes_with_seed():
    assert cache_key(MODEL, "sys", TASK, 0) != cache_key(MODEL, "sys", TASK, 1)


def test_cache_key_changes_with_model_params():
    hot = ModelConfig(name="m", model="ollama_chat/x", tier="tiny", params={"temperature": 0.7, "max_tokens": 2048})
    assert cache_key(MODEL, "sys", TASK, 0) != cache_key(hot, "sys", TASK, 0)


def test_cache_key_changes_with_model_string():
    other = ModelConfig(name="m", model="ollama_chat/y", tier="tiny", params=dict(MODEL.params))
    assert cache_key(MODEL, "sys", TASK, 0) != cache_key(other, "sys", TASK, 0)


def test_cache_key_changes_with_task_prompt():
    t2 = Task(task_id="HumanEval/0", dataset="humaneval", prompt="def f():\n    return 1\n", entry_point="f")
    assert cache_key(MODEL, "sys", TASK, 0) != cache_key(MODEL, "sys", t2, 0)


def _args(monkeypatch, engine, tmp_path):
    monkeypatch.setattr(sandbox, "_engine", lambda: engine)
    return sandbox.sandbox_args(tmp_path / "samples.jsonl", tmp_path / "out")


def _pair(args, flag, value):
    return any(args[i] == flag and args[i + 1] == value for i in range(len(args) - 1))


def test_sandbox_args_podman(monkeypatch, tmp_path):
    args = _args(monkeypatch, "podman", tmp_path)
    assert args[0] == "podman"
    assert _pair(args, "--network", "none")
    assert "--read-only" in args
    assert _pair(args, "--userns", "keep-id")
    mounts = [args[i + 1] for i, a in enumerate(args) if a == "-v"]
    cache = [m for m in mounts if m.startswith(f"{sandbox.CACHE_VOLUME}:")]
    assert len(cache) == 1 and cache[0].endswith(":ro")
    samples = [m for m in mounts if ":/in/samples.jsonl:" in m]
    assert len(samples) == 1 and samples[0].endswith(":ro,Z")
    assert samples[0].startswith(str((tmp_path / "samples.jsonl").resolve()))
    assert args[-1] == sandbox.IMAGE


def test_sandbox_args_docker_no_keep_id(monkeypatch, tmp_path):
    args = _args(monkeypatch, "docker", tmp_path)
    assert args[0] == "docker"
    assert "keep-id" not in args
    assert "--userns" not in args
    assert _pair(args, "--network", "none")
    assert "--read-only" in args


def test_load_models_real_config():
    models = load_models(DEFAULT_CONFIG)
    assert "frontier" in models
    assert len(models) > 1
    for name, m in models.items():
        assert m.name == name
        assert m.params["temperature"] == 0.0
        assert m.params["max_tokens"] == 2048
        if name == "frontier":
            assert "think" not in m.params
        else:
            assert m.params["think"] is False


def test_load_models_overrides_defaults(tmp_path):
    p = tmp_path / "m.yaml"
    p.write_text(
        "defaults:\n  params: {temperature: 0.0, max_tokens: 100}\n"
        "models:\n  a:\n    model: x/a\n    params: {temperature: 0.5}\n  b:\n    model: x/b\n"
    )
    models = load_models(p)
    assert models["a"].params == {"temperature": 0.5, "max_tokens": 100}
    assert models["b"].params == {"temperature": 0.0, "max_tokens": 100}
    assert models["b"].tier == ""
