# Self-Improving-LLM-Agents
Self Improving LLM Agents: Comparing Self Modification Strategies Using Watchdog Safeguards

## Setup

Requires [uv](https://docs.astral.sh/uv/), podman (or docker), and Ollama for local models.

```bash
uv sync                 # Python 3.12 env + deps
uv run bench prepare    # build sandbox image, cache datasets + ground truth (one-time, needs network)
uv run bench models     # list configured models (configs/models.yaml)
```

## Benchmark harness

```bash
uv run bench run -m qwen3.5-9b -d humaneval --limit 10 --tag smoke   # smoke test
uv run bench run -m qwen3.5-4b -d mbpp --split val --seed 0          # one split
uv run pytest                                                         # unit tests
```

- Generated code is scored by EvalPlus inside a container with no network, a read-only root FS,
  read-only datasets and memory/CPU/pid limits (`sandbox/`).
- Each run writes `runs/<time>_<model>_<dataset>/` (config, samples, per-task results, summary)
  and appends a row to `results/baselines.csv`.
- Generations are cached by (model, params, system prompt, task, seed), so repeat evaluations are free.
- The framework calls `sia.bench.runner.evaluate()` directly; the CLI is a thin wrapper.
