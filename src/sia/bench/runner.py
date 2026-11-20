"""`evaluate()`: generate solutions for tasks, score them in the sandbox, record everything.

This is the function the self-improvement loop will call as
`evaluate(prompt, val_tasks)`. Generations are cached on disk keyed by
(model identity, system prompt, task prompt, seed), so re-scoring a prompt that
was seen before (e.g. after a rollback) costs no LLM calls.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from . import sandbox
from .extract import extract_code
from .models import ModelConfig, complete
from .tasks import Task

REPO = Path(__file__).resolve().parents[3]
RUNS_DIR = REPO / "runs"
CACHE_DIR = RUNS_DIR / "_cache" / "generations"
RESULTS_CSV = REPO / "results" / "baselines.csv"


@dataclass
class TaskRecord:
    task_id: str
    passed_base: bool
    passed_plus: bool
    base_status: str | None
    plus_status: str | None
    base_fail_input: str | None
    plus_fail_input: str | None
    response: str
    solution: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    cost_usd: float
    finish_reason: str | None
    cached: bool


@dataclass
class RunResult:
    run_dir: Path
    summary: dict
    records: list[TaskRecord]

    @property
    def failures(self) -> list[TaskRecord]:
        return [r for r in self.records if not r.passed_plus]


def sha(text: str, n: int = 12) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:n]


def cache_key(model: ModelConfig, system_prompt: str, task: Task, seed: int) -> str:
    payload = json.dumps(
        {"model": model.identity(), "system": system_prompt, "user": task.user_message(), "seed": seed},
        sort_keys=True,
    )
    return sha(payload, 32)


def _generate(model: ModelConfig, system_prompt: str, task: Task, seed: int) -> tuple[dict, bool]:
    path = CACHE_DIR / f"{cache_key(model, system_prompt, task, seed)}.json"
    if path.exists():
        return json.loads(path.read_text()), True
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task.user_message()},
    ]
    c = complete(model, messages, seed=seed)
    gen = asdict(c)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(gen))
    tmp.replace(path)
    return gen, False


def _git_sha() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO,
                               capture_output=True, text=True).stdout.strip()
        return out + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def evaluate(
    model: ModelConfig,
    system_prompt: str,
    tasks: list[Task],
    *,
    seed: int = 0,
    workers: int = 4,
    run_dir: Path | None = None,
    meta: dict | None = None,
    progress=None,
) -> RunResult:
    if not tasks:
        raise ValueError("no tasks to evaluate")
    datasets = {t.dataset for t in tasks}
    if len(datasets) != 1:
        raise ValueError(f"tasks span several datasets: {datasets}")
    dataset = datasets.pop()

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = run_dir or RUNS_DIR / f"{stamp}_{model.name}_{dataset}"
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "timestamp": stamp,
        "model_name": model.name,
        "tier": model.tier,
        **model.identity(),
        "dataset": dataset,
        "n_tasks": len(tasks),
        "task_ids": [t.task_id for t in tasks],
        "seed": seed,
        "system_prompt": system_prompt,
        "prompt_sha": sha(system_prompt),
        "git_sha": _git_sha(),
        **(meta or {}),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=1))

    # 1. Generate (cached, parallel).
    t0 = time.perf_counter()
    gens: dict[str, tuple[dict, bool]] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_generate, model, system_prompt, t, seed): t for t in tasks}
        for fut, t in futs.items():
            gens[t.task_id] = fut.result()
            if progress:
                progress(t.task_id)
    gen_s = time.perf_counter() - t0

    # 2. Extract and write samples in EvalPlus format.
    solutions = {t.task_id: extract_code(gens[t.task_id][0]["text"], t.entry_point) for t in tasks}
    samples = run_dir / "samples.jsonl"
    with samples.open("w") as f:
        for t in tasks:
            f.write(json.dumps({"task_id": t.task_id, "solution": solutions[t.task_id]}) + "\n")

    # 3. Score in the sandbox.
    t1 = time.perf_counter()
    scored = sandbox.score(dataset, samples, run_dir)
    eval_s = time.perf_counter() - t1

    # 4. Assemble per-task records.
    records = []
    for t in tasks:
        g, cached = gens[t.task_id]
        s = scored.get(t.task_id, {})
        base, plus = s.get("base_status"), s.get("plus_status")
        records.append(TaskRecord(
            task_id=t.task_id,
            passed_base=base == "pass",
            passed_plus=base == "pass" and plus == "pass",
            base_status=base, plus_status=plus,
            base_fail_input=s.get("base_fail_input"), plus_fail_input=s.get("plus_fail_input"),
            response=g["text"], solution=solutions[t.task_id],
            prompt_tokens=g["prompt_tokens"], completion_tokens=g["completion_tokens"],
            latency_s=g["latency_s"], cost_usd=g["cost_usd"], finish_reason=g["finish_reason"],
            cached=cached,
        ))
    with (run_dir / "results.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(asdict(r)) + "\n")

    n = len(records)
    new_calls = [r for r in records if not r.cached]
    summary = {
        "model_name": model.name,
        "tier": model.tier,
        "dataset": dataset,
        "n_tasks": n,
        "seed": seed,
        "temperature": model.params.get("temperature"),
        "prompt_sha": config["prompt_sha"],
        "pass_base": sum(r.passed_base for r in records) / n,
        "pass_plus": sum(r.passed_plus for r in records) / n,
        "truncated": sum(r.finish_reason == "length" for r in records),
        "llm_calls": len(new_calls),
        "cached_calls": n - len(new_calls),
        "prompt_tokens": sum(r.prompt_tokens for r in records),
        "completion_tokens": sum(r.completion_tokens for r in records),
        "mean_completion_tokens": sum(r.completion_tokens for r in records) / n,
        "mean_latency_s": sum(r.latency_s for r in records) / n,
        "cost_usd": sum(r.cost_usd for r in new_calls),
        "gen_wall_s": round(gen_s, 2),
        "eval_wall_s": round(eval_s, 2),
        "git_sha": config["git_sha"],
        "run_dir": str(run_dir.relative_to(REPO)) if run_dir.is_relative_to(REPO) else str(run_dir),
        **(meta or {}),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    return RunResult(run_dir=run_dir, summary=summary, records=records)


CSV_FIELDS = [
    "timestamp", "tag", "model_name", "tier", "dataset", "split", "n_tasks", "seed",
    "temperature", "prompt_sha", "pass_base", "pass_plus", "truncated", "llm_calls",
    "completion_tokens", "mean_completion_tokens", "mean_latency_s", "cost_usd",
    "gen_wall_s", "eval_wall_s", "git_sha", "run_dir",
]


def append_results_csv(summary: dict, path: Path = RESULTS_CSV) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    row = {"timestamp": datetime.now().isoformat(timespec="seconds"), **summary}
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)
