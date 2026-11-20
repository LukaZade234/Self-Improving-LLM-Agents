"""Benchmark tasks and deterministic train/val/test splits."""

from __future__ import annotations

import random
from dataclasses import dataclass

DATASETS = ("humaneval", "mbpp")

# EvalPlus's own instruction prefix for instruct models, which keeps our numbers
# comparable with published EvalPlus results.
INSTRUCTION = (
    "Please provide a self-contained Python script that solves the following "
    "problem in a markdown code block:"
)


@dataclass(frozen=True)
class Task:
    task_id: str
    dataset: str
    prompt: str
    entry_point: str

    def user_message(self) -> str:
        return f"{INSTRUCTION}\n```python\n{self.prompt.strip()}\n```"


def load_tasks(dataset: str) -> list[Task]:
    """Load all tasks for a dataset, sorted by task_id."""
    from evalplus.data import get_human_eval_plus, get_mbpp_plus

    if dataset == "humaneval":
        problems = get_human_eval_plus()
    elif dataset == "mbpp":
        problems = get_mbpp_plus()
    else:
        raise ValueError(f"unknown dataset {dataset!r}, expected one of {DATASETS}")
    return [
        Task(task_id=tid, dataset=dataset, prompt=p["prompt"], entry_point=p["entry_point"])
        for tid, p in sorted(problems.items(), key=lambda kv: _task_sort_key(kv[0]))
    ]


def _task_sort_key(task_id: str) -> tuple[str, int]:
    prefix, _, num = task_id.rpartition("/")
    return (prefix, int(num)) if num.isdigit() else (task_id, -1)


SPLIT_FRACTIONS = {"train": 0.5, "val": 0.25, "test": 0.25}


def split_tasks(tasks: list[Task], seed: int) -> dict[str, list[Task]]:
    """Seeded split into train/val/test. Same tasks + seed always yields the same split."""
    ordered = sorted(tasks, key=lambda t: _task_sort_key(t.task_id))
    rng = random.Random(seed)
    rng.shuffle(ordered)
    n = len(ordered)
    n_train = round(n * SPLIT_FRACTIONS["train"])
    n_val = round(n * SPLIT_FRACTIONS["val"])
    return {
        "train": ordered[:n_train],
        "val": ordered[n_train : n_train + n_val],
        "test": ordered[n_train + n_val :],
    }


def select_tasks(
    dataset: str, split: str = "all", seed: int = 0, limit: int | None = None
) -> list[Task]:
    tasks = load_tasks(dataset)
    if split != "all":
        tasks = split_tasks(tasks, seed)[split]
    return tasks[:limit] if limit else tasks
