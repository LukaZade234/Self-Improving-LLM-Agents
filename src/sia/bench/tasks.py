"""Benchmark tasks and deterministic train/val/test splits."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

DATASETS = ("humaneval", "mbpp", "livecodebench")
LCB_META = Path(__file__).resolve().parents[3] / "data" / "lcb" / "meta.jsonl"

# EvalPlus's own instruction prefix for instruct models, which keeps our numbers
# comparable with published EvalPlus results.
INSTRUCTION = (
    "Please provide a self-contained Python script that solves the following "
    "problem in a markdown code block:"
)

# LiveCodeBench's generic prompt (lcb_runner/prompts/code_generation.py,
# get_generic_question_template_answer), so our numbers are comparable to theirs.
LCB_WITH_STARTER = (
    "You will use the following starter code to write the solution to the problem "
    "and enclose your code within delimiters."
)
LCB_WITHOUT_STARTER = (
    "Read the inputs from stdin solve the problem and write the answer to stdout "
    "(do not directly test on the sample inputs). Enclose your code within delimiters "
    "as follows. Ensure that when the python program runs, it reads the inputs, runs "
    "the algorithm and writes output to STDOUT."
)


@dataclass(frozen=True)
class Task:
    task_id: str
    dataset: str
    prompt: str
    entry_point: str | None
    # Extra descriptive fields (difficulty, contest_date, platform...); not part of identity.
    meta: dict = field(default_factory=dict, compare=False, hash=False)

    def user_message(self) -> str:
        if self.dataset == "livecodebench":
            return self.prompt  # already the full LCB question template
        return f"{INSTRUCTION}\n```python\n{self.prompt.strip()}\n```"


def lcb_prompt(p: dict) -> str:
    prompt = f"### Question:\n{p['question_content']}\n\n"
    if p["starter_code"]:
        prompt += f"### Format: {LCB_WITH_STARTER}\n"
        prompt += f"```python\n{p['starter_code']}\n```\n\n"
    else:
        prompt += f"### Format: {LCB_WITHOUT_STARTER}\n"
        prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += "### Answer: (use the provided format with backticks)\n\n"
    return prompt


def _release_files(release: str) -> set[int]:
    """'release_v6' -> files 1..6 (cumulative); 'v6' -> file 6 only; 'v4_v6' -> 4..6."""
    if release.startswith("release_v"):
        return set(range(1, int(release.removeprefix("release_v")) + 1))
    if "_" in release:
        a, b = release.split("_")
        return set(range(int(a[1:]), int(b[1:]) + 1))
    return {int(release[1:])}


def load_lcb(
    release: str = "release_v6",
    start_date: str | None = None,
    end_date: str | None = None,
    platform: str | None = None,
    difficulty: str | None = None,
    meta_path: Path = LCB_META,
) -> list[Task]:
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} missing; run `bench download-lcb` and `bench prepare-lcb`")
    files = _release_files(release)
    tasks = []
    for line in meta_path.read_text().splitlines():
        p = json.loads(line)
        date = p["contest_date"][:10]
        if p["release_file"] not in files:
            continue
        if start_date and date < start_date:
            continue
        if end_date and date > end_date:
            continue
        if platform and p["platform"] != platform:
            continue
        if difficulty and p["difficulty"] != difficulty:
            continue
        tasks.append(Task(
            task_id=f"lcb/{p['question_id']}",
            dataset="livecodebench",
            prompt=lcb_prompt(p),
            entry_point=p["func_name"],
            meta={"difficulty": p["difficulty"], "platform": p["platform"],
                  "contest_date": date, "release_file": p["release_file"]},
        ))
    return sorted(tasks, key=lambda t: t.task_id)


def load_tasks(dataset: str, **lcb_filters) -> list[Task]:
    """Load all tasks for a dataset, sorted by task_id."""
    if dataset == "livecodebench":
        return load_lcb(**lcb_filters)

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
    dataset: str,
    split: str = "all",
    seed: int = 0,
    limit: int | None = None,
    sample: int | None = None,
    **lcb_filters,
) -> list[Task]:
    """Filter -> split -> optional seeded random sample -> optional first-N limit."""
    tasks = load_tasks(dataset, **lcb_filters)
    if split != "all":
        tasks = split_tasks(tasks, seed)[split]
    if sample is not None and sample < len(tasks):
        tasks = sorted(random.Random(seed).sample(tasks, sample), key=lambda t: _task_sort_key(t.task_id))
    return tasks[:limit] if limit else tasks
