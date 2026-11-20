"""Runs INSIDE the sandbox container. Scores samples with EvalPlus's checker.

Kept self-contained (no `sia` imports) and baked into the image, so nothing the
agent writes can change how it is scored.

  python eval_worker.py prepare                         # cache datasets + ground truth
  python eval_worker.py eval <dataset> <samples.jsonl> <out.json>
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout

from evalplus.data import (
    get_human_eval_plus,
    get_human_eval_plus_hash,
    get_mbpp_plus,
    get_mbpp_plus_hash,
)
from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
from evalplus.evaluate import check_correctness, get_groundtruth


def load(dataset):
    with redirect_stdout(sys.stderr):
        if dataset == "humaneval":
            problems = get_human_eval_plus()
            gt = get_groundtruth(problems, get_human_eval_plus_hash(), [])
        elif dataset == "mbpp":
            problems = get_mbpp_plus()
            gt = get_groundtruth(problems, get_mbpp_plus_hash(), MBPP_OUTPUT_NOT_NONE_TASKS)
        else:
            raise SystemExit(f"unknown dataset {dataset}")
    return problems, gt


def short(x, n=300):
    s = repr(x)
    return s if len(s) <= n else s[:n] + "..."


def evaluate(dataset, samples_path, out_path, workers=4):
    problems, gt = load(dataset)
    with open(samples_path) as f:
        samples = [json.loads(line) for line in f if line.strip()]

    results = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {}
        for s in samples:
            tid = s["task_id"]
            if tid not in problems:
                results[tid] = {"base_status": "unknown_task", "plus_status": None}
                continue
            fut = ex.submit(
                check_correctness, dataset, 0, problems[tid], s["solution"], gt[tid],
                False, True, tid,
            )
            futures[fut] = tid
        for fut in as_completed(futures):
            tid = futures[fut]
            r = fut.result()
            base_stat, base_det = r["base"]
            plus_stat, plus_det = r["plus"]
            p = problems[tid]

            def first_fail(stat, details, inputs):
                if stat == "pass" or not details:
                    return None
                return short(inputs[len(details) - 1])

            results[tid] = {
                "base_status": base_stat,
                "plus_status": plus_stat,
                "base_fail_input": first_fail(base_stat, base_det, p["base_input"]),
                "plus_fail_input": first_fail(plus_stat, plus_det, p["plus_input"]),
            }
    with open(out_path, "w") as f:
        json.dump(results, f, indent=1, sort_keys=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "prepare":
        for ds in ("humaneval", "mbpp"):
            load(ds)
        print("prepared")
    elif cmd == "eval":
        evaluate(*sys.argv[2:5])
    else:
        raise SystemExit(__doc__)
