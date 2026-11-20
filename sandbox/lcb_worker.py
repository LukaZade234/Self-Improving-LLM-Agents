"""LiveCodeBench support. Runs INSIDE the sandbox container.

prepare: decode the raw HF jsonl files (private tests are base64 > zlib >
pickle, so they are only ever unpickled in here, with no network) into one
gzip'd test file per problem plus a test-free metadata file for the host.

eval: score solutions with LiveCodeBench's own `run_test`, using the same
per-problem process and global timeout as their compute_code_generation_metrics.
"""

import base64
import gzip
import json
import multiprocessing
import os
import pickle
import re
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.set_int_max_str_digits(50000)  # as in LiveCodeBench's evaluator

RAW_FILES = ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"]
TIMEOUT = 6  # LiveCodeBench default per-test timeout (seconds)


def safe_name(qid):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", qid)


def _decode_private(s):
    try:
        return json.loads(s)
    except Exception:
        return json.loads(pickle.loads(zlib.decompress(base64.b64decode(s.encode("utf-8")))))


def prepare(src_dir, cache_dir, meta_out):
    tests_dir = os.path.join(cache_dir, "tests")
    os.makedirs(tests_dir, exist_ok=True)
    seen = set()
    n = 0
    with open(meta_out, "w") as meta_f:
        for idx, fname in enumerate(RAW_FILES, start=1):
            with open(os.path.join(src_dir, fname)) as f:
                for line in f:
                    p = json.loads(line)
                    qid = p["question_id"]
                    if qid in seen:
                        raise SystemExit(f"duplicate question_id {qid}")
                    seen.add(qid)
                    public = json.loads(p["public_test_cases"])
                    private = _decode_private(p["private_test_cases"])
                    metadata = json.loads(p["metadata"] or "{}")
                    tests = public + private
                    io = {
                        "inputs": [t["input"] for t in tests],
                        "outputs": [t["output"] for t in tests],
                        "fn_name": metadata.get("func_name"),
                    }
                    with gzip.open(os.path.join(tests_dir, safe_name(qid) + ".json.gz"), "wt") as tf:
                        json.dump(io, tf)
                    meta_f.write(json.dumps({
                        "question_id": qid,
                        "question_title": p["question_title"],
                        "question_content": p["question_content"],
                        "platform": p["platform"],
                        "contest_id": p["contest_id"],
                        "contest_date": p["contest_date"],
                        "starter_code": p["starter_code"],
                        "difficulty": p["difficulty"],
                        "func_name": metadata.get("func_name"),
                        "release_file": idx,
                        "n_public_tests": len(public),
                        "n_tests": len(tests),
                    }) + "\n")
                    n += 1
            print(f"{fname}: {n} problems so far", file=sys.stderr, flush=True)
    print(f"prepared {n} LiveCodeBench problems")


# --- evaluation (mirrors lcb_runner/evaluation/compute_code_generation_metrics.py) ---

def _temp_run(sample, generation, debug, result, metadata_list, timeout):
    from lcb_testing_util import run_test

    res, metadata = run_test(sample, test=generation, debug=debug, timeout=timeout)
    result.append(res)
    metadata_list.append(metadata)


def check_correctness(sample, generation, timeout):
    manager = multiprocessing.Manager()
    result = manager.list()
    metadata_list = manager.list()
    p = multiprocessing.Process(
        target=_temp_run, args=(sample, generation, False, result, metadata_list, timeout)
    )
    p.start()
    p.join(timeout=(timeout + 1) * len(json.loads(sample["input_output"])["inputs"]) + 5)
    if p.is_alive():
        p.kill()
    if not result:
        n = len(json.loads(sample["input_output"])["inputs"])
        return [-1] * n, {"error_code": -1, "error_message": "global timeout"}
    return list(result[0]), dict(metadata_list[0])


def _score_one(task_id, solution, tests_path, timeout):
    with gzip.open(tests_path, "rt") as f:
        io = json.load(f)
    sample = {"input_output": json.dumps(io)}
    try:
        res, meta = check_correctness(sample, solution, timeout)
    except Exception as e:  # test-runner failure counts as a fail, as in LCB
        res, meta = [-5], {"error_code": -5, "error_message": f"TestRunnerError {e!r}"}
    res = [bool(x) if isinstance(x, bool) else int(x) for x in res]
    passed = len(res) > 0 and all(x is True or (not isinstance(x, bool) and x > 0) for x in res)
    status = "pass" if passed else "fail"
    fail_info = None
    if not passed:
        fail_info = json.dumps({k: meta.get(k) for k in ("error_code", "error_message", "inputs", "expected", "output") if k in meta})[:400]
    return task_id, {
        "base_status": status,
        "plus_status": status,  # LCB has a single test suite; mirror it so passed_plus works
        "base_fail_input": fail_info,
        "plus_fail_input": None,
        "n_tests_run": len(res),
    }


def evaluate(cache_dir, samples_path, out_path, workers=4):
    workers = int(workers)
    tests_dir = os.path.join(cache_dir, "tests")
    with open(samples_path) as f:
        samples = [json.loads(line) for line in f if line.strip()]
    results = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = []
        for s in samples:
            qid = s["task_id"].removeprefix("lcb/")
            path = os.path.join(tests_dir, safe_name(qid) + ".json.gz")
            if not os.path.exists(path):
                results[s["task_id"]] = {"base_status": "unknown_task", "plus_status": None}
                continue
            futs.append(ex.submit(_score_one, s["task_id"], s["solution"], path, TIMEOUT))
        for fut in as_completed(futs):
            tid, r = fut.result()
            results[tid] = r
    with open(out_path, "w") as f:
        json.dump(results, f, indent=1, sort_keys=True)
