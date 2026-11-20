import base64
import importlib.util
import json
import pickle
import zlib
from pathlib import Path

import pytest

from sia.bench import sandbox, tasks
from sia.bench.models import load_models
from sia.bench.tasks import (
    INSTRUCTION,
    LCB_WITH_STARTER,
    LCB_WITHOUT_STARTER,
    Task,
    _release_files,
    lcb_prompt,
    load_lcb,
    select_tasks,
)

REPO = Path(__file__).resolve().parents[1]
ANSWER_LINE = "### Answer: (use the provided format with backticks)\n\n"


# --- _release_files ---

@pytest.mark.parametrize("release, expected", [
    ("release_v6", {1, 2, 3, 4, 5, 6}),
    ("release_v1", {1}),
    ("v6", {6}),
    ("v4_v6", {4, 5, 6}),
])
def test_release_files(release, expected):
    assert _release_files(release) == expected


# --- lcb_prompt ---

def test_lcb_prompt_with_starter():
    starter = "class Solution:\n    def f(self, x: int) -> int:\n        "
    p = lcb_prompt({"question_content": "Return x.", "starter_code": starter})
    assert p.startswith("### Question:\nReturn x.")
    assert f"```python\n{starter}\n```" in p
    assert LCB_WITH_STARTER in p
    assert LCB_WITHOUT_STARTER not in p
    assert p.endswith(ANSWER_LINE)


def test_lcb_prompt_without_starter():
    p = lcb_prompt({"question_content": "Read N.", "starter_code": ""})
    assert "### Question:\nRead N." in p
    assert LCB_WITHOUT_STARTER in p
    assert "```python\n# YOUR CODE HERE\n```" in p
    assert LCB_WITH_STARTER not in p
    assert p.endswith(ANSWER_LINE)


# --- load_lcb ---

ROWS = [
    # (question_id, platform, contest_date, difficulty, func_name, release_file)
    ("abc300_a", "atcoder", "2023-05-01T00:00:00", "easy", None, 1),
    ("1873_A", "codeforces", "2023-08-21T00:00:00", "easy", None, 2),
    ("2800", "leetcode", "2023-10-15T00:00:00", "medium", "minLength", 3),
    ("abc350_c", "atcoder", "2024-04-20T00:00:00", "medium", None, 4),
    ("3200", "leetcode", "2024-06-01T00:00:00", "hard", "maxScore", 5),
    ("arc180_b", "atcoder", "2024-08-03T00:00:00", "hard", None, 5),
    ("3500", "leetcode", "2024-08-03T12:30:00", "easy", "countPairs", 6),
    ("abc390_d", "atcoder", "2025-02-01T00:00:00", "medium", None, 6),
]


def _row(qid, platform, date, difficulty, func_name, release_file):
    return {
        "question_id": qid,
        "question_title": f"Title {qid}",
        "question_content": f"Content of {qid}",
        "platform": platform,
        "contest_id": f"c-{qid}",
        "contest_date": date,
        "starter_code": f"class Solution:\n    def {func_name}(self):\n        " if func_name else "",
        "difficulty": difficulty,
        "func_name": func_name,
        "release_file": release_file,
        "n_public_tests": 2,
        "n_tests": 10,
    }


@pytest.fixture
def meta(tmp_path):
    path = tmp_path / "meta.jsonl"
    path.write_text("".join(json.dumps(_row(*r)) + "\n" for r in ROWS))
    return path


def lcb_ids(ts):
    return [t.task_id for t in ts]


def test_load_lcb_all(meta):
    ts = load_lcb(meta_path=meta)
    assert len(ts) == len(ROWS)
    assert lcb_ids(ts) == sorted(f"lcb/{r[0]}" for r in ROWS)
    assert all(t.dataset == "livecodebench" for t in ts)


def test_load_lcb_fields(meta):
    by_id = {t.task_id: t for t in load_lcb(meta_path=meta)}
    t = by_id["lcb/3500"]
    assert t.entry_point == "countPairs"
    assert t.meta == {"difficulty": "easy", "platform": "leetcode",
                      "contest_date": "2024-08-03", "release_file": 6}
    assert t.prompt == lcb_prompt(_row(*ROWS[6]))
    stdin = by_id["lcb/abc350_c"]
    assert stdin.entry_point is None
    assert LCB_WITHOUT_STARTER in stdin.prompt


@pytest.mark.parametrize("release, expected", [
    ("release_v1", {"abc300_a"}),
    ("release_v2", {"abc300_a", "1873_A"}),
    ("v5", {"3200", "arc180_b"}),
    ("v5_v6", {"3200", "arc180_b", "3500", "abc390_d"}),
])
def test_load_lcb_release_filter(meta, release, expected):
    assert set(lcb_ids(load_lcb(release, meta_path=meta))) == {f"lcb/{q}" for q in expected}


def test_load_lcb_date_filter_inclusive(meta):
    ts = load_lcb(start_date="2024-04-20", end_date="2024-08-03", meta_path=meta)
    # end_date equals the date part of a datetime with a nonzero time: still included.
    assert set(lcb_ids(ts)) == {"lcb/abc350_c", "lcb/3200", "lcb/arc180_b", "lcb/3500"}
    assert set(lcb_ids(load_lcb(start_date="2024-08-04", meta_path=meta))) == {"lcb/abc390_d"}
    assert set(lcb_ids(load_lcb(end_date="2023-05-01", meta_path=meta))) == {"lcb/abc300_a"}


def test_load_lcb_platform_and_difficulty(meta):
    assert set(lcb_ids(load_lcb(platform="leetcode", meta_path=meta))) == {"lcb/2800", "lcb/3200", "lcb/3500"}
    assert set(lcb_ids(load_lcb(difficulty="hard", meta_path=meta))) == {"lcb/3200", "lcb/arc180_b"}
    ts = load_lcb(platform="atcoder", difficulty="medium", meta_path=meta)
    assert lcb_ids(ts) == ["lcb/abc350_c", "lcb/abc390_d"]


def test_load_lcb_missing_meta(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_lcb(meta_path=tmp_path / "nope.jsonl")


# --- Task ---

def test_user_message_livecodebench_unchanged():
    prompt = lcb_prompt({"question_content": "Q", "starter_code": ""})
    t = Task(task_id="lcb/1", dataset="livecodebench", prompt=prompt, entry_point=None)
    assert t.user_message() == prompt


def test_user_message_humaneval_wrapped():
    t = Task(task_id="HumanEval/0", dataset="humaneval", prompt="def f():\n    pass\n", entry_point="f")
    assert t.user_message() == f"{INSTRUCTION}\n```python\ndef f():\n    pass\n```"


def test_task_meta_not_part_of_identity():
    a = Task("lcb/1", "livecodebench", "p", None, meta={"difficulty": "easy"})
    b = Task("lcb/1", "livecodebench", "p", None, meta={"difficulty": "hard", "platform": "x"})
    assert a == b
    assert hash(a) == hash(b)
    assert len({a, b}) == 1


# --- select_tasks sampling ---

def _synthetic(n=50):
    return [Task(f"HumanEval/{i}", "humaneval", f"def f{i}(): pass\n", f"f{i}") for i in range(n)]


@pytest.fixture
def fake_load(monkeypatch):
    monkeypatch.setattr(tasks, "load_tasks", lambda dataset, **kw: _synthetic())


def test_select_sample_deterministic(fake_load):
    a = lcb_ids(select_tasks("humaneval", seed=3, sample=10))
    assert len(a) == 10
    assert a == lcb_ids(select_tasks("humaneval", seed=3, sample=10))


def test_select_sample_seed_changes_sample(fake_load):
    assert lcb_ids(select_tasks("humaneval", seed=0, sample=10)) != lcb_ids(select_tasks("humaneval", seed=1, sample=10))


def test_select_sample_sorted(fake_load):
    ts = select_tasks("humaneval", seed=5, sample=15)
    assert ts == sorted(ts, key=lambda t: tasks._task_sort_key(t.task_id))


@pytest.mark.parametrize("sample", [50, 51, 1000])
def test_select_sample_at_least_len_returns_all(fake_load, sample):
    assert lcb_ids(select_tasks("humaneval", seed=0, sample=sample)) == lcb_ids(_synthetic())


# --- models ---

def test_model_concurrency():
    models = load_models()
    local = [m for name, m in models.items() if name.startswith("qwen")]
    assert local
    assert all(m.concurrency == 1 for m in local)
    assert models["frontier"].concurrency == 4


# --- sandbox args ---

def test_sandbox_args_mounts(monkeypatch, tmp_path):
    monkeypatch.setattr(sandbox, "_engine", lambda: "podman")
    args = sandbox.sandbox_args(tmp_path / "samples.jsonl", tmp_path / "out", memory="6g")
    assert args[0] == "podman"
    assert any(sandbox.LCB_VOLUME in a and a.endswith(":ro") for a in args)
    mem = args[args.index("--memory") + 1]
    assert mem == "6g"
    assert args[args.index("--memory-swap") + 1] == mem
    assert args[args.index("--network") + 1] == "none"


# --- sandbox/lcb_worker.py helpers ---

@pytest.fixture
def lcb_worker(monkeypatch):
    sandbox_dir = REPO / "sandbox"
    monkeypatch.syspath_prepend(str(sandbox_dir))
    spec = importlib.util.spec_from_file_location("lcb_worker", sandbox_dir / "lcb_worker.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_safe_name(lcb_worker):
    assert lcb_worker.safe_name("abc300_a") == "abc300_a"
    assert lcb_worker.safe_name("1873_A.v-2") == "1873_A.v-2"
    assert lcb_worker.safe_name("a/b c:d*?") == "a_b_c_d__"
    assert lcb_worker.safe_name("../x") == ".._x"


def test_decode_private_plain_json(lcb_worker):
    tests = [{"input": "1\n", "output": "2\n", "testtype": "stdin"}]
    assert lcb_worker._decode_private(json.dumps(tests)) == tests


def test_decode_private_compressed(lcb_worker):
    tests = [{"input": "[1, 2]", "output": "3", "testtype": "functional"}]
    encoded = base64.b64encode(zlib.compress(pickle.dumps(json.dumps(tests)))).decode("utf-8")
    assert lcb_worker._decode_private(encoded) == tests
