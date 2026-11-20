import random

import pytest

from sia.bench.tasks import INSTRUCTION, SPLIT_FRACTIONS, Task, _task_sort_key, split_tasks


def make_tasks(n=164):
    return [
        Task(task_id=f"HumanEval/{i}", dataset="humaneval", prompt=f"def f{i}():\n    pass\n", entry_point=f"f{i}")
        for i in range(n)
    ]


def ids(split):
    return {k: [t.task_id for t in v] for k, v in split.items()}


def test_deterministic_for_same_seed():
    assert ids(split_tasks(make_tasks(), 0)) == ids(split_tasks(make_tasks(), 0))


def test_different_seed_differs():
    assert ids(split_tasks(make_tasks(), 0)) != ids(split_tasks(make_tasks(), 1))


@pytest.mark.parametrize("n", [164, 378, 10, 7, 1])
def test_disjoint_and_complete(n):
    tasks = make_tasks(n)
    s = ids(split_tasks(tasks, 42))
    train, val, test = set(s["train"]), set(s["val"]), set(s["test"])
    assert not (train & val) and not (train & test) and not (val & test)
    assert train | val | test == {t.task_id for t in tasks}
    assert sum(len(v) for v in s.values()) == n


@pytest.mark.parametrize("n", [164, 378, 10, 7])
def test_sizes_match_fractions(n):
    s = split_tasks(make_tasks(n), 0)
    n_train = round(n * SPLIT_FRACTIONS["train"])
    n_val = round(n * SPLIT_FRACTIONS["val"])
    assert len(s["train"]) == n_train
    assert len(s["val"]) == n_val
    assert len(s["test"]) == n - n_train - n_val
    if n == 164:
        assert (len(s["train"]), len(s["val"]), len(s["test"])) == (82, 41, 41)


def test_independent_of_input_order():
    tasks = make_tasks()
    shuffled = tasks[:]
    random.Random(123).shuffle(shuffled)
    assert ids(split_tasks(tasks, 7)) == ids(split_tasks(shuffled, 7))
    assert ids(split_tasks(tasks, 7)) == ids(split_tasks(list(reversed(tasks)), 7))


def test_task_sort_key_numeric():
    assert _task_sort_key("HumanEval/9") < _task_sort_key("HumanEval/10")
    assert sorted(["HumanEval/10", "HumanEval/9", "HumanEval/100"], key=_task_sort_key) == [
        "HumanEval/9", "HumanEval/10", "HumanEval/100",
    ]


def test_user_message_wraps_prompt_in_python_block():
    t = Task(task_id="HumanEval/0", dataset="humaneval", prompt="def add(a, b):\n    \"\"\"Add.\"\"\"\n", entry_point="add")
    msg = t.user_message()
    assert msg.startswith(INSTRUCTION)
    assert "```python\ndef add(a, b):\n    \"\"\"Add.\"\"\"\n```" in msg
