from sia.bench.extract import extract_code


def test_python_fence():
    r = "Here you go:\n```python\ndef f(x):\n    return x\n```\nDone."
    assert extract_code(r, "f") == "def f(x):\n    return x"


def test_py_fence():
    r = "```py\ndef g():\n    return 1\n```"
    assert extract_code(r) == "def g():\n    return 1"


def test_bare_fence():
    r = "Answer:\n```\ndef h():\n    return 2\n```\n"
    assert extract_code(r) == "def h():\n    return 2"


def test_no_fences_returns_raw_text():
    r = "  def f():\n    return 3\n  "
    assert extract_code(r, "f") == "def f():\n    return 3"


def test_entry_point_block_preferred_over_longer_block():
    long_block = "\n".join(f"x{i} = {i}" for i in range(50))
    r = (
        f"```python\n{long_block}\n```\n"
        "Solution:\n```python\ndef solve(a):\n    return a\n```\n"
    )
    assert extract_code(r, "solve") == "def solve(a):\n    return a"


def test_longest_python_block_without_entry_point():
    r = (
        "```python\nx = 1\n```\n"
        "```python\ndef longer():\n    return 'much longer body'\n```\n"
        "```bash\necho this is an even longer non-python block of text\n```\n"
    )
    assert extract_code(r) == "def longer():\n    return 'much longer body'"


def test_unterminated_fence_yields_code():
    r = "```python\ndef f(x):\n    y = x + 1\n    return"
    assert extract_code(r, "f") == "def f(x):\n    y = x + 1\n    return"


def test_think_block_stripped():
    r = "<think>```python\ndef f():\n    return 'wrong'\n```</think>\n```python\ndef f():\n    return 'right'\n```"
    assert extract_code(r, "f") == "def f():\n    return 'right'"


def test_unclosed_think_stripped():
    r = "```python\ndef f():\n    return 1\n```\n<think>still reasoning ```python\ndef f():\n    return 'a very long wrong answer'"
    assert extract_code(r, "f") == "def f():\n    return 1"


def test_only_unclosed_think_returns_empty():
    assert extract_code("<think>I am thinking forever", "f") == ""


def test_empty_and_none():
    assert extract_code("") == ""
    assert extract_code(None) == ""  # type: ignore[arg-type]
    assert extract_code("   \n ") == ""
