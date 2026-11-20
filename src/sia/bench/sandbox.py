"""Score samples inside a locked-down podman container."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

IMAGE = "localhost/sia-eval:0.1"
CACHE_VOLUME = "sia-evalplus-cache"
CACHE_MOUNT = "/home/runner/.cache/evalplus"
MEMORY = "8g"  # container RAM cap, swap disabled. MBPP+ scoring peaks at 6.1G with 4 workers (measured 2026-11-20)
WORKERS = 4  # parallel test processes inside the container
LCB_VOLUME = "sia-lcb-cache"
LCB_MOUNT = "/home/runner/lcb"
REPO = Path(__file__).resolve().parents[3]
SANDBOX_DIR = REPO / "sandbox"
LCB_DATA = REPO / "data" / "lcb"  # raw/ (HF download) and meta.jsonl (no tests)


def _engine() -> str:
    for name in ("podman", "docker"):
        if shutil.which(name):
            return name
    raise RuntimeError("neither podman nor docker is installed")


def build_image() -> None:
    subprocess.run([_engine(), "build", "-t", IMAGE, str(SANDBOX_DIR)], check=True)


def prepare() -> None:
    """Build the image and fill the dataset/ground-truth cache volume.

    This is the only step with network access; it downloads datasets and
    computes reference outputs inside the container, so ground-truth timings
    come from the same environment that later scores samples.
    """
    build_image()
    subprocess.run(
        [_engine(), "run", "--rm", "-v", f"{CACHE_VOLUME}:{CACHE_MOUNT}:U", IMAGE, "prepare"],
        check=True,
    )


def _userns(engine: str) -> list[str]:
    # Rootless podman maps container uid 1000 to a subuid that can't write
    # host dirs; keep-id maps it to the invoking user instead.
    return ["--userns", "keep-id"] if engine == "podman" else []


def prepare_lcb() -> None:
    """Decode LiveCodeBench into the cache volume (tests) and data/lcb/meta.jsonl.

    Private tests are pickled, so decoding happens in the container with no
    network; the raw files are mounted read-only. Download them first with
    `bench download-lcb`.
    """
    raw = LCB_DATA / "raw"
    if not (raw / "test6.jsonl").exists():
        raise FileNotFoundError(f"{raw} is missing; run `bench download-lcb` first")
    build_image()
    engine = _engine()
    subprocess.run(
        [engine, "run", "--rm", *_userns(engine), "--network", "none",
         "--memory", MEMORY, "--memory-swap", MEMORY,
         "-v", f"{LCB_VOLUME}:{LCB_MOUNT}:U",
         "-v", f"{raw.resolve()}:/src:ro,Z",
         "-v", f"{LCB_DATA.resolve()}:/out:rw,Z",
         IMAGE, "prepare-lcb", "/src", "/out/meta.jsonl"],
        check=True,
    )


def sandbox_args(samples: Path, out_dir: Path, memory: str = MEMORY) -> list[str]:
    engine = _engine()
    return [
        engine, "run", "--rm", *_userns(engine),
        "--network", "none",
        "--read-only", "--tmpfs", "/tmp:rw,size=512m",
        # memory-swap == memory disables swap: a runaway solution is OOM-killed
        # inside the container instead of pushing the host into swap (2026-11-20:
        # a 2B MBPP+ run swapped 1.2G and triggered host memory pressure).
        "--memory", memory, "--memory-swap", memory, "--cpus", "4", "--pids-limit", "1024",
        # EvalPlus's per-test limit (RLIMIT_AS, 4 GiB) is left at its default so
        # scores stay identical to the official scorer; lowering it to 1 GiB made
        # 4 canonical HumanEval+ solutions fail.
        "--security-opt", "no-new-privileges",
        "-v", f"{CACHE_VOLUME}:{CACHE_MOUNT}:ro",
        "-v", f"{LCB_VOLUME}:{LCB_MOUNT}:ro",
        "-v", f"{samples.resolve()}:/in/samples.jsonl:ro,Z",
        "-v", f"{out_dir.resolve()}:/out:rw,Z",
        IMAGE,
    ]


def score(dataset: str, samples: Path, out_dir: Path, *, workers: int = WORKERS,
          memory: str = MEMORY, timeout_s: int = 1800) -> dict:
    """Run the scorer on a samples.jsonl; returns {task_id: {base_status, plus_status, ...}}."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = sandbox_args(samples, out_dir, memory) + [
        "eval", dataset, "/in/samples.jsonl", "/out/eval.json", str(workers)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    if proc.returncode != 0:
        raise RuntimeError(f"sandbox failed ({proc.returncode}):\n{proc.stderr[-3000:]}")
    return json.loads((out_dir / "eval.json").read_text())
