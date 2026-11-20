"""Score samples inside a locked-down podman container."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

IMAGE = "localhost/sia-eval:0.1"
CACHE_VOLUME = "sia-evalplus-cache"
CACHE_MOUNT = "/home/runner/.cache/evalplus"
SANDBOX_DIR = Path(__file__).resolve().parents[3] / "sandbox"


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


def sandbox_args(samples: Path, out_dir: Path) -> list[str]:
    engine = _engine()
    # Rootless podman maps container uid 1000 to a subuid that can't write the
    # host's out_dir; keep-id maps it to the invoking user instead.
    userns = ["--userns", "keep-id"] if engine == "podman" else []
    return [
        engine, "run", "--rm", *userns,
        "--network", "none",
        "--read-only", "--tmpfs", "/tmp:rw,size=512m",
        "--memory", "4g", "--cpus", "4", "--pids-limit", "1024",
        "--security-opt", "no-new-privileges",
        "-v", f"{CACHE_VOLUME}:{CACHE_MOUNT}:ro",
        "-v", f"{samples.resolve()}:/in/samples.jsonl:ro,Z",
        "-v", f"{out_dir.resolve()}:/out:rw,Z",
        IMAGE,
    ]


def score(dataset: str, samples: Path, out_dir: Path, timeout_s: int = 1800) -> dict:
    """Run the scorer on a samples.jsonl; returns {task_id: {base_status, plus_status, ...}}."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = sandbox_args(samples, out_dir) + ["eval", dataset, "/in/samples.jsonl", "/out/eval.json"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    if proc.returncode != 0:
        raise RuntimeError(f"sandbox failed ({proc.returncode}):\n{proc.stderr[-3000:]}")
    return json.loads((out_dir / "eval.json").read_text())
