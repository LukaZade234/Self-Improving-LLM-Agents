"""`bench` command-line tool (standalone for now; the framework imports runner.evaluate)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from . import sandbox
from .models import get_model, load_models
from .runner import REPO, append_results_csv, evaluate
from .tasks import select_tasks

app = typer.Typer(help="Benchmark harness for self-improving LLM agents.", no_args_is_help=True)


@app.command()
def prepare() -> None:
    """Build the sandbox image and cache datasets + ground truth (needs network once)."""
    sandbox.prepare()


LCB_FILES = ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"]
LCB_REVISION = "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"  # HF dataset commit used for all runs


@app.command("download-lcb")
def download_lcb() -> None:
    """Download LiveCodeBench code_generation_lite (~4.5 GB) to data/lcb/raw at a pinned revision."""
    from huggingface_hub import hf_hub_download

    for f in LCB_FILES:
        hf_hub_download("livecodebench/code_generation_lite", f, repo_type="dataset",
                        revision=LCB_REVISION, local_dir=sandbox.LCB_DATA / "raw")
        typer.echo(f"downloaded {f}")


@app.command("prepare-lcb")
def prepare_lcb() -> None:
    """Decode LiveCodeBench tests inside the sandbox (no network) and write data/lcb/meta.jsonl."""
    sandbox.prepare_lcb()


@app.command()
def models() -> None:
    """List configured models."""
    for name, m in load_models().items():
        typer.echo(f"{name:18} {m.tier:9} {m.model}")


@app.command()
def run(
    model: str = typer.Option(..., "--model", "-m", help="Name from configs/models.yaml"),
    dataset: str = typer.Option("humaneval", "--dataset", "-d", help="humaneval | mbpp | livecodebench"),
    split: str = typer.Option("all", help="all | train | val | test"),
    limit: Optional[int] = typer.Option(None, help="Only the first N tasks (smoke tests)"),
    sample: Optional[int] = typer.Option(None, help="Seeded random sample of N tasks"),
    release: str = typer.Option("release_v6", help="LCB: release_vN (cumulative), vN, or vA_vB"),
    start_date: Optional[str] = typer.Option(None, help="LCB: earliest contest date YYYY-MM-DD"),
    end_date: Optional[str] = typer.Option(None, help="LCB: latest contest date YYYY-MM-DD"),
    platform: Optional[str] = typer.Option(None, help="LCB: leetcode | atcoder | codeforces"),
    difficulty: Optional[str] = typer.Option(None, help="LCB: easy | medium | hard"),
    prompt: Path = typer.Option(REPO / "prompts" / "base.md", help="System prompt file"),
    seed: int = typer.Option(0, help="Split seed and sampling seed"),
    temperature: Optional[float] = typer.Option(None, help="Override the configured temperature"),
    max_tokens: Optional[int] = typer.Option(None, help="Override the configured max_tokens"),
    param: list[str] = typer.Option([], help="Extra model param KEY=VALUE (JSON value), e.g. num_ctx=16384"),
    workers: Optional[int] = typer.Option(None, help="Concurrent generation requests (default: model's concurrency)"),
    unload: bool = typer.Option(
        False, "--unload-before-scoring",
        help="Evict the Ollama model after generation so its RAM is free for scoring (use for 27B)"),
    tag: str = typer.Option("", help="Free-text label stored in results/baselines.csv"),
    record: bool = typer.Option(True, help="Append the summary to results/baselines.csv"),
) -> None:
    """Generate solutions with MODEL and score them in the sandbox."""
    cfg = get_model(model)
    overrides = {k: v for k, v in {"temperature": temperature, "max_tokens": max_tokens}.items()
                 if v is not None}
    for kv in param:
        key, _, val = kv.partition("=")
        try:
            overrides[key] = json.loads(val)
        except json.JSONDecodeError:
            overrides[key] = val
    if overrides:
        cfg = type(cfg)(name=cfg.name, model=cfg.model, tier=cfg.tier,
                        params={**cfg.params, **overrides}, concurrency=cfg.concurrency)
    lcb = {}
    if dataset == "livecodebench":
        lcb = {"release": release, "start_date": start_date, "end_date": end_date,
               "platform": platform, "difficulty": difficulty}
    tasks = select_tasks(dataset, split=split, seed=seed, limit=limit, sample=sample, **lcb)
    system_prompt = prompt.read_text()

    done = 0
    def tick(_):
        nonlocal done
        done += 1
        if done % 10 == 0 or done == len(tasks):
            typer.echo(f"  generated {done}/{len(tasks)}")

    typer.echo(f"{cfg.name} on {dataset}/{split}: {len(tasks)} tasks")
    result = evaluate(cfg, system_prompt, tasks, seed=seed, workers=workers or cfg.concurrency,
                      unload_before_scoring=unload,
                      meta={"split": split, "tag": tag,
                            "filters": {k: v for k, v in lcb.items() if v} or None,
                            "sample": sample},
                      progress=tick)
    s = result.summary
    typer.echo(
        f"pass@1 base={s['pass_base']:.3f}  base+extra={s['pass_plus']:.3f}  "
        f"truncated={s['truncated']}  new_calls={s['llm_calls']}  "
        f"tokens={s['completion_tokens']}  gen={s['gen_wall_s']}s  eval={s['eval_wall_s']}s"
    )
    for d, v in s.get("by_difficulty", {}).items():
        typer.echo(f"  {d:7} n={v['n']:4}  pass@1={v['pass']:.3f}")
    typer.echo(f"run dir: {s['run_dir']}")
    if record:
        append_results_csv(s)


if __name__ == "__main__":
    app()
