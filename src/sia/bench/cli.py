"""`bench` command-line tool (standalone for now; the framework imports runner.evaluate)."""

from __future__ import annotations

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


@app.command()
def models() -> None:
    """List configured models."""
    for name, m in load_models().items():
        typer.echo(f"{name:18} {m.tier:9} {m.model}")


@app.command()
def run(
    model: str = typer.Option(..., "--model", "-m", help="Name from configs/models.yaml"),
    dataset: str = typer.Option("humaneval", "--dataset", "-d", help="humaneval | mbpp"),
    split: str = typer.Option("all", help="all | train | val | test"),
    limit: Optional[int] = typer.Option(None, help="Only the first N tasks (smoke tests)"),
    prompt: Path = typer.Option(REPO / "prompts" / "base.md", help="System prompt file"),
    seed: int = typer.Option(0, help="Split seed and sampling seed"),
    temperature: Optional[float] = typer.Option(None, help="Override the configured temperature"),
    workers: int = typer.Option(4, help="Concurrent generation requests"),
    unload: bool = typer.Option(
        False, "--unload-before-scoring",
        help="Evict the Ollama model after generation so its RAM is free for scoring (use for 27B)"),
    tag: str = typer.Option("", help="Free-text label stored in results/baselines.csv"),
    record: bool = typer.Option(True, help="Append the summary to results/baselines.csv"),
) -> None:
    """Generate solutions with MODEL and score them in the sandbox."""
    cfg = get_model(model)
    if temperature is not None:
        cfg = type(cfg)(name=cfg.name, model=cfg.model, tier=cfg.tier,
                        params={**cfg.params, "temperature": temperature})
    tasks = select_tasks(dataset, split=split, seed=seed, limit=limit)
    system_prompt = prompt.read_text()

    done = 0
    def tick(_):
        nonlocal done
        done += 1
        if done % 10 == 0 or done == len(tasks):
            typer.echo(f"  generated {done}/{len(tasks)}")

    typer.echo(f"{cfg.name} on {dataset}/{split}: {len(tasks)} tasks")
    result = evaluate(cfg, system_prompt, tasks, seed=seed, workers=workers,
                      unload_before_scoring=unload,
                      meta={"split": split, "tag": tag}, progress=tick)
    s = result.summary
    typer.echo(
        f"pass@1 base={s['pass_base']:.3f}  base+extra={s['pass_plus']:.3f}  "
        f"truncated={s['truncated']}  new_calls={s['llm_calls']}  "
        f"tokens={s['completion_tokens']}  gen={s['gen_wall_s']}s  eval={s['eval_wall_s']}s"
    )
    typer.echo(f"run dir: {s['run_dir']}")
    if record:
        append_results_csv(s)


if __name__ == "__main__":
    app()
