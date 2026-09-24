from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import typer
from rich.console import Console
from rich.progress import Progress

from stochast.adapters import AgentAdapter
from stochast.records import save_run_records
from stochast.runner import RunInterrupted, run_scenario
from stochast.scenario import Scenario, clear_registry, registered_scenarios

app = typer.Typer()
console = Console()


# Forces "stochast run ..." to require the "run" subcommand name even while
# it's the app's only command; Typer otherwise collapses a single command.
@app.callback()
def _callback() -> None:
    pass


# Executes a Python file as a fresh module, so its top-level @scenario
# decorators (or adapter factory) register/run as a side effect.
def _import_file(path: Path) -> ModuleType:
    module_name = "_stochast_" + path.resolve().as_posix().replace("/", "_").replace(".", "_")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path} as a Python module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Imports every scenario file under `path` (or `path` itself if it's a file)
# so their @scenario-decorated functions register into the global registry.
def _discover_scenarios(path: Path) -> None:
    files = [path] if path.is_file() else sorted(path.rglob("*.py"))
    for file in files:
        _import_file(file)


# Resolves "module:factory" or "path/to/file.py:factory" into the callable.
def _resolve_factory(spec: str) -> Callable[[], AgentAdapter]:
    target, sep, attr = spec.rpartition(":")
    if not sep:
        raise typer.BadParameter("expected format module:factory or file.py:factory")
    module = (
        _import_file(Path(target)) if target.endswith(".py") else importlib.import_module(target)
    )
    return getattr(module, attr)  # type: ignore[no-any-return]


@app.command()
def run(
    path: Path = typer.Argument(..., exists=True, help="scenario file or directory to run"),
    adapter: str = typer.Option(
        ..., "--adapter", help="module:factory or file.py:factory returning an AgentAdapter"
    ),
    keyword: str = typer.Option("", "-k", help="only run scenarios whose name contains this"),
    runs: int = typer.Option(0, "--runs", help="override every scenario's run count"),
    concurrency: int = typer.Option(5, "--concurrency", help="max runs executing at once"),
    retries: int = typer.Option(3, "--retries", help="transport-error retries per run"),
    seed: int = typer.Option(-1, "--seed", help="base seed for deterministic runs"),
    fail_under: float = typer.Option(
        1.0, "--fail-under", help="exit 1 if any pass rate is below this"
    ),
    out: Path = typer.Option(
        Path("stochast-results"), "-o", "--out", help="directory for JSON reports"
    ),
) -> None:
    clear_registry()
    _discover_scenarios(path)
    factory = _resolve_factory(adapter)

    scenarios = [s for s in registered_scenarios() if keyword in s.name]
    if not scenarios:
        console.print("[yellow]No scenarios matched.[/yellow]")
        raise typer.Exit(code=1)

    ok = True
    for scenario in scenarios:
        ok &= _run_one(
            replace(scenario, runs=runs) if runs else scenario,
            factory,
            concurrency=concurrency,
            retries=retries,
            seed=None if seed < 0 else seed,
            fail_under=fail_under,
            out=out,
        )

    raise typer.Exit(code=0 if ok else 1)


# Runs a single scenario, writes its RunRecords to disk, and prints its pass
# rate. Returns whether its pass rate met fail_under.
def _run_one(
    scenario: Scenario,
    factory: Callable[[], AgentAdapter],
    *,
    concurrency: int,
    retries: int,
    seed: int | None,
    fail_under: float,
    out: Path,
) -> bool:
    console.print(f"[bold]{scenario.name}[/bold] ({scenario.runs} runs)")

    with Progress(console=console, transient=True) as progress:
        task = progress.add_task("running", total=scenario.runs)
        try:
            records = run_scenario(
                scenario,
                factory,
                concurrency=concurrency,
                retries=retries,
                seed=seed,
                on_progress=lambda done, _total: progress.update(task, completed=done),
            )
        except RunInterrupted as exc:
            save_run_records(exc.records, out / f"{scenario.name}.json")
            console.print("[red]Interrupted; partial results written.[/red]")
            raise typer.Exit(code=130) from None

    save_run_records(records, out / f"{scenario.name}.json")

    passed = sum(1 for r in records if r.passed)
    total = len(records)
    rate = passed / total if total else 0.0
    console.print(f"  pass rate: {passed}/{total} ({rate:.0%})")
    return rate >= fail_under
