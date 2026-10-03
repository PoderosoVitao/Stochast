from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

from stochast.adapters import AgentAdapter


# Executes a Python file as a fresh module, so its top-level @scenario
# decorators (or adapter factory) register/run as a side effect.
def import_file(path: Path) -> ModuleType:
    module_name = "_stochast_" + path.resolve().as_posix().replace("/", "_").replace(".", "_")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path} as a Python module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Imports every scenario file under `path` (or `path` itself if it's a file)
# so their @scenario-decorated functions register into the global registry.
def discover_scenarios(path: Path) -> None:
    files = [path] if path.is_file() else sorted(path.rglob("*.py"))
    for file in files:
        import_file(file)


# Resolves "module:factory" or "path/to/file.py:factory" into the callable.
def resolve_adapter_factory(spec: str) -> Callable[[], AgentAdapter]:
    target, sep, attr = spec.rpartition(":")
    if not sep:
        raise ValueError("expected format module:factory or file.py:factory")
    module = (
        import_file(Path(target)) if target.endswith(".py") else importlib.import_module(target)
    )
    return getattr(module, attr)  # type: ignore[no-any-return]
