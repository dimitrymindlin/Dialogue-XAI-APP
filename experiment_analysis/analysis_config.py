"""Dynamic configuration loader for experiment analysis.

Each ``data_*`` folder under ``experiment_analysis`` can provide an
``experiment_analysis_config.py`` describing its dataset, DB settings,
and any experiment-specific constants.  This module locates the active
experiment configuration, imports it, and re-exports all uppercase
attributes so existing analysis scripts keep working without change.

The active experiment folder is selected by the ``ANALYSIS_EXPERIMENT_DIR``
environment variable.  When unset, ``DEFAULT_EXPERIMENT_DIR`` is used.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import ModuleType
from typing import Iterable, Optional, Set

ENV_VAR = "ANALYSIS_EXPERIMENT_DIR"
DEFAULT_EXPERIMENT_DIR = "data_diabetes_thesis_mapek_09_2025"
DEFAULT_DB_CONFIG = {
    "user": "postgres",
    "password": "password",
    "host": "localhost",
    "port": "5432",
    "dbname": "portainer_latest",
}

def _resolve_experiment_directory() -> tuple[str, Path]:
    """Return the folder name and absolute path for the active experiment."""

    folder_name = os.getenv(ENV_VAR, DEFAULT_EXPERIMENT_DIR)
    base_dir = Path(__file__).parent
    experiment_path = (base_dir / folder_name).resolve()

    if not experiment_path.is_dir():
        raise FileNotFoundError(
            f"Configured experiment folder '{folder_name}' does not exist at {experiment_path}."
        )

    return folder_name, experiment_path


def _load_module(path: Path) -> ModuleType:
    """Dynamically load the experiment config module from ``path``."""

    spec = importlib.util.spec_from_file_location("experiment_analysis.active_config", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive guard
        raise ImportError(f"Unable to load experiment config from {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[assignment]
    return module


def _export_uppercase_attributes(module: ModuleType) -> Iterable[str]:
    """Expose all UPPER_CASE attributes from ``module`` to this namespace."""

    exported = []
    for attribute in dir(module):
        if attribute.isupper():
            globals()[attribute] = getattr(module, attribute)
            exported.append(attribute)
    return exported


def _infer_identifiers(folder_name: str) -> Optional[dict[str, str]]:
    """Infer dataset metadata from a ``data_*`` folder name."""

    prefix = "data_"
    cleaned = folder_name[len(prefix):] if folder_name.startswith(prefix) else folder_name
    parts = cleaned.split("_")

    if len(parts) < 4:
        return None

    dataset = parts[0]
    experiment_handle = parts[1]
    group = parts[2]
    experiment_date = "_".join(parts[3:])

    return {
        "DATASET": dataset,
        "EXPERIMENT_HANDLE": experiment_handle,
        "GROUP": group,
        "EXPERIMENT_DATE": experiment_date,
    }


def _set_default(name: str, value: str, exported: Set[str]) -> None:
    if name not in exported:
        globals()[name] = value
        exported.add(name)


def _initialise() -> Iterable[str]:
    """Load the active experiment config and return exported names."""

    # Remove previously exported uppercase attributes so stale values do not linger.
    for attribute in list(_EXPORTED_ATTRIBUTES):
        globals().pop(attribute, None)
    _EXPORTED_ATTRIBUTES.clear()

    folder_name, experiment_path = _resolve_experiment_directory()
    config_path = experiment_path / "experiment_analysis_config.py"

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Active experiment folder '{folder_name}' is missing 'experiment_analysis_config.py'."
        )

    module = _load_module(config_path)
    exported = set(_export_uppercase_attributes(module))

    inferred = _infer_identifiers(folder_name)
    if inferred:
        for key, value in inferred.items():
            _set_default(key, value, exported)

    _set_default("PROLIFIC_CSV_FOLDER", str(experiment_path), exported)
    _set_default("PROLIFIC_FOLDER_NAME", folder_name, exported)

    if "RESULTS_BASE" not in exported:
        _set_default("RESULTS_BASE", str(experiment_path.parent / "results"), exported)

    if "RESULTS_DIR" not in exported:
        results_base = globals()["RESULTS_BASE"]
        _set_default("RESULTS_DIR", str(Path(results_base) / folder_name), exported)

    if "DB_CONFIG" not in exported:
        _set_default("DB_CONFIG", DEFAULT_DB_CONFIG.copy(), exported)

    if "WORK_LIFE_BALANCE_USERS" not in exported:
        _set_default("WORK_LIFE_BALANCE_USERS", [], exported)

    if "STEP3_EXCLUDE_FILTERS" not in exported:
        _set_default("STEP3_EXCLUDE_FILTERS", [], exported)

    # Expose metadata about the active configuration for downstream tooling.
    globals()["ACTIVE_EXPERIMENT_DIR"] = folder_name
    globals()["ACTIVE_EXPERIMENT_PATH"] = str(experiment_path)

    _EXPORTED_ATTRIBUTES.update(exported)

    return exported


def reload_active_config() -> None:
    """Refresh exported settings (useful after changing ``ANALYSIS_EXPERIMENT_DIR``)."""

    exported_names = sorted(_initialise())
    globals()["__all__"] = ["ACTIVE_EXPERIMENT_DIR", "ACTIVE_EXPERIMENT_PATH", *exported_names]


# Perform initial load on module import.
_EXPORTED_ATTRIBUTES: Set[str] = set()
exported_names = sorted(_initialise())
__all__ = ["ACTIVE_EXPERIMENT_DIR", "ACTIVE_EXPERIMENT_PATH", *exported_names]
