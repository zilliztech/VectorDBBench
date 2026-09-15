"""Single choke point that finalizes a ``db_case_config`` before it enters a task.

All post-construction transformations of case configs (FTS routing and
dataset-derived metric defaults) are composed here, so a case config is
produced at exactly one place and only read afterwards. Both entry points —
the CLI (``vectordb_bench.cli.cli.run``) and the web UI
(``frontend/components/run_test/generateTasks.py``) — call
:func:`finalize_db_case_config` with the same semantics.
"""

from __future__ import annotations

from typing import Any

from .cases import CaseType
from .clients import DB, EmptyDBCaseConfig
from .clients.api import DBCaseConfig, IndexType

# Fields that remain meaningful when a non-FTS vector config is routed to a
# backend's FTS config class (e.g. a vector CLI command run with
# ``--case-type FTSBm25Performance``).
FTS_COMPATIBLE_FIELDS = (
    "number_of_shards",
    "number_of_replicas",
    "refresh_interval",
    "use_force_merge",
    "force_merge_enabled",
    "force_merge_target_size_mb",
    "disable_backpressure",
    "level",
)


def copy_fts_compatible_db_case_fields(source: DBCaseConfig, target: DBCaseConfig) -> DBCaseConfig:
    """Copy CLI fields that remain meaningful when routing a backend to its FTS config."""
    updates = {
        field: getattr(source, field)
        for field in FTS_COMPATIBLE_FIELDS
        if hasattr(source, field) and hasattr(target, field)
    }
    if not updates:
        return target
    return target.model_copy(update=updates)


def apply_fts_cli_db_case_params(
    db_case_config: DBCaseConfig,
    parameters: dict[str, Any] | None,
) -> DBCaseConfig:
    """Apply CLI-level BM25 overrides onto an FTS case config."""
    if not parameters:
        return db_case_config

    updates = {
        field: parameters[field]
        for field in ("bm25_k1", "bm25_b")
        if parameters.get(field) is not None and hasattr(db_case_config, field)
    }
    if not updates:
        return db_case_config
    return db_case_config.model_copy(update=updates)


def select_fts_db_case_config(
    db: DB,
    db_case_config: DBCaseConfig,
    case_type: str,
    parameters: dict[str, Any] | None = None,
) -> DBCaseConfig:
    """Route a CLI-constructed config to the backend's FTS config when needed.

    Only FTS performance case types route; everything else passes through
    unchanged.
    """
    if case_type != CaseType.FTSBm25Performance.name:
        return db_case_config

    fts_case_config_cls = db.case_config_cls(IndexType.FTS)
    if isinstance(db_case_config, fts_case_config_cls):
        return apply_fts_cli_db_case_params(db_case_config, parameters)
    fts_db_case_config = copy_fts_compatible_db_case_fields(db_case_config, fts_case_config_cls())
    return apply_fts_cli_db_case_params(fts_db_case_config, parameters)


def _apply_dataset_metric(case_type: str, config: DBCaseConfig, dataset: Any) -> bool:
    """Whether the dataset's metric type should be written onto the config."""
    if case_type == CaseType.FTSBm25Performance.name:
        return False
    if type(config) is EmptyDBCaseConfig:
        return False
    return hasattr(config, "metric_type") and hasattr(dataset, "metric_type")


def finalize_db_case_config(
    db: DB,
    case_type: str,
    base_config: DBCaseConfig,
    *,
    parameters: dict[str, Any] | None = None,
    dataset: Any | None = None,
) -> DBCaseConfig:
    """Produce the final ``db_case_config`` for a task.

    Composes, in order:

    1. CLI-only FTS routing (``parameters`` present only on the CLI path);
    2. dataset-derived ``metric_type`` default (previously applied in the
       assembler), via ``model_copy`` so the input config is never mutated.

    Args:
        db: target database.
        case_type: ``CaseType`` member name (e.g. ``"Performance1536D50K"``).
        base_config: config constructed by the CLI command or the web UI.
        parameters: CLI parameters dict; ``None`` on the web UI path.
        dataset: resolved dataset data (``case.dataset.data``); ``None`` to
            skip the metric default.

    Returns:
        A new config (or the same instance when no transform applies). The
        returned config is considered final: consumers must treat it as
        read-only.
    """
    config = base_config
    if parameters is not None:
        config = select_fts_db_case_config(db, config, case_type, parameters)
    if dataset is not None and _apply_dataset_metric(case_type, config, dataset):
        config = config.model_copy(update={"metric_type": dataset.metric_type})
    return config
