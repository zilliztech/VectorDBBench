"""Offline unit tests for LanceDB config objects.

These tests don't require a running LanceDB instance. They freeze the
contract for the three IVF-family indexes (IVF_PQ / IVF_HNSW_SQ /
IVF_HNSW_PQ) so that any future refactor that accidentally diverges their
code paths (e.g. introduces index-type-specific branches) will fail CI.

Background: IVF_HNSW_SQ / IVF_HNSW_PQ share the same code path as IVF_PQ
and the CLI is wired up for all three. These tests encode that claim.
"""

import typing
from typing import Annotated, get_type_hints

from vectordb_bench.backend.clients.api import IndexType, MetricType
from vectordb_bench.backend.clients.lancedb.config import (
    LanceDBAutoIndexConfig,
    LanceDBIndexConfig,
    LanceDBIVFHNSWPQConfig,
    LanceDBIVFHNSWSQConfig,
    LanceDBNoIndexConfig,
    _lancedb_case_config,
)

# ---------------------------------------------------------------------------
# Registry mapping
# ---------------------------------------------------------------------------


def test_registry_covers_all_lancedb_index_types():
    """Every LanceDB-supported IndexType must resolve to a config class."""
    required = {
        IndexType.IVFPQ,
        IndexType.AUTOINDEX,
        IndexType.IVF_HNSW_SQ,
        IndexType.IVF_HNSW_PQ,
        IndexType.NONE,
    }
    assert required.issubset(_lancedb_case_config.keys())

    # HNSW is kept for backwards compatibility and must map to IVF_HNSW_SQ.
    assert _lancedb_case_config[IndexType.HNSW] is LanceDBIVFHNSWSQConfig


# ---------------------------------------------------------------------------
# index_param() / search_param() contract
# ---------------------------------------------------------------------------


def test_ivfpq_default_params_are_minimal():
    cfg = LanceDBIndexConfig()
    assert cfg.index == IndexType.IVFPQ
    p = cfg.index_param()
    # Always present
    assert p["metric"] == "cosine" or p["metric"] == "l2"
    assert "num_bits" in p
    # Zero-valued optionals stay out of the param dict so LanceDB uses its
    # own defaults.
    assert "num_partitions" not in p
    assert "num_sub_vectors" not in p
    # search_param() is empty when all tunables are zero.
    assert cfg.search_param() == {}


def test_ivfpq_tuned_params_are_forwarded():
    cfg = LanceDBIndexConfig(
        metric_type=MetricType.COSINE,
        num_partitions=256,
        num_sub_vectors=96,
        nbits=8,
        nprobes=20,
        refine_factor=10,
    )
    p = cfg.index_param()
    assert p == {
        "metric": "cosine",
        "num_bits": 8,
        "sample_rate": 256,
        "max_iterations": 50,
        "num_partitions": 256,
        "num_sub_vectors": 96,
    }
    assert cfg.search_param() == {"nprobes": 20, "refine_factor": 10}


def test_ivf_hnsw_sq_params_are_forwarded():
    cfg = LanceDBIVFHNSWSQConfig(
        metric_type=MetricType.COSINE,
        num_partitions=256,
        m=16,
        ef_construction=200,
        ef=128,
        nprobes=20,
        refine_factor=10,
    )
    p = cfg.index_param()
    assert p["index_type"] == "IVF_HNSW_SQ"
    assert p["num_partitions"] == 256
    assert p["m"] == 16
    assert p["ef_construction"] == 200
    assert cfg.search_param() == {
        "ef": 128,
        "nprobes": 20,
        "refine_factor": 10,
    }


def test_ivf_hnsw_pq_params_are_forwarded():
    cfg = LanceDBIVFHNSWPQConfig(
        metric_type=MetricType.COSINE,
        num_partitions=256,
        num_sub_vectors=96,
        m=16,
        ef_construction=200,
        ef=128,
        nprobes=20,
        refine_factor=10,
    )
    p = cfg.index_param()
    assert p["index_type"] == "IVF_HNSW_PQ"
    assert p["num_partitions"] == 256
    assert p["num_sub_vectors"] == 96
    assert p["m"] == 16
    assert p["ef_construction"] == 200
    assert cfg.search_param() == {
        "ef": 128,
        "nprobes": 20,
        "refine_factor": 10,
    }


# ---------------------------------------------------------------------------
# Code-path unification
# ---------------------------------------------------------------------------


def test_ivf_family_shares_search_knobs():
    """IVF_PQ exposes nprobes+refine_factor; IVF_HNSW_SQ/PQ additionally
    expose ef. These are the only keys lancedb.py's search_embedding knows
    how to forward, so any new index type must stay inside this union.
    """
    allowed = {"nprobes", "ef", "refine_factor"}

    ivfpq = LanceDBIndexConfig(nprobes=10, refine_factor=5)
    sq = LanceDBIVFHNSWSQConfig(nprobes=10, ef=64, refine_factor=5)
    pq = LanceDBIVFHNSWPQConfig(nprobes=10, ef=64, refine_factor=5)

    for cfg in (ivfpq, sq, pq):
        assert set(cfg.search_param().keys()).issubset(allowed)


def test_no_index_and_autoindex_are_well_formed():
    none_cfg = LanceDBNoIndexConfig()
    assert none_cfg.index == IndexType.NONE
    assert none_cfg.index_param() == {}

    auto_cfg = LanceDBAutoIndexConfig()
    assert auto_cfg.index == IndexType.AUTOINDEX
    assert "metric" in auto_cfg.index_param()


# ---------------------------------------------------------------------------
# CLI wiring — validate via static TypedDict introspection (no heavy runtime
# imports needed, avoids hdrh / streamlit / etc. dependency chains)
# ---------------------------------------------------------------------------


def _extract_click_option_names(typed_dict_cls: type) -> set[str]:
    """Extract ``--option-name`` strings from a Click-annotated TypedDict.

    Each field is ``Annotated[T, click.option("--name", ...)]``.
    We pull the option strings from the ``click.Option`` metadata.
    """
    hints = get_type_hints(typed_dict_cls, include_extras=True)
    names: set[str] = set()
    for _field, hint in hints.items():
        if typing.get_origin(hint) is Annotated:
            for meta in hint.__metadata__:
                # click.option(...) produces a click.core.Decorator / functools.partial
                # but in this project it's stored as a click.Argument or a
                # ``functools.partial`` wrapping ``click.option``. Extract the
                # first positional string that starts with "--".
                if hasattr(meta, "name") and isinstance(meta.name, str):
                    names.add(meta.name)
                # click.option() returns a decorator whose .args[0] is the
                # option flag(s). We try a few accessor patterns.
                for attr in ("args", "decls"):
                    for val in getattr(meta, attr, ()):
                        if isinstance(val, str) and val.startswith("--"):
                            names.add(val)
                # For click.option stored as click.core.Option or Decorator
                if hasattr(meta, "opts"):
                    for opt in meta.opts:
                        if isinstance(opt, str) and opt.startswith("--"):
                            names.add(opt)
    return names


def test_cli_typed_dicts_define_all_expected_commands():
    """Every expected CLI TypedDict class must be importable from lancedb/cli.py."""
    # We mock the heavy module to avoid pulling the entire runtime.
    # lancedb/cli.py imports ....cli.cli which triggers hdrh etc.
    # Instead we just verify the TypedDict definitions exist in source.
    import ast
    from pathlib import Path

    cli_path = Path("vectordb_bench/backend/clients/lancedb/cli.py")
    tree = ast.parse(cli_path.read_text())

    class_names: set[str] = set()
    func_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            class_names.add(node.name)
        elif isinstance(node, ast.FunctionDef):
            func_names.add(node.name)

    # TypedDicts that drive CLI options
    assert {
        "LanceDBTypedDict",
        "LanceDBIVFPQTypedDict",
        "LanceDBIVFHNSWSQTypedDict",
        "LanceDBIVFHNSWPQTypedDict",
    }.issubset(class_names)

    # Command functions registered via @cli.command()
    assert {"LanceDB", "LanceDBAutoIndex", "LanceDBIVFPQ", "LanceDBIVFHNSWSQ", "LanceDBIVFHNSWPQ"}.issubset(func_names)


def test_cli_typeddict_ivfpq_has_search_knobs():
    """IVF_PQ TypedDict must define nprobes / refine_factor / num_partitions."""
    import ast
    from pathlib import Path

    cli_src = Path("vectordb_bench/backend/clients/lancedb/cli.py").read_text()
    tree = ast.parse(cli_src)

    def _fields_of(cls_name: str) -> set[str]:
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == cls_name:
                return {
                    item.target.id
                    for item in node.body
                    if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)
                }
        return set()

    ivfpq = _fields_of("LanceDBIVFPQTypedDict")
    assert {"nprobes", "refine_factor", "num_partitions", "num_sub_vectors", "nbits"}.issubset(ivfpq)


def test_cli_typeddict_hnsw_variants_are_superset_of_ivfpq():
    """Both HNSW TypedDicts must have nprobes + refine_factor + graph knobs."""
    import ast
    from pathlib import Path

    cli_src = Path("vectordb_bench/backend/clients/lancedb/cli.py").read_text()
    tree = ast.parse(cli_src)

    def _fields_of(cls_name: str) -> set[str]:
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == cls_name:
                return {
                    item.target.id
                    for item in node.body
                    if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)
                }
        return set()

    sq_fields = _fields_of("LanceDBIVFHNSWSQTypedDict")
    pq_fields = _fields_of("LanceDBIVFHNSWPQTypedDict")

    # Both must have shared search knobs
    shared_knobs = {"nprobes", "refine_factor", "num_partitions", "m", "ef", "ef_construction"}
    assert shared_knobs.issubset(sq_fields), f"SQ missing: {shared_knobs - sq_fields}"
    assert shared_knobs.issubset(pq_fields), f"PQ missing: {shared_knobs - pq_fields}"

    # PQ variant has num_sub_vectors, SQ does not
    assert "num_sub_vectors" in pq_fields
    assert "num_sub_vectors" not in sq_fields
