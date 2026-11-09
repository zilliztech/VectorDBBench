import pickle
from pathlib import Path

import numpy as np
import pytest

from vectordb_bench.backend.clients import DB, MetricType
from vectordb_bench.backend.clients.warm.config import WarmConfig, WarmIndexConfig

DIM = 32


def test_registry_and_param_mapping():
    assert DB.Warm.init_cls.__name__ == "Warm"
    assert DB.Warm.config_cls is WarmConfig
    assert DB.Warm.case_config_cls() is WarmIndexConfig
    cfg = WarmIndexConfig(metric_type=MetricType.COSINE, m=16, ef_construction=300, ef_search=150)
    assert cfg.index_param()["degree"] == 32
    assert cfg.index_param()["build_ef"] == 300
    assert cfg.index_param()["metric"] == "angular"
    assert cfg.search_param() == {"ef": 150, "rerank": 0}


def test_insert_optimize_search_across_processes_state(tmp_path: Path):
    """Ids come back as dataset ids, not row numbers, and a pickled copy (as a spawned worker gets)
    finds the index that optimize() saved."""
    pytest.importorskip("writerslogic_warm")
    rng = np.random.default_rng(0)
    x = rng.standard_normal((600, DIM)).astype(np.float32)
    # Non-contiguous ids inserted out of order in two batches.
    ids = (np.arange(600) * 3 + 7).tolist()
    order = rng.permutation(600)
    cfg = WarmIndexConfig(metric_type=MetricType.COSINE, m=8, ef_construction=100, ef_search=100)

    identity = {"name": "random", "size": 600, "seed": 0}
    db = DB.Warm.init_cls(DIM, {"path": str(tmp_path)}, cfg, drop_old=True, dataset_identity=identity)
    with db.init():
        for part in (order[:250], order[250:]):
            n, err = db.insert_embeddings(x[part].tolist(), [ids[i] for i in part])
            assert err is None
            assert n == len(part)
    with db.init():
        db.optimize()

    worker = pickle.loads(pickle.dumps(db))  # noqa: S301
    with worker.init():
        for q in (0, 123, 599):
            assert worker.search_embedding(x[q].tolist(), k=5)[0] == ids[q]
        batch = worker.search_embeddings(x[[0, 123, 599]].tolist(), k=5)
        assert [r[0] for r in batch] == [ids[0], ids[123], ids[599]]
        assert batch[1] == worker.search_embedding(x[123].tolist(), k=5)

    sweep = DB.Warm.init_cls(
        DIM,
        {"path": str(tmp_path)},
        cfg.model_copy(update={"ef_search": 150, "rerank": 150}),
        dataset_identity=identity,
    )
    with sweep.init():
        assert sweep.search_embedding(x[123].tolist(), k=5)[0] == ids[123]
        sweep.optimize()
        count, err = sweep.insert_embeddings(x[:1].tolist(), [9999])
        assert count == 0
        assert isinstance(err, ValueError)

    for change in ({"metric_type": MetricType.L2}, {"m": 16}, {"code_bits": 8}):
        with pytest.raises(ValueError, match="dataset or build settings do not match"):
            DB.Warm.init_cls(DIM, {"path": str(tmp_path)}, cfg.model_copy(update=change), dataset_identity=identity)


@pytest.mark.parametrize(
    "change",
    [
        {"metric_type": MetricType.L2},
        {"m": 32},
        {"ef_construction": 301},
        {"code_bits": 8},
        {"residual_bits": 4},
        {"threads": 1},
    ],
)
def test_changed_build_settings_reject_saved_identity(tmp_path: Path, change: dict):
    cfg = WarmIndexConfig()
    identity = {"name": "first", "size": 2}
    db = DB.Warm.init_cls(DIM, {"path": str(tmp_path)}, cfg, dataset_identity=identity)
    assert db.insert_embeddings(np.zeros((2, DIM)).tolist(), [0, 1]) == (2, None)
    with pytest.raises(ValueError, match="dataset or build settings do not match"):
        DB.Warm.init_cls(DIM, {"path": str(tmp_path)}, cfg.model_copy(update=change), dataset_identity=identity)


@pytest.mark.parametrize("dim,identity", [(DIM + 1, {"name": "first"}), (DIM, {"name": "second"})])
def test_changed_dataset_or_dimension_rejects_reuse(tmp_path: Path, dim: int, identity: dict):
    DB.Warm.init_cls(DIM, {"path": str(tmp_path)}, WarmIndexConfig(), dataset_identity={"name": "first"})
    with pytest.raises(ValueError, match="dataset or build settings do not match"):
        DB.Warm.init_cls(dim, {"path": str(tmp_path)}, WarmIndexConfig(), dataset_identity=identity)


@pytest.mark.parametrize("damage", ["legacy", "corrupt", "partial-index", "partial-ids"])
def test_incomplete_or_unidentified_index_fails_before_sdk_load(tmp_path: Path, damage: str):
    from vectordb_bench.backend.clients.warm.warm import IDS_FILE, INDEX_FILE, MANIFEST_FILE

    identity = {"name": "first"}
    db = DB.Warm.init_cls(DIM, {"path": str(tmp_path)}, WarmIndexConfig(), dataset_identity=identity)
    manifest = db.root / MANIFEST_FILE
    if damage == "legacy":
        manifest.unlink()
        (db.root / INDEX_FILE).write_bytes(b"legacy")
    elif damage == "corrupt":
        manifest.write_text("{")
    else:
        (db.root / (INDEX_FILE if damage == "partial-index" else IDS_FILE)).write_bytes(b"partial")
    with pytest.raises(ValueError, match="rebuild without --skip-drop-old"):
        DB.Warm.init_cls(DIM, {"path": str(tmp_path)}, WarmIndexConfig(), dataset_identity=identity)
    rebuilt = DB.Warm.init_cls(
        DIM, {"path": str(tmp_path)}, WarmIndexConfig(), dataset_identity=identity, drop_old=True
    )
    assert rebuilt.insert_embeddings(np.zeros((1, DIM)).tolist(), [0]) == (1, None)


def test_identity_is_rechecked_in_spawned_worker(tmp_path: Path):
    from vectordb_bench.backend.clients.warm.warm import MANIFEST_FILE

    db = DB.Warm.init_cls(DIM, {"path": str(tmp_path)}, WarmIndexConfig(), dataset_identity={"name": "first"})
    worker = pickle.loads(pickle.dumps(db))  # noqa: S301
    (db.root / MANIFEST_FILE).unlink()
    with pytest.raises(ValueError, match="manifest is unreadable"), worker.init():
        pytest.fail("worker must not load unidentified state")


def test_search_requires_completed_build(tmp_path: Path):
    db = DB.Warm.init_cls(DIM, {"path": str(tmp_path)}, WarmIndexConfig(), dataset_identity={"name": "first"})
    with db.init(), pytest.raises(RuntimeError, match="run the load stage"):
        db.search_embedding([0.0] * DIM)


def test_batch_search_is_advertised():
    assert DB.Warm.init_cls.supports_batch_search is True
