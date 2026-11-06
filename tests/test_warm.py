import pickle
from pathlib import Path

import numpy as np
import pytest

from vectordb_bench.backend.clients import DB, MetricType
from vectordb_bench.backend.clients.warm.config import WarmConfig, WarmIndexConfig

pytest.importorskip("writerslogic_warm")

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
    rng = np.random.default_rng(0)
    x = rng.standard_normal((600, DIM)).astype(np.float32)
    # Non-contiguous ids inserted out of order in two batches.
    ids = (np.arange(600) * 3 + 7).tolist()
    order = rng.permutation(600)
    cfg = WarmIndexConfig(metric_type=MetricType.COSINE, m=8, ef_construction=100, ef_search=100)

    db = DB.Warm.init_cls(DIM, {"path": str(tmp_path)}, cfg, drop_old=True)
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


def test_batch_search_is_advertised():
    assert DB.Warm.init_cls.supports_batch_search is True
