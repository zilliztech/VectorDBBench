from unittest.mock import patch

import pytest

chromadb = pytest.importorskip("chromadb")

from chromadb.config import Settings

from vectordb_bench.backend.clients.api import MetricType
from vectordb_bench.backend.clients.chroma.chroma import ChromaClient
from vectordb_bench.backend.clients.chroma.config import ChromaIndexConfig


@pytest.mark.parametrize("allow_reset", [False, True])
@pytest.mark.parametrize("drop_old", [False, True])
@pytest.mark.parametrize("target_exists", [False, True])
def test_drop_old_only_resets_benchmark_collection(tmp_path, allow_reset, drop_old, target_exists):
    client = chromadb.PersistentClient(
        path=str(tmp_path), settings=Settings(allow_reset=allow_reset, anonymized_telemetry=False)
    )
    unrelated = client.create_collection("unrelated_collection", embedding_function=None)
    unrelated.add(ids=["99"], embeddings=[[0.0, 1.0, 0.0]])
    if target_exists:
        target = client.create_collection("benchmark_target", embedding_function=None)
        target.add(ids=["1"], embeddings=[[1.0, 0.0, 0.0]])

    with patch("vectordb_bench.backend.clients.chroma.chroma.chromadb.HttpClient", return_value=client):
        benchmark = ChromaClient(
            dim=3,
            db_config={},
            db_case_config=ChromaIndexConfig(metric_type=MetricType.L2),
            collection_name="benchmark_target",
            drop_old=drop_old,
        )
        with benchmark.init():
            assert benchmark.collection.count() == int(target_exists and not drop_old)
            assert benchmark.insert_embeddings([[0.0, 0.0, 1.0]], [2]) == (1, None)
            assert benchmark.search_embedding([0.0, 0.0, 1.0], k=1) == [2]

    assert client.get_collection("unrelated_collection").get()["ids"] == ["99"]


def test_drop_old_propagates_unexpected_delete_error(tmp_path):
    client = chromadb.PersistentClient(path=str(tmp_path), settings=Settings(anonymized_telemetry=False))
    error = RuntimeError("collection deletion failed")

    with (
        patch("vectordb_bench.backend.clients.chroma.chroma.chromadb.HttpClient", return_value=client),
        patch.object(client, "delete_collection", side_effect=error),
        pytest.raises(RuntimeError) as caught,
    ):
        ChromaClient(
            dim=3,
            db_config={},
            db_case_config=ChromaIndexConfig(metric_type=MetricType.L2),
            collection_name="benchmark_target",
            drop_old=True,
        )

    assert caught.value is error
