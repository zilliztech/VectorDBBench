from contextlib import nullcontext

import pytest

pytest.importorskip("qdrant_client")

from vectordb_bench.backend.clients.qdrant_cloud.config import QdrantIndexConfig
from vectordb_bench.backend.clients.qdrant_cloud.qdrant_cloud import QdrantCloud


@pytest.mark.parametrize("raises", [False, True])
def test_qdrant_client_is_closed_after_context(raises: bool) -> None:
    database = QdrantCloud(2, {"location": ":memory:"}, QdrantIndexConfig())
    expected = pytest.raises(ValueError, match="benchmark failed") if raises else nullcontext()
    client = None
    try:
        with expected, database.init():
            client = database.qdrant_client
            assert client.get_collections().collections == []
            if raises:
                raise ValueError("benchmark failed")

        with pytest.raises(RuntimeError, match="closed"):
            client.get_collections()
        assert not hasattr(database, "qdrant_client")
    finally:
        if client is not None:
            client.close()
