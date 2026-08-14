from types import SimpleNamespace

import pytest

from vectordb_bench import config
from vectordb_bench.backend.clients.aws_opensearch.aws_opensearch import AWSOpenSearch


def test_serverless_insert_uses_configured_batch_size(monkeypatch) -> None:
    bulk_requests = []

    def bulk(*, body):
        bulk_requests.append(body)

    monkeypatch.setattr(config, "NUM_PER_BATCH", 2)

    db = object.__new__(AWSOpenSearch)
    db.client = SimpleNamespace(bulk=bulk)
    db._is_serverless = True
    db.index_name = "test-index"
    db.vector_col_name = "embedding"
    db.with_scalar_labels = False

    inserted, error = db._insert_with_single_client(
        embeddings=[[0.1], [0.2], [0.3], [0.4], [0.5]],
        metadata=[1, 2, 3, 4, 5],
    )

    assert inserted == 5
    assert error is None
    assert [len(request) // 2 for request in bulk_requests] == [2, 2, 1]
    assert [document["id"] for request in bulk_requests for document in request[1::2]] == [1, 2, 3, 4, 5]


@pytest.mark.parametrize("batch_size", [0, -1])
def test_serverless_insert_rejects_non_positive_batch_size(monkeypatch, batch_size: int) -> None:
    monkeypatch.setattr(config, "NUM_PER_BATCH", batch_size)

    db = object.__new__(AWSOpenSearch)
    db._is_serverless = True

    with pytest.raises(ValueError, match="NUM_PER_BATCH must be greater than 0"):
        db._insert_with_single_client(
            embeddings=[[0.1]],
            metadata=[1],
        )
