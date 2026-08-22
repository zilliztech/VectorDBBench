from types import SimpleNamespace

import pytest
from opensearchpy import ConnectionTimeout, TransportError

from vectordb_bench import config
from vectordb_bench.backend.clients.aws_opensearch.aws_opensearch import (
    BULK_MAX_ATTEMPTS,
    AWSOpenSearch,
    OpenSearchBulkInsertError,
)


def _bulk_response(*statuses: int) -> dict[str, object]:
    items = []
    for position, status in enumerate(statuses):
        result = {"_id": str(position), "status": status}
        if status >= 300:
            result["error"] = {"type": "rejected", "reason": "test failure"}
        items.append({"index": result})
    return {"errors": any(status >= 300 for status in statuses), "items": items}


def test_serverless_insert_uses_configured_batch_size(monkeypatch: pytest.MonkeyPatch) -> None:
    bulk_requests: list[list[dict[str, object]]] = []

    def bulk(*, body: list[dict[str, object]]) -> dict[str, object]:
        bulk_requests.append(body)
        return _bulk_response(*(201 for _ in body[::2]))

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
def test_serverless_insert_rejects_non_positive_batch_size(
    monkeypatch: pytest.MonkeyPatch,
    batch_size: int,
) -> None:
    monkeypatch.setattr(config, "NUM_PER_BATCH", batch_size)

    db = object.__new__(AWSOpenSearch)
    db._is_serverless = True

    with pytest.raises(ValueError, match="NUM_PER_BATCH must be greater than 0"):
        db._insert_with_single_client(
            embeddings=[[0.1]],
            metadata=[1],
        )


def test_serverless_insert_retries_only_failed_documents(monkeypatch: pytest.MonkeyPatch) -> None:
    bulk_requests: list[list[dict[str, object]]] = []
    retry_delays: list[int] = []
    responses = iter([_bulk_response(201, 429, 201), _bulk_response(201)])

    def bulk(*, body: list[dict[str, object]]) -> dict[str, object]:
        bulk_requests.append(body)
        return next(responses)

    monkeypatch.setattr(config, "NUM_PER_BATCH", 3)
    monkeypatch.setattr(
        "vectordb_bench.backend.clients.aws_opensearch.aws_opensearch.time.sleep",
        retry_delays.append,
    )

    db = object.__new__(AWSOpenSearch)
    db.client = SimpleNamespace(bulk=bulk)
    db._is_serverless = True
    db.index_name = "test-index"
    db.vector_col_name = "embedding"
    db.with_scalar_labels = False

    inserted, error = db._insert_with_single_client(
        embeddings=[[0.1], [0.2], [0.3]],
        metadata=[1, 2, 3],
    )

    assert inserted == 3
    assert error is None
    assert [len(request) // 2 for request in bulk_requests] == [3, 1]
    assert bulk_requests[1][1]["id"] == 2
    assert retry_delays == [2]


def test_serverless_insert_fails_after_partial_failure_exhausts_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bulk_requests: list[list[dict[str, object]]] = []
    retry_delays: list[int] = []
    errors: list[str] = []

    def bulk(*, body: list[dict[str, object]]) -> dict[str, object]:
        bulk_requests.append(body)
        if len(bulk_requests) == 1:
            return _bulk_response(201, 429)
        return _bulk_response(429)

    monkeypatch.setattr(config, "NUM_PER_BATCH", 2)
    monkeypatch.setattr(
        "vectordb_bench.backend.clients.aws_opensearch.aws_opensearch.time.sleep",
        retry_delays.append,
    )
    monkeypatch.setattr(
        "vectordb_bench.backend.clients.aws_opensearch.aws_opensearch.log.error",
        errors.append,
    )

    db = object.__new__(AWSOpenSearch)
    db.client = SimpleNamespace(bulk=bulk)
    db._is_serverless = True
    db.index_name = "test-index"
    db.vector_col_name = "embedding"
    db.with_scalar_labels = False

    inserted, error = db._insert_with_single_client(
        embeddings=[[0.1], [0.2]],
        metadata=[1, 2],
    )

    assert inserted == 1
    assert isinstance(error, OpenSearchBulkInsertError)
    assert error.non_retryable is True
    assert "left 1 documents uninserted after 30 attempts; successful=1" in str(error)
    assert len(bulk_requests) == BULK_MAX_ATTEMPTS
    assert [len(request) // 2 for request in bulk_requests] == [2] + [1] * (BULK_MAX_ATTEMPTS - 1)
    assert retry_delays == [2, 4, 8, 16, 32] + [60] * 24
    assert sum(retry_delays) == 1502
    assert any("left 1 documents uninserted after 30 attempts; successful=1" in message for message in errors)


def test_serverless_insert_retries_request_level_429(monkeypatch: pytest.MonkeyPatch) -> None:
    bulk_requests: list[list[dict[str, object]]] = []
    retry_delays: list[int] = []

    def bulk(*, body: list[dict[str, object]]) -> dict[str, object]:
        bulk_requests.append(body)
        if len(bulk_requests) <= 2:
            raise TransportError(429, "too many requests")
        return _bulk_response(201)

    monkeypatch.setattr(config, "NUM_PER_BATCH", 1)
    monkeypatch.setattr(
        "vectordb_bench.backend.clients.aws_opensearch.aws_opensearch.time.sleep",
        retry_delays.append,
    )

    db = object.__new__(AWSOpenSearch)
    db.client = SimpleNamespace(bulk=bulk)
    db._is_serverless = True
    db.index_name = "test-index"
    db.vector_col_name = "embedding"
    db.with_scalar_labels = False

    inserted, error = db._insert_with_single_client(
        embeddings=[[0.1]],
        metadata=[1],
    )

    assert inserted == 1
    assert error is None
    assert len(bulk_requests) == 3
    assert retry_delays == [2, 4]


def test_serverless_insert_does_not_retry_ambiguous_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    bulk_requests: list[list[dict[str, object]]] = []
    retry_delays: list[int] = []

    def bulk(*, body: list[dict[str, object]]) -> dict[str, object]:
        bulk_requests.append(body)
        raise ConnectionTimeout(None, "timed out", None)

    monkeypatch.setattr(config, "NUM_PER_BATCH", 1)
    monkeypatch.setattr(
        "vectordb_bench.backend.clients.aws_opensearch.aws_opensearch.time.sleep",
        retry_delays.append,
    )

    db = object.__new__(AWSOpenSearch)
    db.client = SimpleNamespace(bulk=bulk)
    db._is_serverless = True
    db.index_name = "test-index"
    db.vector_col_name = "embedding"
    db.with_scalar_labels = False

    inserted, error = db._insert_with_single_client(
        embeddings=[[0.1]],
        metadata=[1],
    )

    assert inserted == 0
    assert isinstance(error, OpenSearchBulkInsertError)
    assert error.non_retryable is True
    assert "ambiguous outcome" in str(error)
    assert len(bulk_requests) == 1
    assert retry_delays == []


def test_multiple_clients_fallback_preserves_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    class BulkClient:
        def __init__(self, response: dict[str, object]) -> None:
            self.response = response

        def bulk(self, *, body: list[dict[str, object]]) -> dict[str, object]:
            return self.response

        def close(self) -> None:
            return None

    clients = iter([BulkClient(_bulk_response(201)), BulkClient(_bulk_response(429))])
    monkeypatch.setattr(
        "vectordb_bench.backend.clients.aws_opensearch.aws_opensearch.OpenSearch",
        lambda **_: next(clients),
    )
    monkeypatch.setattr("vectordb_bench.backend.clients.aws_opensearch.aws_opensearch.time.sleep", lambda _: None)

    fallback_requests: list[list[dict[str, object]]] = []

    def fallback_bulk(*, body: list[dict[str, object]]) -> dict[str, object]:
        fallback_requests.append(body)
        return _bulk_response(*(201 for _ in body[::2]))

    db = object.__new__(AWSOpenSearch)
    db.client = SimpleNamespace(
        bulk=fallback_bulk,
        indices=SimpleNamespace(
            stats=lambda **_: {"_all": {"primaries": {"indexing": {"index_total": 1}}}},
        ),
    )
    db.db_config = {}
    db.case_config = SimpleNamespace(use_routing=False)
    db.index_name = "test-index"
    db.id_col_name = "_id"
    db.vector_col_name = "embedding"
    db.label_col_name = "label"
    db.with_scalar_labels = True
    db._is_serverless = False

    inserted, error = db._insert_with_multiple_clients(
        embeddings=[[0.1], [0.2]],
        metadata=[1, 2],
        num_clients=2,
        labels_data=["first", "second"],
    )

    assert inserted == 2
    assert error is None
    assert [document["label"] for document in fallback_requests[0][1::2]] == ["first", "second"]
