from unittest.mock import Mock

import pytest

from vectordb_bench.backend.clients.lindorm.lindorm_search import LindormVector


def _response(*statuses, errors=False):
    return {
        "errors": errors,
        "items": [{"index": {"_id": str(doc_id), "status": status}} for doc_id, status in enumerate(statuses, 1)],
    }


@pytest.fixture
def vector(monkeypatch):
    db = object.__new__(LindormVector)
    db.index_name = "test"
    db.id_col_name = "_id"
    db.vector_col_name = "embedding"
    db.metadata_col_name = "metadata_fields"
    db.label_col_name = "label"
    db.write_thread_num = 1
    db.write_vector_batch = 1000
    db.client = Mock()
    monkeypatch.setattr("vectordb_bench.backend.clients.lindorm.lindorm_search.time.sleep", Mock())
    return db


def test_bulk_retries_entire_batch_and_counts_once(vector, caplog):
    failed = _response(201, 429, errors=True)
    failed["items"][1]["index"]["error"] = {"type": "rejected_execution_exception", "reason": "queue full"}
    vector.client.bulk.side_effect = [failed, _response(200, 201)]

    assert vector.insert_embeddings([[0.1], [0.2]], [1, 2], ["a", "b"]) == (2, None)
    calls = vector.client.bulk.call_args_list
    assert len(calls) == 2
    assert calls[0] == calls[1]
    assert [item["index"]["_id"] for item in calls[1].kwargs["body"][::2]] == [1, 2]
    assert "'id': 2" in caplog.text
    assert "rejected_execution_exception" in caplog.text
    assert "queue full" in caplog.text


def test_bulk_exhaustion_propagates_error_without_counting_failed_batch(vector):
    failed = _response(201, 500, errors=True)
    failed["items"][1]["index"]["error"] = {"type": "server_error", "reason": "write failed"}
    vector.client.bulk.return_value = failed

    count, error = vector.insert_embeddings([[0.1], [0.2]], [1, 2])

    assert count == 0
    assert isinstance(error, RuntimeError)
    assert "'id': 2" in str(error)
    assert "server_error" in str(error)
    assert vector.client.bulk.call_count == 10


def test_bulk_missing_item_retries_and_reports_unconfirmed_id(vector, caplog):
    vector.client.bulk.side_effect = [_response(201), _response(200, 201)]

    assert vector.insert_embeddings([[0.1], [0.2]], [1, 2]) == (2, None)
    assert vector.client.bulk.call_count == 2
    assert "expected_items=2, actual_items=1" in caplog.text
    assert "'id': 2" in caplog.text
