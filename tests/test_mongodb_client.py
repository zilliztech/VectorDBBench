"""Offline unit tests for MongoDB label-filter wiring.

Runners call prepare_filter(filters) then search_embedding(query, k) — they
never pass filters= into search. Atlas also requires {type: "filter", path:
"label"} on the vector search index. These tests mock insert_many / aggregate
so they do not need a live MongoDB.
"""

from unittest.mock import MagicMock, patch

import pytest

from vectordb_bench.backend.clients.mongodb.config import MongoDBIndexConfig
from vectordb_bench.backend.clients.mongodb.mongodb import MongoDB
from vectordb_bench.backend.filter import FilterOp, IntFilter, LabelFilter, NonFilter

_DB_CONFIG = {
    "connection_string": "mongodb://localhost:27017",
    "database": "vdb_bench",
}


class _IndexConfig(MongoDBIndexConfig):
    """main's search_param() omits exact (sibling PR). Search tests need the key."""

    def search_param(self) -> dict:
        params = super().search_param()
        params.setdefault("exact", False)
        return params


def _make_client(with_scalar_labels: bool = False):
    mock_client = MagicMock()
    mock_db = MagicMock()
    mock_collection = MagicMock()
    mock_client.__getitem__.return_value = mock_db
    mock_db.__getitem__.return_value = mock_collection
    mock_db.list_collection_names.return_value = []

    with patch(
        "vectordb_bench.backend.clients.mongodb.mongodb.MongoClient",
        return_value=mock_client,
    ):
        client = MongoDB(
            dim=4,
            db_config=_DB_CONFIG,
            db_case_config=_IndexConfig(),
            drop_old=False,
            with_scalar_labels=with_scalar_labels,
        )
    client.collection = mock_collection
    client.client = mock_client
    client.db = mock_db
    return client, mock_collection


def test_supported_filter_types_include_str_equal():
    assert MongoDB.supported_filter_types == [FilterOp.NonFilter, FilterOp.StrEqual]


def test_insert_embeddings_writes_label_when_with_scalar_labels():
    client, collection = _make_client(with_scalar_labels=True)
    count, err = client.insert_embeddings(
        embeddings=[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
        metadata=[1, 2],
        labels_data=["label_25p", "label_50p"],
    )
    assert err is None
    assert count == 2
    documents = collection.insert_many.call_args[0][0]
    assert documents[0]["label"] == "label_25p"
    assert documents[1]["label"] == "label_50p"
    assert documents[0]["id"] == 1
    assert documents[0]["vector"] == [0.1, 0.2, 0.3, 0.4]


def test_insert_embeddings_omits_label_without_scalar_labels():
    client, collection = _make_client(with_scalar_labels=False)
    client.insert_embeddings(
        embeddings=[[0.1, 0.2, 0.3, 0.4]],
        metadata=[1],
        labels_data=["label_25p"],
    )
    documents = collection.insert_many.call_args[0][0]
    assert "label" not in documents[0]


def test_search_embedding_applies_label_filter_after_prepare_filter():
    # Regression: runners never pass filters= into search_embedding.
    client, collection = _make_client(with_scalar_labels=True)
    collection.aggregate.return_value = [{"id": 1}]
    client.prepare_filter(LabelFilter(label_percentage=0.25))
    ids = client.search_embedding([0.1, 0.2, 0.3, 0.4], k=10)
    assert ids == [1]
    pipeline = collection.aggregate.call_args[0][0]
    assert pipeline[0]["$vectorSearch"]["filter"] == {"label": "label_25p"}


def test_prepare_filter_nonfilter_clears_label_filter():
    client, collection = _make_client(with_scalar_labels=True)
    collection.aggregate.return_value = [{"id": 1}]
    client.prepare_filter(LabelFilter(label_percentage=0.25))
    client.prepare_filter(NonFilter())
    client.search_embedding([0.1, 0.2, 0.3, 0.4], k=10)
    pipeline = collection.aggregate.call_args[0][0]
    assert "filter" not in pipeline[0]["$vectorSearch"]


def test_prepare_filter_rejects_numge():
    client, _ = _make_client()
    with pytest.raises(ValueError, match="(?i)not support|NumGE"):
        client.prepare_filter(IntFilter(int_value=500, filter_rate=0.99))


def _ready_search_indexes():
    return [
        [],
        [{"name": "vector_index", "queryable": True}],
        [{"name": "vector_index", "queryable": True}],
    ]


def test_create_index_declares_label_filter_when_with_scalar_labels():
    client, collection = _make_client(with_scalar_labels=True)
    collection.list_indexes.return_value = [True]
    collection.list_search_indexes.side_effect = _ready_search_indexes()
    client._create_index()
    fields = client.index_params["fields"]
    assert {"type": "filter", "path": "label"} in fields
    collection.create_search_index.assert_called_once()
    model = collection.create_search_index.call_args[0][0]
    assert {"type": "filter", "path": "label"} in model.document["definition"]["fields"]


def test_create_index_omits_label_filter_without_scalar_labels():
    client, collection = _make_client(with_scalar_labels=False)
    collection.list_indexes.return_value = [True]
    collection.list_search_indexes.side_effect = _ready_search_indexes()
    client._create_index()
    assert not any(field.get("type") == "filter" for field in client.index_params["fields"])


def test_filter_supported_matches_assembler_contract():
    assert MongoDB.filter_supported(NonFilter()) is True
    assert MongoDB.filter_supported(LabelFilter(label_percentage=0.25)) is True
    assert MongoDB.filter_supported(IntFilter(int_value=500, filter_rate=0.99)) is False


def test_insert_embeddings_accepts_runner_kwargs():
    client, collection = _make_client(with_scalar_labels=True)
    insert_kwargs = {
        "embeddings": [[0.1, 0.2, 0.3, 0.4]],
        "metadata": [7],
        "labels_data": ["label_5p"],
    }
    count, err = client.insert_embeddings(**insert_kwargs)
    assert err is None
    assert count == 1
    documents, kwargs = collection.insert_many.call_args[0][0], collection.insert_many.call_args[1]
    assert documents[0]["label"] == "label_5p"
    assert kwargs["ordered"] is False


def test_insert_embeddings_omits_label_when_labels_data_missing():
    client, collection = _make_client(with_scalar_labels=True)
    client.insert_embeddings(embeddings=[[0.1, 0.2, 0.3, 0.4]], metadata=[1])
    documents = collection.insert_many.call_args[0][0]
    assert "label" not in documents[0]


def test_insert_embeddings_returns_error_from_insert_many():
    client, collection = _make_client()
    collection.insert_many.side_effect = RuntimeError("bulk write failed")
    count, err = client.insert_embeddings([[0.1, 0.2, 0.3, 0.4]], [1])
    assert count == 0
    assert isinstance(err, RuntimeError)


def test_search_embedding_ignores_filters_kwarg():
    # Old client applied filters={"id": ...} here. Runners never pass that;
    # leftover callers must not resurrect the unfiltered/wrong-gte path.
    client, collection = _make_client(with_scalar_labels=True)
    collection.aggregate.return_value = [{"id": 1}]
    client.prepare_filter(LabelFilter(label_percentage=0.25))
    client.search_embedding([0.1, 0.2, 0.3, 0.4], k=10, filters={"id": 500})
    pipeline = collection.aggregate.call_args[0][0]
    assert pipeline[0]["$vectorSearch"]["filter"] == {"label": "label_25p"}
    assert "gte" not in str(pipeline[0]["$vectorSearch"]["filter"])


def test_search_embedding_sets_num_candidates_for_ann():
    client, collection = _make_client()
    collection.aggregate.return_value = [{"id": 3}, {"id": 1}]
    ids = client.search_embedding([0.1, 0.2, 0.3, 0.4], k=10)
    assert ids == [3, 1]
    vector_search = collection.aggregate.call_args[0][0][0]["$vectorSearch"]
    assert vector_search["limit"] == 10
    assert vector_search["numCandidates"] == 100
    assert "exact" not in vector_search
    assert "filter" not in vector_search


def test_create_index_does_not_duplicate_label_filter_field():
    client, collection = _make_client(with_scalar_labels=True)
    collection.list_indexes.return_value = [True]
    collection.list_search_indexes.side_effect = [
        [],
        [{"name": "vector_index", "queryable": True}],
        [],
        [{"name": "vector_index", "queryable": True}],
    ]
    client._create_index()
    client._create_index()
    fields = [field for field in client.index_params["fields"] if field.get("type") == "filter"]
    assert fields == [{"type": "filter", "path": "label"}]


def test_optimize_declares_label_filter_on_search_index():
    client, collection = _make_client(with_scalar_labels=True)
    collection.list_indexes.return_value = [True]
    collection.list_search_indexes.side_effect = [
        [],
        [{"name": "vector_index", "queryable": True}],
        [{"name": "vector_index", "queryable": True}],
    ]
    client.optimize()
    model = collection.create_search_index.call_args[0][0]
    assert {"type": "filter", "path": "label"} in model.document["definition"]["fields"]
