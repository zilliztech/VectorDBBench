"""Offline unit tests for the DynamoDB vector client.

These tests do not require AWS credentials or a running DynamoDB. They freeze
the config contract (metric parsing, credential passthrough, to_dict shape)
and the CLI wiring so a future refactor that breaks them fails CI.
"""

import ast
from pathlib import Path

import pytest
from pydantic import SecretStr

import vectordb_bench.backend.clients.dynamodb.dynamodb as ddb_mod
from vectordb_bench.backend.clients import DB
from vectordb_bench.backend.clients.api import MetricType
from vectordb_bench.backend.clients.dynamodb.config import (
    DynamoDBConfig,
    DynamoDBIndexConfig,
)
from vectordb_bench.backend.clients.dynamodb.dynamodb import DynamoDB as DynamoDBClient
from vectordb_bench.backend.filter import LabelFilter, NewIntFilter, NonFilter

# ---------------------------------------------------------------------------
# Enum registration
# ---------------------------------------------------------------------------


def test_dynamodb_enum_resolves_config_and_init():
    assert DB.DynamoDB.value == "DynamoDB"
    assert DB.DynamoDB.config_cls is DynamoDBConfig
    assert DB.DynamoDB.init_cls is DynamoDBClient


# ---------------------------------------------------------------------------
# Metric parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("metric", "expected"),
    [
        (MetricType.COSINE, "COSINE"),
        (MetricType.L2, "EUCLIDEAN"),
        (MetricType.IP, "DOT_PRODUCT"),
    ],
)
def test_metric_parsing(metric: MetricType, expected: str):
    assert DynamoDBIndexConfig(metric_type=metric).parse_metric() == expected


def test_metric_parsing_rejects_unset():
    with pytest.raises(ValueError, match="Unsupported metric"):
        DynamoDBIndexConfig(metric_type=None).parse_metric()


def test_index_param_carries_partitions_and_search_param_empty():
    cfg = DynamoDBIndexConfig(metric_type=MetricType.COSINE)
    assert cfg.index_param() == {"num_partitions": 1}
    assert cfg.search_param() == {}


# ---------------------------------------------------------------------------
# Partition key (SearchSchema HASH)
# ---------------------------------------------------------------------------


def test_use_partition_key_toggles_on_count():
    assert DynamoDBIndexConfig(num_partitions=1).use_partition_key() is False
    assert DynamoDBIndexConfig(num_partitions=4).use_partition_key() is True


def test_search_schema_elements_add_hash_and_inline_filter():
    client = DynamoDBClient.__new__(DynamoDBClient)
    client.use_partition_key = True
    client.with_scalar_labels = True

    types = [e["SearchSchemaElementType"] for e in client._search_schema_elements()]
    assert types == ["HASH", "INLINE_FILTER"]

    client.use_partition_key = False
    client.with_scalar_labels = False
    assert client._search_schema_elements() == []


def test_prepare_filter_builds_equality_condition():
    client = DynamoDBClient.__new__(DynamoDBClient)

    client.prepare_filter(NonFilter())
    assert client._condition_expr is None
    assert client._expr_values == {}

    lbl = LabelFilter(label_percentage=0.2)
    client.prepare_filter(lbl)
    assert client._condition_expr == "label = :label"
    assert client._expr_values == {":label": {"S": lbl.label_value}}


def test_prepare_filter_rejects_numge():
    client = DynamoDBClient.__new__(DynamoDBClient)
    with pytest.raises(ValueError, match="Unsupported filter"):
        client.prepare_filter(NewIntFilter(int_value=100, filter_rate=0.01))


def test_build_item_sets_partition_and_label():
    client = DynamoDBClient.__new__(DynamoDBClient)
    client.use_partition_key = True
    client.num_partitions = 3
    client.with_scalar_labels = True

    item = client._build_item(7, [0.1, 0.2], ["lbl"], 0)
    assert item["pk"] == {"N": "7"}
    assert item["id"] == {"N": "7"}
    assert item["vector"] == {"L": [{"N": "0.1"}, {"N": "0.2"}]}
    assert item["part"] == {"N": "1"}  # 7 % 3
    assert item["label"] == {"S": "lbl"}


def test_partition_search_scopes_to_one_random_partition():
    """With a partition key, a search issues ONE scoped SearchVectors call
    (not a fan-out), filtered to a single partition value."""
    captured = {}

    class _SearchDDB:
        def search_vectors(self, **params):
            captured.update(params)
            return {"SearchResults": [{"Item": {"id": {"N": "5"}}, "Score": 0.1}]}

    client = DynamoDBClient.__new__(DynamoDBClient)
    client.client = _SearchDDB()
    client.table_name = "t"
    client.index_name = "i"
    client.use_partition_key = True
    client.num_partitions = 4
    client._condition_expr = None
    client._expr_values = {}

    out = client.search_embedding([0.1, 0.2], k=10)
    assert out == [5]
    # exactly one partition equality in the condition
    assert captured["SearchConditionExpression"].startswith("part = :part")
    assert "AND" not in captured["SearchConditionExpression"]  # no inline filter here
    part_val = int(captured["ExpressionAttributeValues"][":part"]["N"])
    assert 0 <= part_val < 4


# ---------------------------------------------------------------------------
# Throttling: UnprocessedItems backoff re-drive
# ---------------------------------------------------------------------------


class _FakeDDB:
    """Returns the given queue of UnprocessedItems maps, one per call."""

    def __init__(self, unprocessed_sequence: list[dict]):
        self._seq = list(unprocessed_sequence)
        self.calls = 0

    def batch_write_item(self, RequestItems: dict) -> dict:  # noqa: N803 (boto3 arg name)
        self.calls += 1
        up = self._seq.pop(0) if self._seq else {}
        return {"UnprocessedItems": up}


def test_write_batch_drains_after_throttled_redrives(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(ddb_mod.time, "sleep", lambda _s: None)  # no real waiting

    client = DynamoDBClient.__new__(DynamoDBClient)
    one = {"t": [{"PutRequest": {"Item": {"pk": {"N": "1"}}}}]}
    # first call -> 1 unprocessed, second -> 1 unprocessed, third -> drained
    client.client = _FakeDDB([one, one, {}])

    client._write_batch_with_retry(one)
    assert client.client.calls == 3  # 1 initial + 2 re-drives


def test_write_batch_gives_up_after_max_retries(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(ddb_mod.time, "sleep", lambda _s: None)

    client = DynamoDBClient.__new__(DynamoDBClient)
    one = {"t": [{"PutRequest": {"Item": {"pk": {"N": "1"}}}}]}
    # always throttled -> never drains
    client.client = _FakeDDB([one] * 50)

    with pytest.raises(RuntimeError, match="unprocessed items after"):
        client._write_batch_with_retry(one)
    assert client.client.calls == ddb_mod.UNPROCESSED_MAX_RETRIES + 1


# ---------------------------------------------------------------------------
# Config to_dict() contract
# ---------------------------------------------------------------------------


def test_to_dict_without_explicit_credentials():
    """Omitting keys must leave them None so boto3's default chain is used."""
    cfg = DynamoDBConfig(region_name="us-west-2", table_name="t", index_name="i")
    d = cfg.to_dict()
    assert d["region_name"] == "us-west-2"
    assert d["access_key_id"] is None
    assert d["secret_access_key"] is None
    assert d["session_token"] is None
    assert d["table_name"] == "t"
    assert d["index_name"] == "i"


def test_to_dict_unwraps_secrets():
    cfg = DynamoDBConfig(
        access_key_id=SecretStr("AKIA_EXAMPLE"),
        secret_access_key=SecretStr("secret_example"),
        session_token=SecretStr("token_example"),
    )
    d = cfg.to_dict()
    assert d["access_key_id"] == "AKIA_EXAMPLE"
    assert d["secret_access_key"] == "secret_example"  # noqa: S105
    assert d["session_token"] == "token_example"  # noqa: S105


# ---------------------------------------------------------------------------
# CLI wiring (static source introspection, no heavy runtime import)
# ---------------------------------------------------------------------------


def test_cli_defines_command_and_options():
    cli_src = Path("vectordb_bench/backend/clients/dynamodb/cli.py").read_text()
    tree = ast.parse(cli_src)

    class_names = {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
    func_names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}

    assert "DynamoDBTypedDict" in class_names
    assert "DynamoDBIndexTypedDict" in class_names
    assert "DynamoDB" in func_names

    def _fields(cls_name: str) -> set[str]:
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == cls_name:
                return {
                    i.target.id
                    for i in node.body
                    if isinstance(i, ast.AnnAssign) and isinstance(i.target, ast.Name)
                }
        return set()

    assert "partition_count" in _fields("DynamoDBTypedDict")
