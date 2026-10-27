from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from glide_sync import Batch
from pydantic import ValidationError

from vectordb_bench.backend.clients import DB
from vectordb_bench.backend.clients.api import IndexType, MetricType
from vectordb_bench.backend.clients.valkey.config import ValkeyConfig, ValkeyHNSWConfig
from vectordb_bench.backend.clients.valkey.valkey import Valkey
from vectordb_bench.backend.filter import IntFilter, LabelFilter, non_filter


def command_name(command: Any) -> str:
    command = getattr(command, "value", command)
    return command.decode() if isinstance(command, bytes) else str(command)


class FakeClusterCursor:
    def __init__(self):
        self.finished = False

    def is_finished(self):
        return self.finished


class FakeClient:
    def __init__(self):
        self.created = False
        self.closed = False
        self.custom_commands = []
        self.exec_batches = []
        self.scan_keys = []

    def custom_command(self, args: list[Any]):
        self.custom_commands.append(args)
        command = command_name(args[0])
        if command == "FT._LIST":
            return [b"vdbbench_valkey"] if self.created else []
        if command == "FT.CREATE":
            self.created = True
            return b"OK"
        if command == "FT.DROPINDEX":
            self.created = False
            return b"OK"
        if command == "FT.SEARCH":
            return [1, {b"vdbbench_valkey:7": {}}]
        msg = f"Unexpected command: {args}"
        raise AssertionError(msg)

    def exec(self, batch: Any, raise_on_error: bool):
        assert raise_on_error is True
        self.exec_batches.append(batch.commands)
        return [1] * len(batch.commands)

    def scan(self, cursor: Any, match: str, count: int):
        assert match == "vdbbench_valkey:*"
        assert count > 0
        keys, self.scan_keys = self.scan_keys, []
        if isinstance(cursor, FakeClusterCursor):
            cursor.finished = True
            return [cursor, keys]
        return [b"0", keys]

    def close(self):
        self.closed = True


def make_adapter():
    setup_client = FakeClient()
    runtime_client = FakeClient()
    config = ValkeyHNSWConfig(M=24, efConstruction=300, ef=40)
    db_config = {
        "host": "localhost",
        "port": 6379,
        "password": None,
        "ssl": False,
        "insecure_tls": False,
        "request_timeout_ms": 12_000,
        "connection_timeout_ms": 13_000,
        "cmd": True,
    }
    with patch(
        "vectordb_bench.backend.clients.valkey.valkey.GlideClient.create",
        side_effect=[setup_client, runtime_client],
    ) as create:
        adapter = Valkey(dim=3, db_config=db_config, db_case_config=config)
        context = adapter.init()
        context.__enter__()
    return adapter, context, setup_client, runtime_client, create


def test_valkey_registration_and_config():
    assert DB.Valkey.value == "Valkey"
    assert DB.Valkey.init_cls is Valkey
    assert DB.Valkey.config_cls is ValkeyConfig
    assert DB.Valkey.case_config_cls(IndexType.HNSW) is ValkeyHNSWConfig
    assert DB.Valkey.case_config_cls(IndexType.AUTOINDEX) is ValkeyHNSWConfig
    assert Valkey.supports_full_text_search() is False

    config = ValkeyConfig(host="localhost")
    assert config.to_dict()["password"] is None
    assert config.to_dict()["port"] == 6379
    assert config.to_dict()["ssl"] is True
    assert config.to_dict()["insecure_tls"] is False
    assert config.to_dict()["cmd"] is False
    assert config.to_dict()["request_timeout_ms"] == 600_000
    assert config.to_dict()["connection_timeout_ms"] == 10_000
    assert config.to_dict()["collection_name"] == "vdbbench_valkey"

    assert ValkeyHNSWConfig().model_dump() == {
        "metric_type": None,
        "M": 16,
        "efConstruction": 200,
        "ef": 10,
        "index": IndexType.HNSW,
    }
    with pytest.raises(ValidationError):
        ValkeyHNSWConfig(M=0, efConstruction=200)
    with pytest.raises(ValidationError):
        ValkeyHNSWConfig(M=16, efConstruction=200, ef=0)
    with pytest.raises(ValidationError):
        ValkeyConfig(host="localhost", collection_name="unsafe*")
    with pytest.raises(ValidationError):
        ValkeyConfig(host="localhost", port=0)
    with pytest.raises(ValidationError):
        ValkeyConfig(host="localhost", request_timeout_ms=0)
    with pytest.raises(ValidationError):
        ValkeyConfig(host="localhost", connection_timeout_ms=0)
    with pytest.raises(ValidationError, match="requires ssl"):
        ValkeyConfig(host="localhost", ssl=False, insecure_tls=True)


def test_valkey_index_insert_and_search():
    adapter, context, setup_client, runtime_client, create = make_adapter()
    try:
        assert setup_client.closed is True
        assert setup_client.created is True
        create_args = next(args for args in setup_client.custom_commands if command_name(args[0]) == "FT.CREATE")
        assert create_args[1:7] == [
            "vdbbench_valkey",
            "ON",
            "HASH",
            "PREFIX",
            "1",
            "vdbbench_valkey:",
        ]

        glide_config = create.call_args_list[0].args[0]
        assert glide_config.addresses[0].host == "localhost"
        assert glide_config.addresses[0].port == 6379
        assert glide_config.use_tls is False
        assert glide_config.request_timeout == 12_000
        assert glide_config.advanced_config.connection_timeout == 13_000
        assert glide_config.advanced_config.tls_config is None
        assert glide_config.database_id == 0

        embeddings = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        inserted, error = adapter.insert_embeddings(embeddings, [1, 2])
        assert (inserted, error) == (2, None)
        assert len(runtime_client.exec_batches) == 1
        assert len(runtime_client.exec_batches[0]) == 2
        assert runtime_client.exec_batches[0][0][1][0] == "vdbbench_valkey:1"
        assert runtime_client.exec_batches[0][0][1][-1] == np.asarray(embeddings[0], dtype=np.float32).tobytes()

        adapter.prepare_filter(non_filter)
        assert adapter.search_embedding([1.0, 2.0, 3.0], k=5) == [7]
        search_args = next(args for args in runtime_client.custom_commands if command_name(args[0]) == "FT.SEARCH")
        assert search_args[:3] == [
            "FT.SEARCH",
            "vdbbench_valkey",
            "*=>[KNN 5 @vector $vec EF_RUNTIME 40]",
        ]
        assert "NOCONTENT" in search_args
        assert search_args[-2:] == ["DIALECT", "2"]
    finally:
        context.__exit__(None, None, None)

    assert runtime_client.closed is True


def test_valkey_validates_insert_data():
    adapter, context, _, _, _ = make_adapter()
    try:
        with pytest.raises(ValueError, match="same length"):
            adapter.insert_embeddings([[1.0, 2.0, 3.0]], [])
    finally:
        context.__exit__(None, None, None)

    adapter, context, _, _, _ = make_adapter()
    adapter.with_scalar_labels = True
    try:
        with pytest.raises(ValueError, match="Scalar labels"):
            adapter.insert_embeddings([[1.0, 2.0, 3.0]], [1])
    finally:
        context.__exit__(None, None, None)


def test_valkey_filters_and_metrics():
    adapter, context, _, runtime_client, _ = make_adapter()
    try:
        adapter.prepare_filter(IntFilter(filter_rate=0.5, int_value=42))
        adapter.search_embedding([1.0, 2.0, 3.0])
        assert runtime_client.custom_commands[-1][2].startswith("@metadata:[42 +inf]=>[")

        adapter.prepare_filter(LabelFilter(label_percentage=0.1))
        adapter.search_embedding([1.0, 2.0, 3.0])
        assert runtime_client.custom_commands[-1][2].startswith("@label:{label_10p}=>[")
    finally:
        context.__exit__(None, None, None)

    config_kwargs = {"M": 16, "efConstruction": 200}
    assert ValkeyHNSWConfig(metric_type=MetricType.COSINE, **config_kwargs).parse_metric() == "COSINE"
    assert ValkeyHNSWConfig(metric_type=MetricType.L2, **config_kwargs).parse_metric() == "L2"
    assert ValkeyHNSWConfig(metric_type=MetricType.IP, **config_kwargs).parse_metric() == "IP"
    with pytest.raises(ValueError, match="Unsupported metric type"):
        ValkeyHNSWConfig(metric_type=MetricType.HAMMING, **config_kwargs).parse_metric()


def test_valkey_cluster_client_and_drop_old(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("vectordb_bench.backend.clients.valkey.valkey._DROP_CHUNK_SIZE", 1)
    client = FakeClient()
    client.created = True
    client.scan_keys = [b"vdbbench_valkey:1", b"vdbbench_valkey:2"]
    db_config = {
        "host": "localhost",
        "port": 6379,
        "password": None,
        "ssl": True,
        "insecure_tls": True,
        "request_timeout_ms": 20_000,
        "connection_timeout_ms": 21_000,
        "cmd": False,
    }

    with (
        patch(
            "vectordb_bench.backend.clients.valkey.valkey.GlideClusterClient.create",
            return_value=client,
        ) as create,
        patch(
            "vectordb_bench.backend.clients.valkey.valkey.ClusterScanCursor",
            FakeClusterCursor,
        ),
    ):
        Valkey(
            dim=3,
            db_config=db_config,
            db_case_config=ValkeyHNSWConfig(M=16, efConstruction=200, ef=10),
            drop_old=True,
        )

    assert client.closed is True
    commands = [command_name(args[0]) for args in client.custom_commands]
    assert commands == ["FT._LIST", "FT.DROPINDEX", "FT._LIST", "FT.CREATE"]
    assert client.custom_commands[1] == ["FT.DROPINDEX", "vdbbench_valkey"]
    assert len(client.exec_batches) == 2
    assert all(len(batch) == 1 for batch in client.exec_batches)
    unlink_command = Batch(is_atomic=False).unlink(["key"]).commands[0][0]
    assert all(batch[0][0] == unlink_command for batch in client.exec_batches)

    glide_config = create.call_args.args[0]
    assert glide_config.use_tls is True
    assert glide_config.request_timeout == 20_000
    assert glide_config.advanced_config.connection_timeout == 21_000
    assert glide_config.advanced_config.tls_config.use_insecure_tls is True
    assert glide_config.database_id is None


def test_valkey_uses_config_defaults():
    client = FakeClient()
    with patch(
        "vectordb_bench.backend.clients.valkey.valkey.GlideClient.create",
        return_value=client,
    ) as create:
        Valkey(
            dim=3,
            db_config=ValkeyConfig(host="localhost", cmd=True).to_dict(),
            db_case_config=ValkeyHNSWConfig(M=16, efConstruction=200, ef=10),
        )

    glide_config = create.call_args.args[0]
    assert glide_config.use_tls is True
    assert glide_config.request_timeout == 600_000
    assert glide_config.advanced_config.connection_timeout == 10_000
    assert glide_config.advanced_config.tls_config is None


def test_valkey_checks_index_list():
    adapter = object.__new__(Valkey)
    adapter.collection_name = "vdbbench_valkey"
    client = FakeClient()

    assert adapter._index_exists(client) is False
    client.created = True
    assert adapter._index_exists(client) is True
