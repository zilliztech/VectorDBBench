import importlib.util
import os
from pathlib import Path
import pickle
import random
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
from click.testing import CliRunner
from pydantic import ValidationError

from vectordb_bench.backend.clients import DB
from vectordb_bench.backend.clients.api import IndexType, MetricType
from vectordb_bench.backend.clients.zvec.config import ZvecDiskANNIndexConfig, ZvecHNSWIndexConfig
from vectordb_bench.backend.filter import non_filter


def test_diskann_config_and_registry_roundtrip():
    config = ZvecDiskANNIndexConfig(metric_type=MetricType.COSINE, pq_chunk_num=96)
    assert config.index_param() == {"max_degree": 64, "list_size": 100, "pq_chunk_num": 96}
    assert config.search_param() == {"index": IndexType.DISKANN, "list_size": 300}
    assert DB.Zvec.case_config_cls(IndexType.DISKANN) is ZvecDiskANNIndexConfig
    saved = config.model_dump(mode="json")
    assert DB.Zvec.case_config_cls(saved["index"])(**saved) == config
    assert pickle.loads(pickle.dumps(config)) == config
    assert DB.Zvec.case_config_cls() is ZvecHNSWIndexConfig
    assert DB.Zvec.case_config_cls(IndexType.HNSW) is ZvecHNSWIndexConfig


@pytest.mark.parametrize(
    "values",
    [
        {"max_degree": 0},
        {"max_degree": 101},
        {"build_list_size": 9},
        {"build_list_size": 101},
        {"search_list_size": 0},
        {"pq_chunk_num": -1},
        {"pq_chunk_num": 1025},
        {"index": "HNSW"},
    ],
)
def test_diskann_config_rejects_unsupported_values(values):
    with pytest.raises(ValidationError):
        ZvecDiskANNIndexConfig(**values)


@pytest.fixture
def adapter(monkeypatch):
    sdk = ModuleType("zvec")
    sdk.IndexType = SimpleNamespace(DISKANN="DISKANN", HNSW="HNSW")
    sdk.MetricType = SimpleNamespace(IP="IP", COSINE="COSINE", L2="L2")
    sdk.QuantizeType = SimpleNamespace(UNDEFINED="UNDEFINED", INT8="INT8", FP16="FP16", INT4="INT4")
    sdk.DataType = SimpleNamespace(INT64="INT64", STRING="STRING", VECTOR_FP32="VECTOR_FP32")
    sdk.LogLevel = SimpleNamespace(WARN="WARN")
    sdk.init = Mock()
    sdk.CollectionOption = lambda **kwargs: SimpleNamespace(**kwargs)
    sdk.OptimizeOption = lambda: SimpleNamespace()
    sdk.InvertIndexParam = lambda **kwargs: SimpleNamespace(**kwargs)
    sdk.FieldSchema = lambda *args, **kwargs: SimpleNamespace()
    sdk.VectorSchema = lambda name, data_type, **kwargs: SimpleNamespace(name=name, data_type=data_type, **kwargs)
    sdk.CollectionSchema = lambda **kwargs: SimpleNamespace(**kwargs)
    sdk.Doc = lambda **kwargs: SimpleNamespace(**kwargs)
    sdk.VectorQuery = lambda **kwargs: SimpleNamespace(**kwargs)
    sdk.DiskAnnIndexParam = lambda **kwargs: SimpleNamespace(type="DISKANN", **kwargs)
    sdk.DiskAnnQueryParam = lambda **kwargs: SimpleNamespace(type="DISKANN", **kwargs)
    sdk.HnswIndexParam = lambda **kwargs: SimpleNamespace(type="HNSW", **kwargs)
    sdk.HnswQueryParam = lambda **kwargs: SimpleNamespace(type="HNSW", **kwargs)
    sdk.open = Mock()
    sdk.create_and_open = Mock()
    monkeypatch.setitem(sys.modules, "zvec", sdk)
    source = Path(__file__).parents[1] / "vectordb_bench/backend/clients/zvec/zvec.py"
    spec = importlib.util.spec_from_file_location("vectordb_bench.backend.clients.zvec._unit_adapter", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Zvec, sdk


def make_collection(adapter, *, config=None, count=100, completeness=1.0, dimension=768):
    client, _sdk = adapter
    config = config or ZvecDiskANNIndexConfig(pq_chunk_num=96)
    field = SimpleNamespace(dimension=dimension, index_param=client._parse_index_param(config))
    collection = Mock()
    collection.schema.vector.return_value = field
    collection.stats = SimpleNamespace(doc_count=count, index_completeness={"dense": completeness})
    return collection


def make_client(adapter, config=None):
    client, _sdk = adapter
    instance = object.__new__(client)
    instance.case_config = config or ZvecDiskANNIndexConfig(pq_chunk_num=96)
    instance.table_name = "test"
    instance.dim = 768
    instance.path = "unused"
    instance.with_scalar_labels = False
    instance._scalar_label_field = "label"
    instance.query_param = client._parse_query_param(instance.case_config)
    instance.option = SimpleNamespace(read_only=False, enable_mmap=True)
    instance.expr = ""
    return instance


def test_sdk_parameter_mapping_does_not_conflate_build_and_query(adapter):
    client, sdk = adapter
    config = ZvecDiskANNIndexConfig(metric_type=MetricType.COSINE, pq_chunk_num=96)
    index = client._parse_index_param(config)
    query = client._parse_query_param(config)
    assert index.type == sdk.IndexType.DISKANN
    assert index.metric_type == sdk.MetricType.COSINE
    assert index.quantize_type == sdk.QuantizeType.UNDEFINED
    assert (index.max_degree, index.list_size, index.pq_chunk_num) == (64, 100, 96)
    assert query.list_size == 300
    hnsw = ZvecHNSWIndexConfig(M=50, ef_construction=500, ef_search=118, quantize_type="int8", is_using_refiner=True)
    index = client._parse_index_param(hnsw)
    query = client._parse_query_param(hnsw)
    assert (index.type, index.m, index.ef_construction, index.quantize_type) == ("HNSW", 50, 500, "INT8")
    assert query.ef == 118 and query.is_using_refiner
    assert index.metric_type == sdk.MetricType.IP


@pytest.mark.parametrize(
    "symbol, method",
    [
        ("DiskAnnIndexParam", "_parse_index_param"),
        ("DiskAnnQueryParam", "_parse_query_param"),
    ],
)
def test_diskann_reports_missing_sdk_support(adapter, monkeypatch, symbol, method):
    client, sdk = adapter
    monkeypatch.delattr(sdk, symbol)
    with pytest.raises(ValueError, match=symbol):
        getattr(client, method)(ZvecDiskANNIndexConfig())
    assert client._parse_query_param(ZvecHNSWIndexConfig()).type == "HNSW"


def test_diskann_creation_rejects_hnsw_directory_before_delete(adapter, tmp_path):
    client, sdk = adapter
    collection = make_collection(adapter)
    collection.schema.vector.return_value.index_param.type = "HNSW"
    sdk.open.return_value = collection
    with pytest.raises(ValueError, match="non-DiskANN"):
        client(768, {"path": str(tmp_path)}, ZvecDiskANNIndexConfig(), drop_old=True)
    collection.destroy.assert_not_called()
    collection.close.assert_called_once()
    sdk.create_and_open.assert_not_called()


def test_diskann_rebuild_allows_changed_build_parameters(adapter, tmp_path):
    client, sdk = adapter
    collection = make_collection(adapter)

    def mark_destroyed():
        collection.close.side_effect = ValueError("collection is already destroyed.")

    collection.destroy.side_effect = mark_destroyed
    sdk.open.return_value = collection
    config = ZvecDiskANNIndexConfig(max_degree=32, pq_chunk_num=48)
    client(768, {"path": str(tmp_path)}, config, drop_old=True)
    assert not sdk.open.call_args.kwargs["option"].read_only
    collection.destroy.assert_called_once()
    collection.close.assert_not_called()
    sdk.create_and_open.assert_called_once()
    sdk.create_and_open.return_value.close.assert_called_once()


def test_diskann_failed_destroy_closes_collection_without_recreating(adapter, tmp_path):
    client, sdk = adapter
    collection = make_collection(adapter)
    collection.destroy.side_effect = ValueError("destroy failed")
    sdk.open.return_value = collection
    with pytest.raises(ValueError, match="destroy failed"):
        client(768, {"path": str(tmp_path)}, ZvecDiskANNIndexConfig(), drop_old=True)
    collection.destroy.assert_called_once()
    collection.close.assert_called_once()
    sdk.create_and_open.assert_not_called()


def test_diskann_validates_pq_dimensions_before_open(adapter, tmp_path):
    client, sdk = adapter
    with pytest.raises(ValueError, match="pq_chunk_num"):
        client(8, {"path": str(tmp_path)}, ZvecDiskANNIndexConfig(pq_chunk_num=96), drop_old=True)
    sdk.open.assert_not_called()
    sdk.create_and_open.assert_not_called()


def test_diskann_new_directory_and_search_only_reopen(adapter, tmp_path):
    client, sdk = adapter
    path = tmp_path / "diskann"
    config = ZvecDiskANNIndexConfig(pq_chunk_num=96)
    with pytest.raises(ValueError, match="does not exist"):
        client(768, {"path": str(path)}, config, drop_old=False)
    client(768, {"path": str(path)}, config, drop_old=True)
    sdk.create_and_open.assert_called_once()
    sdk.create_and_open.return_value.close.assert_called_once()
    path.mkdir()
    collection = make_collection(adapter, config=config)
    sdk.open.return_value = collection
    client(768, {"path": str(path)}, config, drop_old=False)
    assert sdk.open.call_args.kwargs["option"].read_only
    collection.destroy.assert_not_called()
    collection.close.assert_called_once()


@pytest.mark.parametrize("change", ["metric", "pq", "degree", "dimension", "incomplete", "empty"])
def test_diskann_search_only_checks_persisted_schema_and_readiness(adapter, tmp_path, change):
    client, sdk = adapter
    collection = make_collection(adapter)
    field = collection.schema.vector.return_value
    if change == "metric":
        field.index_param.metric_type = "COSINE"
    elif change == "pq":
        field.index_param.pq_chunk_num = 48
    elif change == "degree":
        field.index_param.max_degree = 32
    elif change == "dimension":
        field.dimension = 128
    elif change == "incomplete":
        collection.stats.index_completeness["dense"] = 0.5
    else:
        collection.stats.doc_count = 0
    sdk.open.return_value = collection
    with pytest.raises((ValueError, RuntimeError)):
        client(768, {"path": str(tmp_path)}, ZvecDiskANNIndexConfig(pq_chunk_num=96), drop_old=False)
    collection.close.assert_called_once()
    collection.destroy.assert_not_called()


def test_diskann_session_closes_on_error(adapter):
    instance = make_client(adapter)
    collection = make_collection(adapter)
    adapter[1].open.return_value = collection
    with pytest.raises(RuntimeError, match="test body"):
        with instance.init():
            raise RuntimeError("test body")
    collection.close.assert_called_once()
    assert instance.collection is None


def test_diskann_insert_checks_partial_and_malformed_statuses(adapter):
    instance = make_client(adapter)
    instance.collection = Mock()
    good = SimpleNamespace(ok=lambda: True)
    bad = SimpleNamespace(ok=lambda: False)
    instance.collection.insert.return_value = [good, bad]
    count, error = instance.insert_embeddings([[0.0], [1.0]], [10, 11])
    assert count == 1 and error.non_retryable and error.inserted_count == 1
    instance.collection.insert.return_value = [good]
    count, error = instance.insert_embeddings([[0.0], [1.0]], [10, 11])
    assert count == 0 and error.non_retryable
    instance.collection.insert.return_value = [good, good]
    assert instance.insert_embeddings([[0.0], [1.0]], [10, 11]) == (2, None)
    count, error = instance.insert_embeddings([[0.0]], [10, 11])
    assert count == 0 and error.non_retryable


def test_diskann_optimize_requires_exact_count_and_index_completion(adapter):
    instance = make_client(adapter)
    instance.collection = make_collection(adapter, count=99, completeness=0.0)
    with pytest.raises(RuntimeError, match="expected 100"):
        instance.optimize(data_size=100)
    instance.collection.optimize.assert_not_called()
    instance.collection.stats.doc_count = 100
    with pytest.raises(RuntimeError, match="not fully indexed"):
        instance.optimize(data_size=100)
    instance.collection.optimize.side_effect = lambda **_kwargs: instance.collection.stats.index_completeness.update(
        dense=1.0
    )
    instance.optimize(data_size=100)
    assert instance.collection.stats.index_completeness["dense"] == 1.0


def test_search_passes_diskann_query_parameters_without_refiner(adapter):
    instance = make_client(adapter)
    instance.collection = Mock()
    instance.collection.query.return_value = [SimpleNamespace(id="7"), SimpleNamespace(id="8")]
    assert instance.search_embedding([0.1] * 768, k=2) == [7, 8]
    arguments = instance.collection.query.call_args.kwargs
    assert arguments["vectors"].param.type == "DISKANN"
    assert arguments["vectors"].param.list_size == 300
    assert arguments["topk"] == 2 and arguments["output_fields"] == []


def test_cli_defaults_and_diskann_selection(monkeypatch):
    from vectordb_bench.backend.clients.zvec import cli as zvec_cli

    captured = {}
    monkeypatch.setattr(zvec_cli, "run", lambda **values: captured.update(values))
    runner = CliRunner()
    result = runner.invoke(zvec_cli.Zvec, ["--path", "unused", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert isinstance(captured["db_case_config"], ZvecHNSWIndexConfig)
    assert captured["db_case_config"] == ZvecHNSWIndexConfig()
    result = runner.invoke(
        zvec_cli.Zvec,
        [
            "--path",
            "unused",
            "--index-type",
            "diskann",
            "--metric-type",
            "cosine",
            "--max-degree",
            "64",
            "--build-list-size",
            "100",
            "--pq-chunk-num",
            "96",
            "--search-list-size",
            "400",
            "--case-type",
            "Performance768D10M",
            "--skip-drop-old",
            "--skip-load",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    config = captured["db_case_config"]
    assert isinstance(config, ZvecDiskANNIndexConfig)
    assert config.metric_type == MetricType.COSINE
    assert config.search_list_size == 400 and config.build_list_size == 100
    assert captured["drop_old"] is False and captured["load"] is False


@pytest.mark.parametrize(
    "arguments",
    [
        ["--index-type", "diskann", "--ef-search", "118"],
        ["--index-type", "diskann", "--m", "50"],
        ["--index-type", "diskann", "--quantize-type", "int8"],
        ["--index-type", "diskann", "--is-using-refiner"],
        ["--index-type", "hnsw", "--pq-chunk-num", "96"],
        ["--index-type", "diskann", "--max-degree", "101"],
        ["--index-type", "diskann", "--search-list-size", "0"],
    ],
)
def test_cli_rejects_mixed_or_invalid_parameters(monkeypatch, arguments):
    from vectordb_bench.backend.clients.zvec import cli as zvec_cli

    run = Mock()
    monkeypatch.setattr(zvec_cli, "run", run)
    result = CliRunner().invoke(zvec_cli.Zvec, ["--path", "unused", "--dry-run", *arguments])
    assert result.exit_code == 2, result.output
    run.assert_not_called()


@pytest.mark.skipif(
    sys.platform != "linux" or os.getenv("ZVEC_NATIVE_TESTS") != "1",
    reason="Opt-in Linux DiskANN smoke test: ZVEC_NATIVE_TESTS=1",
)
def test_native_diskann_build_reopen_and_search(tmp_path):
    import zvec as sdk

    from vectordb_bench.backend.clients.zvec.zvec import Zvec

    if not hasattr(sdk, "DiskAnnIndexParam"):
        pytest.fail("Installed zvec must support DiskAnnIndexParam")
    if not callable(getattr(sdk.Collection, "close", None)):
        pytest.fail("Installed zvec must support Collection.close()")
    random_values = random.Random(71)
    vectors = [[random_values.random() for _ in range(32)] for _ in range(1200)]
    config = ZvecDiskANNIndexConfig(metric_type=MetricType.L2, max_degree=32, pq_chunk_num=8)
    connection = {"path": str(tmp_path / "diskann")}
    client = Zvec(32, connection, config, drop_old=True)
    with client.init():
        for start in range(0, len(vectors), 100):
            batch = vectors[start : start + 100]
            count, error = client.insert_embeddings(batch, list(range(start, start + len(batch))))
            assert error is None and count == len(batch)
        client.optimize(data_size=len(vectors))
        assert client.collection.schema.vector("dense").index_param.type == sdk.IndexType.DISKANN
    reopened = Zvec(32, connection, config, drop_old=False)
    reopened.prepare_filter(non_filter)
    with reopened.init():
        assert reopened.collection.stats.doc_count == len(vectors)
        result = reopened.search_embedding(vectors[17], k=10)
        assert len(result) == 10 and len(set(result)) == 10
        assert 17 in result
    rebuilt = Zvec(32, connection, config, drop_old=True)
    with rebuilt.init():
        assert rebuilt.collection.stats.doc_count == 0
