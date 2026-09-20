import array
import logging
import pytest
from pydantic import SecretStr

from vectordb_bench.backend.clients.api import IndexType, MetricType
from vectordb_bench.backend.filter import Filter, FilterOp, IntFilter, LabelFilter
from vectordb_bench.backend.clients.oracle.config import (
    OracleConfig,
    OracleFlatConfig,
    OracleHNSWConfig,
    OracleIVFConfig,
    parse_oracle_metric,
)
from vectordb_bench.backend.clients.oracle.oracle import (
    Oracle,
    get_unique_index_name,
    validate_sql_identifier,
)

log = logging.getLogger(__name__)

# Connection details for local Docker Oracle 23ai instance
DB_CONFIG = {
    "user": "sys",
    "password": "password",
    "host": "localhost",
    "port": 1521,
    "service_name": "FREEPDB1",
    "sysdba": True,
}


def test_oracle_config():
    cfg = OracleConfig(
        user_name=SecretStr("vdbbench_user"),
        password=SecretStr("vdbbench_pass"),
        host="localhost",
        port=1521,
        service_name="FREEPDB1",
        table_name="test_col_123",
    )
    d = cfg.to_dict()
    assert d["user"] == "vdbbench_user"
    assert d["password"] == "vdbbench_pass"
    assert d["collection_name"] == "test_col_123"
    assert "SecretStr" not in str(d["password"])


def test_parse_oracle_metric():
    assert parse_oracle_metric(None) == "COSINE"
    assert parse_oracle_metric(MetricType.COSINE) == "COSINE"
    assert parse_oracle_metric(MetricType.L2) == "EUCLIDEAN"
    assert parse_oracle_metric(MetricType.IP) == "DOT"
    assert parse_oracle_metric(MetricType.DP) == "DOT"
    with pytest.raises(ValueError, match="Unsupported metric type"):
        parse_oracle_metric(MetricType.HAMMING)


def test_oracle_hnsw_config():
    c = OracleHNSWConfig(
        metric_type=MetricType.COSINE,
        neighbors=32,
        ef_construction=200,
        index_target_accuracy=95,
        search_target_accuracy=90,
    )
    idx_p = c.index_param()
    assert idx_p["index_type"] == IndexType.HNSW.value
    assert idx_p["neighbors"] == 32
    assert idx_p["ef_construction"] == 200
    assert idx_p["metric"] == "COSINE"

    sch_p = c.search_param()
    assert sch_p["search_target_accuracy"] == 90
    assert sch_p["metric"] == "COSINE"


def test_oracle_ivf_config():
    c = OracleIVFConfig(
        metric_type=MetricType.L2,
        neighbor_partitions=1024,
        samples_per_partition=10,
        min_vectors_per_partition=5,
        index_target_accuracy=90,
    )
    idx_p = c.index_param()
    assert idx_p["index_type"] == IndexType.IVFFlat.value
    assert idx_p["neighbor_partitions"] == 1024
    assert idx_p["metric"] == "EUCLIDEAN"


def test_oracle_flat_config():
    c = OracleFlatConfig(metric_type=MetricType.IP)
    idx_p = c.index_param()
    assert idx_p["index_type"] == IndexType.Flat.value
    assert idx_p["metric"] == "DOT"


def test_identifier_validation():
    assert validate_sql_identifier("valid_table_123") == "valid_table_123"
    with pytest.raises(ValueError):
        validate_sql_identifier("123invalid_start")
    with pytest.raises(ValueError):
        validate_sql_identifier("invalid;drop table")
    with pytest.raises(ValueError):
        validate_sql_identifier("a" * 130)


def test_unique_index_name():
    idx_name = get_unique_index_name("my_long_collection_name_table", "HNSW")
    assert idx_name.endswith("_hnsw_idx")
    assert len(idx_name) <= 128
    assert validate_sql_identifier(idx_name) == idx_name


def test_task_runner_contract_propagation():
    cfg = OracleConfig(
        user_name=SecretStr("vdbbench_user"),
        password=SecretStr("vdbbench_pass"),
        table_name="runner_table_test",
    )
    db_config_dict = cfg.to_dict()

    # Simulate TaskRunner init_db behavior
    collection_name = None
    if "collection_name" in db_config_dict and not collection_name:
        collection_name = db_config_dict.pop("collection_name")

    extra_db_kwargs = {}
    if collection_name:
        extra_db_kwargs["collection_name"] = collection_name

    assert extra_db_kwargs["collection_name"] == "runner_table_test"


@pytest.mark.integration
def test_oracle_flat_live():
    table_name = "test_oracle_flat"
    dim = 4
    db = Oracle(
        dim=dim,
        db_config={**DB_CONFIG, "collection_name": table_name},
        db_case_config=OracleFlatConfig(metric_type=MetricType.COSINE),
        collection_name=table_name,
        drop_old=True,
        with_scalar_labels=True,
    )

    embeddings = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    metadata = [1, 2, 3, 4]
    label_filter = LabelFilter(label_percentage=0.5)
    labels = ["cat", label_filter.label_value, "cat", label_filter.label_value]

    with db.init():
        count, err = db.insert_embeddings(embeddings, metadata, labels_data=labels)
        assert err is None
        assert count == 4

        # NonFilter search
        db.prepare_filter(Filter(type=FilterOp.NonFilter))
        res = db.search_embedding(query=[1.0, 0.0, 0.0, 0.0], k=2)
        assert len(res) == 2
        assert res[0] == 1

        # NumGE filter (id >= 3)
        db.prepare_filter(IntFilter(filter_op=FilterOp.NumGE, int_value=3))
        res_num = db.search_embedding(query=[0.0, 0.0, 1.0, 0.0], k=2)
        assert len(res_num) > 0
        assert all(idx >= 3 for idx in res_num)

        # StrEqual filter (label = label_filter.label_value)
        db.prepare_filter(label_filter)
        res_str = db.search_embedding(query=[0.0, 1.0, 0.0, 0.0], k=2)
        assert len(res_str) > 0
        assert set(res_str).issubset({2, 4})

        # Cleanup
        db._drop_table()


@pytest.mark.integration
def test_oracle_hnsw_live():
    table_name = "test_oracle_hnsw"
    dim = 8
    db = Oracle(
        dim=dim,
        db_config={**DB_CONFIG, "collection_name": table_name},
        db_case_config=OracleHNSWConfig(
            metric_type=MetricType.COSINE,
            neighbors=16,
            ef_construction=100,
            index_target_accuracy=95,
            search_target_accuracy=95,
        ),
        collection_name=table_name,
        drop_old=True,
        with_scalar_labels=False,
    )

    # Generate synthetic dataset (100 vectors)
    import random
    random.seed(42)
    embeddings = [[random.random() for _ in range(dim)] for _ in range(100)]
    metadata = list(range(1, 101))

    with db.init():
        count, err = db.insert_embeddings(embeddings, metadata)
        assert err is None
        assert count == 100

        # Build HNSW Vector Index
        db.optimize()

        # Query search
        db.prepare_filter(Filter(type=FilterOp.NonFilter))
        query_vec = embeddings[0]
        res = db.search_embedding(query=query_vec, k=10)
        assert len(res) == 10
        assert res[0] == 1  # Exact match top-1

        # Audit Execution Plan via DBMS_XPLAN
        query_array = array.array("f", query_vec)
        db.cursor.execute(
            f"EXPLAIN PLAN FOR SELECT id FROM {table_name} ORDER BY VECTOR_DISTANCE(embedding, :1, COSINE) FETCH APPROXIMATE FIRST 10 ROWS ONLY WITH TARGET ACCURACY 95",
            (query_array,)
        )
        db.cursor.execute("SELECT PLAN_TABLE_OUTPUT FROM TABLE(DBMS_XPLAN.DISPLAY())")
        plan_rows = [row[0] for row in db.cursor.fetchall()]
        plan_text = "\n".join(plan_rows)
        log.info(f"HNSW Execution Plan:\n{plan_text}")
        assert "VECTOR" in plan_text or "GRAPH" in plan_text or "INDEX" in plan_text

        # Cleanup
        db._drop_table()


@pytest.mark.integration
def test_oracle_ivf_live():
    table_name = "test_oracle_ivf"
    dim = 8
    db = Oracle(
        dim=dim,
        db_config={**DB_CONFIG, "collection_name": table_name},
        db_case_config=OracleIVFConfig(
            metric_type=MetricType.COSINE,
            neighbor_partitions=4,
            samples_per_partition=10,
            min_vectors_per_partition=2,
            index_target_accuracy=90,
            search_target_accuracy=90,
        ),
        collection_name=table_name,
        drop_old=True,
        with_scalar_labels=False,
    )

    import random
    random.seed(123)
    embeddings = [[random.random() for _ in range(dim)] for _ in range(100)]
    metadata = list(range(1, 101))

    with db.init():
        count, err = db.insert_embeddings(embeddings, metadata)
        assert err is None
        assert count == 100

        # Build IVF Vector Index
        db.optimize()

        # Query search
        db.prepare_filter(Filter(type=FilterOp.NonFilter))
        query_vec = embeddings[0]
        res = db.search_embedding(query=query_vec, k=10)
        assert len(res) == 10
        assert res[0] == 1  # Exact match top-1

        # Audit Execution Plan via DBMS_XPLAN
        query_array = array.array("f", query_vec)
        db.cursor.execute(
            f"EXPLAIN PLAN FOR SELECT id FROM {table_name} ORDER BY VECTOR_DISTANCE(embedding, :1, COSINE) FETCH APPROXIMATE FIRST 10 ROWS ONLY WITH TARGET ACCURACY 90",
            (query_array,)
        )
        db.cursor.execute("SELECT PLAN_TABLE_OUTPUT FROM TABLE(DBMS_XPLAN.DISPLAY())")
        plan_rows = [row[0] for row in db.cursor.fetchall()]
        plan_text = "\n".join(plan_rows)
        log.info(f"IVF Execution Plan:\n{plan_text}")
        # The optimizer may prefer a full scan on a tiny table, so assert the IVF
        # vector index exists rather than that this particular plan uses it.
        index_name = get_unique_index_name(table_name, IndexType.IVFFlat.value)
        db.cursor.execute(
            "SELECT COUNT(*) FROM user_indexes WHERE index_name = UPPER(:1)",
            (index_name,),
        )
        assert db.cursor.fetchone()[0] == 1

        # Cleanup
        db._drop_table()
