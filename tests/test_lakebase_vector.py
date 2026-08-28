"""Offline tests for the Lakebase Vector config, client, and metric assembly path.

Usage:
  pytest tests/test_lakebase_vector.py -v
"""

from __future__ import annotations

import importlib
import pickle
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, call

import numpy as np
import pytest
from pydantic import SecretStr

from vectordb_bench.backend.assembler import Assembler
from vectordb_bench.backend.cases import CaseType
from vectordb_bench.backend.clients import DB
from vectordb_bench.backend.clients.api import MetricType
from vectordb_bench.backend.clients.lakebase_vector.config import LakebaseANNConfig, LakebaseVectorConfig
from vectordb_bench.backend.data_source import DatasetSource
from vectordb_bench.backend.filter import Filter, IntFilter, LabelFilter, NonFilter
from vectordb_bench.models import CaseConfig, TaskConfig

if TYPE_CHECKING:
    from vectordb_bench.backend.clients.lakebase_vector.lakebase_vector import LakebaseVector

DB_CONFIG = {
    "connect_config": {
        "host": "localhost",
        "port": 5432,
        "dbname": "vectordb",
        "user": "vectordb",
        "password": "vectordb",
    },
    "table_name": "test_lakebase_vector",
}

DIM = 128


# ── Helpers ──────────────────────────────────────────────────────────────────


def make_ann_config(**overrides) -> LakebaseANNConfig:
    values = {
        "metric_type": MetricType.COSINE,
        "probes": None,
        "epsilon": None,
        "max_parallel_workers": None,
    }
    values.update(overrides)
    return LakebaseANNConfig(**values)


def lakebase_client_cls():
    pytest.importorskip("psycopg")
    pytest.importorskip("pgvector")
    from vectordb_bench.backend.clients.lakebase_vector.lakebase_vector import LakebaseVector  # noqa: PLC0415

    return LakebaseVector


def patch_sql_rendering(monkeypatch: pytest.MonkeyPatch):
    client_cls = lakebase_client_cls()
    client_module = importlib.import_module(client_cls.__module__)
    original_as_string = client_module.sql.Composed.as_string
    monkeypatch.setattr(
        client_module.sql.Composed,
        "as_string",
        lambda composed, _context=None: original_as_string(composed),
    )
    return original_as_string


def make_db(
    table_name: str = "test_lakebase_vector",
    drop_old: bool = True,
    *,
    case_config: LakebaseANNConfig | None = None,
    with_scalar_labels: bool = False,
) -> LakebaseVector:
    config = dict(DB_CONFIG)
    config["connect_config"] = dict(DB_CONFIG["connect_config"])
    config["table_name"] = table_name
    return DB.LakebaseVector.init_cls(
        dim=DIM,
        db_config=config,
        db_case_config=case_config or make_ann_config(),
        drop_old=drop_old,
        with_scalar_labels=with_scalar_labels,
    )


@pytest.fixture
def mocked_db_connection(monkeypatch: pytest.MonkeyPatch) -> tuple[MagicMock, MagicMock]:
    client_cls = lakebase_client_cls()
    conn = MagicMock(name="connection")
    cursor = MagicMock(name="cursor")
    monkeypatch.setattr(client_cls, "_create_connection", staticmethod(lambda **_kwargs: (conn, cursor)))
    return conn, cursor


class TestLakebaseVectorConfig:
    def test_connection_config(self):
        config = LakebaseVectorConfig(
            user_name=SecretStr("lakebase-user"),
            password=SecretStr("lakebase-password"),
            host="lakebase.example.com",
            port=6432,
            db_name="benchmark",
            table_name="vectors",
        )

        assert config.to_dict() == {
            "connect_config": {
                "host": "lakebase.example.com",
                "port": 6432,
                "dbname": "benchmark",
                "user": "lakebase-user",
                "password": "lakebase-password",
            },
            "table_name": "vectors",
        }


@pytest.mark.parametrize(
    ("case_type", "expected_metric_type"),
    [
        (CaseType.Performance768D100M, MetricType.L2),
        (CaseType.Performance1536D50K, MetricType.COSINE),
    ],
)
def test_dataset_metric(
    case_type: CaseType,
    expected_metric_type: MetricType,
):
    db_case_config = LakebaseANNConfig()
    task = TaskConfig(
        db=DB.LakebaseVector,
        db_config=LakebaseVectorConfig(password=SecretStr("test-password"), db_name="test-db"),
        db_case_config=db_case_config,
        case_config=CaseConfig(case_id=case_type),
    )

    assert db_case_config.metric_type is None

    runner = Assembler.assemble("test-run", task, DatasetSource.S3)

    assert runner.config.db_case_config.metric_type == expected_metric_type
    assert runner.config.db_case_config.index_param()["metric"] is not None


class TestLakebaseVectorClient:
    @pytest.mark.parametrize(
        ("metric_type", "operator_class", "search_operator"),
        [
            (MetricType.L2, "vector_l2_ops", "<->"),
            (MetricType.IP, "vector_ip_ops", "<#>"),
            (MetricType.COSINE, "vector_cosine_ops", "<=>"),
        ],
    )
    def test_index_sql(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mocked_db_connection: tuple[MagicMock, MagicMock],
        metric_type: MetricType,
        operator_class: str,
        search_operator: str,
    ) -> None:
        conn, cursor = mocked_db_connection
        render_sql = patch_sql_rendering(monkeypatch)
        db = make_db(
            "test_create_index",
            drop_old=False,
            case_config=make_ann_config(metric_type=metric_type, max_parallel_workers=16),
        )
        db.conn = conn
        db.cursor = cursor
        conn.reset_mock()
        cursor.reset_mock()
        monkeypatch.setattr(db, "_set_parallel_index_build_param", MagicMock())

        db._create_index()

        query = cursor.execute.call_args.args[0]
        assert render_sql(query) == (
            'CREATE INDEX IF NOT EXISTS "lakebase_vector_index" '
            'ON public."test_create_index" USING lakebase_ann '
            f'("embedding" {operator_class})'
        )
        assert db.case_config.search_param() == {"metric_fun_op": search_operator}
        assert db.case_config.index_param()["max_parallel_workers"] == 16
        conn.commit.assert_called_once_with()

    def test_optimize(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mocked_db_connection: tuple[MagicMock, MagicMock],
    ) -> None:
        db = make_db("test_optimize", drop_old=False)
        lifecycle = MagicMock()
        monkeypatch.setattr(db, "_drop_index", lifecycle.drop_index)
        monkeypatch.setattr(db, "_create_index", lifecycle.create_index)

        db.optimize()

        assert lifecycle.mock_calls == [call.drop_index(), call.create_index()]

    # The two probes cases configure single-level and two-level IVF, respectively.
    @pytest.mark.parametrize("probes", ["10", "10,20"])
    def test_session_guc(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mocked_db_connection: tuple[MagicMock, MagicMock],
        probes: str,
    ) -> None:
        conn, cursor = mocked_db_connection
        render_sql = patch_sql_rendering(monkeypatch)
        case_config = make_ann_config(probes=probes, epsilon=1.5)
        db = make_db(
            "test_session_gucs",
            drop_old=False,
            case_config=case_config,
        )
        conn.reset_mock()
        cursor.reset_mock()

        with db.init():
            assert db.conn is conn
            assert db.cursor is cursor

        commands = [render_sql(execute_call.args[0]) for execute_call in cursor.execute.call_args_list]
        assert commands == [
            f'SET "lakebase_ann.probes" = "{probes}";',
            'SET "lakebase_ann.epsilon" = "1.5";',
        ]
        conn.commit.assert_called_once_with()
        cursor.close.assert_called_once_with()
        conn.close.assert_called_once_with()
        assert db.conn is None
        assert db.cursor is None

    def test_pickle(self, mocked_db_connection: tuple[MagicMock, MagicMock]) -> None:
        db = make_db("test_pickle", drop_old=False)

        restored = pickle.loads(pickle.dumps(db))  # noqa: S301

        assert restored.dim == DIM
        assert restored.table_name == "test_pickle"
        assert restored.case_config.metric_type == MetricType.COSINE

    @pytest.mark.parametrize(
        ("filters", "expected_sql"),
        [
            (NonFilter(), 'ORDER BY "embedding" <=>'),
            (IntFilter(int_value=42, filter_rate=0.5), "WHERE id >= 42"),
            (LabelFilter(label_percentage=0.2), "WHERE label = 'label_20p'"),
        ],
    )
    def test_filter(
        self,
        mocked_db_connection: tuple[MagicMock, MagicMock],
        filters: Filter,
        expected_sql: str,
    ):
        db = make_db("test_filter", drop_old=False)

        db.prepare_filter(filters)

        assert expected_sql in db._search.as_string()

    @pytest.mark.parametrize("with_scalar_labels", [False, True])
    def test_insert(
        self,
        mocked_db_connection: tuple[MagicMock, MagicMock],
        with_scalar_labels: bool,
    ):
        conn, cursor = mocked_db_connection
        db = make_db("test_insert", drop_old=False, with_scalar_labels=with_scalar_labels)
        db.conn = conn
        db.cursor = cursor
        copy_writer = cursor.copy.return_value.__enter__.return_value
        embeddings = [[0.1, 0.2], [0.3, 0.4]]
        metadata = [7, 8]
        labels = ["label-a", "label-b"] if with_scalar_labels else None

        count, error = db.insert_embeddings(embeddings, metadata, labels)

        assert error is None
        assert count == 2
        expected_types = ["bigint", "vector", "varchar"] if with_scalar_labels else ["bigint", "vector"]
        assert copy_writer.set_types.call_count == 1
        copy_writer.set_types.assert_called_with(expected_types)
        rows = [call.args[0] for call in copy_writer.write_row.call_args_list]
        assert [int(row[0]) for row in rows] == metadata
        np.testing.assert_allclose([row[1] for row in rows], embeddings)
        if with_scalar_labels:
            assert [row[2] for row in rows] == labels
        conn.commit.assert_called()

    def test_search(
        self,
        mocked_db_connection: tuple[MagicMock, MagicMock],
    ):
        conn, cursor = mocked_db_connection
        db = make_db("test_search", drop_old=False)
        db.conn = conn
        db.cursor = cursor
        db.prepare_filter(NonFilter())
        cursor.execute.return_value.fetchall.return_value = [(7,), (3,)]

        result = db.search_embedding([0.1, 0.2], k=2)

        assert result == [7, 3]
        query_args = cursor.execute.call_args.args[1]
        np.testing.assert_allclose(query_args[0], [0.1, 0.2])
        assert query_args[1] == 2
        assert cursor.execute.call_args.kwargs == {"prepare": True, "binary": True}
