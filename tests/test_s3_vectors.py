"""Mock-based unit tests for the AWS S3 Vectors backend.

These tests do not connect to AWS — boto3.client is patched with MagicMock.
Project precedent: this is the first mock-boto3 test file in tests/. Other AWS
backend tests (e.g. test_aws_opensearch_cli.py) only test helper functions.
The mock approach lets us verify chunking, retry config, and filter logic
without an AWS account or moto dependency.
"""

from unittest.mock import MagicMock, patch

from pydantic import SecretStr


def _build_config(**overrides):
    from vectordb_bench.backend.clients.s3_vectors.config import S3VectorsConfig

    defaults = {
        "access_key_id": SecretStr("AKIA_FAKE"),
        "secret_access_key": SecretStr("secret_fake"),
        "bucket_name": "fake-bucket",
    }
    defaults.update(overrides)
    return S3VectorsConfig(**defaults)


def test_config_to_dict_exposes_all_tuning_fields():
    """to_dict() is a hand-maintained whitelist; this guards against forgetting
    to expose newly added fields to the runtime."""
    d = _build_config().to_dict()
    assert d["insert_batch_size"] == 100
    assert d["max_pool_connections"] == 50
    assert d["retry_mode"] == "adaptive"
    assert d["retry_max_attempts"] == 10
    assert d["endpoint_url"] is None


def _default_db_and_case():
    """Build a default (db_config dict, S3VectorsIndexConfig) pair for tests
    that need to construct an S3Vectors instance. Tests still must patch
    boto3 themselves — this helper only assembles the config inputs."""
    from vectordb_bench.backend.clients.api import MetricType
    from vectordb_bench.backend.clients.s3_vectors.config import S3VectorsIndexConfig

    db_config = _build_config().to_dict()
    case_config = S3VectorsIndexConfig(metric_type=MetricType.COSINE)
    return db_config, case_config


def test_client_built_with_botocore_config():
    """boto3.client must receive a botocore Config with our tuning values so
    urllib3 pool size and adaptive retry actually take effect."""
    from vectordb_bench.backend.clients.s3_vectors import s3_vectors as mod

    db_config, case_config = _default_db_and_case()

    with patch.object(mod, "boto3") as mock_boto3:
        fake_client = MagicMock()
        mock_boto3.client.return_value = fake_client

        mod.S3Vectors(
            dim=4,
            db_config=db_config,
            db_case_config=case_config,
            drop_old=False,
        )

        assert mock_boto3.client.called
        call_kwargs = mock_boto3.client.call_args.kwargs
        assert "config" in call_kwargs, "boto3.client must be called with config="
        cfg = call_kwargs["config"]
        assert cfg.max_pool_connections == 50
        assert cfg.retries == {"mode": "adaptive", "max_attempts": 10}


def test_endpoint_url_passed_to_boto3_client():
    """When endpoint_url is set, both __init__ and init() boto3.client calls
    must receive it. Guards against regression if a refactor drops the kwarg."""
    from vectordb_bench.backend.clients.s3_vectors import s3_vectors as mod

    db_config, case_config = _default_db_and_case()
    db_config["endpoint_url"] = "http://192.168.1.100:8080"

    with patch.object(mod, "boto3") as mock_boto3:
        fake_client = MagicMock()
        mock_boto3.client.return_value = fake_client

        db = mod.S3Vectors(
            dim=4,
            db_config=db_config,
            db_case_config=case_config,
            drop_old=False,
        )
        # Verify __init__ client got endpoint_url
        init_call = mock_boto3.client.call_args
        assert init_call.kwargs.get("endpoint_url") == "http://192.168.1.100:8080"

        with db.init():
            # Verify init() client also got endpoint_url
            # init() creates a second client, so total call_count is 2
            assert mock_boto3.client.call_count == 2
            calls = mock_boto3.client.call_args_list
            assert calls[1].kwargs.get("endpoint_url") == "http://192.168.1.100:8080"


def test_insert_chunks_to_batch_size():
    """250 records + insert_batch_size=100 → exactly 3 put_vectors calls
    with sizes 100, 100, 50."""
    from vectordb_bench.backend.clients.s3_vectors import s3_vectors as mod

    db_config, case_config = _default_db_and_case()

    with patch.object(mod, "boto3") as mock_boto3:
        fake_client = MagicMock()
        mock_boto3.client.return_value = fake_client

        db = mod.S3Vectors(
            dim=4,
            db_config=db_config,
            db_case_config=case_config,
            drop_old=False,
        )
        with db.init():
            count, err = db.insert_embeddings(
                embeddings=[[0.1, 0.2, 0.3, 0.4]] * 250,
                metadata=list(range(250)),
            )

        assert count == 250
        assert err is None
        assert fake_client.put_vectors.call_count == 3
        sizes = [len(call.kwargs["vectors"]) for call in fake_client.put_vectors.call_args_list]
        assert sizes == [100, 100, 50]


def test_insert_returns_partial_count_on_error():
    """Second put_vectors raises ClientError → insert_embeddings returns
    (count_from_first_batch, exception). Validates the contract that
    ConcurrentInsertRunner relies on for partial-success accounting."""
    from botocore.exceptions import ClientError

    from vectordb_bench.backend.clients.s3_vectors import s3_vectors as mod

    db_config, case_config = _default_db_and_case()

    err = ClientError(
        error_response={"Error": {"Code": "ThrottlingException", "Message": "throttled"}},
        operation_name="PutVectors",
    )

    with patch.object(mod, "boto3") as mock_boto3:
        fake_client = MagicMock()
        # First call OK (returns mock); second call raises.
        fake_client.put_vectors.side_effect = [MagicMock(), err]
        mock_boto3.client.return_value = fake_client

        db = mod.S3Vectors(
            dim=4,
            db_config=db_config,
            db_case_config=case_config,
            drop_old=False,
        )
        with db.init():
            count, returned_err = db.insert_embeddings(
                embeddings=[[0.1, 0.2, 0.3, 0.4]] * 250,
                metadata=list(range(250)),
            )

        assert count == 100, "First batch (100 records) should have committed"
        assert returned_err is err, "The exact ClientError instance should be returned"


def test_drop_old_skips_when_index_absent():
    """list_indexes returns empty → __init__ must not call delete_index,
    and must still call create_index. Guards against accidentally adding
    an unconditional delete_index that 404s on a fresh bucket."""
    from vectordb_bench.backend.clients.s3_vectors import s3_vectors as mod

    db_config, case_config = _default_db_and_case()

    with patch.object(mod, "boto3") as mock_boto3:
        fake_client = MagicMock()
        fake_client.list_indexes.return_value = {"indexes": []}
        mock_boto3.client.return_value = fake_client

        mod.S3Vectors(
            dim=4,
            db_config=db_config,
            db_case_config=case_config,
            drop_old=True,
        )

        fake_client.delete_index.assert_not_called()
        fake_client.create_index.assert_called_once()


def test_filter_translation():
    """Verify each FilterOp branch in prepare_filter."""
    from vectordb_bench.backend.clients.s3_vectors import s3_vectors as mod
    from vectordb_bench.backend.filter import Filter, FilterOp

    db_config, case_config = _default_db_and_case()

    with patch.object(mod, "boto3") as mock_boto3:
        mock_boto3.client.return_value = MagicMock()
        db = mod.S3Vectors(
            dim=4,
            db_config=db_config,
            db_case_config=case_config,
            drop_old=False,
        )

        f_none = Filter(type=FilterOp.NonFilter)
        db.prepare_filter(f_none)
        assert db.filter is None

        from vectordb_bench.backend.filter import IntFilter, LabelFilter

        f_num = IntFilter(int_value=42)
        db.prepare_filter(f_num)
        assert db.filter == {"id": {"$gte": 42}}

        f_str = LabelFilter(label_percentage=0.05)
        db.prepare_filter(f_str)
        assert db.filter == {"label": "label_5p"}


def test_thread_safe_attribute():
    """Defense against future maintainers flipping thread_safe to False —
    the implementation deliberately shares one boto3 client across threads
    and relies on this attribute being True."""
    from vectordb_bench.backend.clients.s3_vectors.s3_vectors import S3Vectors

    assert S3Vectors.thread_safe is True
