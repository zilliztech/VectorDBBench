from contextlib import nullcontext
from unittest.mock import Mock

import pytest

pytest.importorskip("qdrant_client")

from vectordb_bench.backend.clients.qdrant_cloud.config import QdrantIndexConfig
from vectordb_bench.backend.clients.qdrant_cloud.qdrant_cloud import QdrantCloud


@pytest.mark.parametrize("raises", [False, True])
def test_qdrant_client_is_closed_after_context(raises: bool) -> None:
    database = QdrantCloud(2, {"location": ":memory:"}, QdrantIndexConfig())
    expected = pytest.raises(ValueError, match="benchmark failed") if raises else nullcontext()
    client = None
    try:
        with expected, database.init():
            client = database.qdrant_client
            assert client.get_collections().collections == []
            if raises:
                raise ValueError("benchmark failed")

        with pytest.raises(RuntimeError, match="closed"):
            client.get_collections()
        assert not hasattr(database, "qdrant_client")
    finally:
        if client is not None:
            client.close()


@pytest.mark.parametrize("body_error", [None, ValueError("benchmark failed"), KeyboardInterrupt()])
def test_qdrant_client_cleanup_failure(
    body_error: BaseException | None, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    database = QdrantCloud(2, {"location": ":memory:"}, QdrantIndexConfig())
    close_error = RuntimeError("close failed")
    close = Mock(side_effect=close_error)
    expected_error = close_error if body_error is None else body_error
    original_close = None
    try:
        with pytest.raises(type(expected_error)) as caught:
            with database.init():
                original_close = database.qdrant_client.close
                monkeypatch.setattr(database.qdrant_client, "close", close)
                if body_error is not None:
                    raise body_error

        assert caught.value is expected_error
        close.assert_called_once_with()
        assert not hasattr(database, "qdrant_client")
        if body_error is not None:
            assert "close failed" in caplog.text
    finally:
        if original_close is not None:
            original_close()
