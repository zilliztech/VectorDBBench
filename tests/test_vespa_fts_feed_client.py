from __future__ import annotations

import io
import json
import sys
import types
from contextlib import contextmanager

from vectordb_bench.backend.runner.concurrent_runner import ConcurrentInsertRunner
from vectordb_bench.backend.workload import WorkloadKind


vespa_module = types.ModuleType("vespa")
vespa_module.application = types.SimpleNamespace(Vespa=object)
sys.modules.setdefault("vespa", vespa_module)

from vectordb_bench.backend.clients.vespa.config import VespaFtsConfig  # noqa: E402
from vectordb_bench.backend.clients.vespa import vespa as vespa_mod  # noqa: E402
from vectordb_bench.backend.clients.vespa.vespa import Vespa  # noqa: E402


class FakeFtsDoc:
    def __init__(self, doc_id: str, text: str):
        self.doc_id = doc_id
        self.text = text


class FakeFtsDataset:
    def __init__(self):
        self.iterated = False

    def iter_batches(self, batch_size: int):
        self.iterated = True
        assert batch_size == 2
        yield [FakeFtsDoc("doc-1", "hello world"), FakeFtsDoc("doc-2", "vespa feed")]


class NonClosingStringIO(io.StringIO):
    def close(self):
        self.flush()


class FakeFeedProc:
    def __init__(self):
        self.stdin = NonClosingStringIO()
        self.killed = False

    def poll(self):
        return None

    def wait(self):
        return 0

    def kill(self):
        self.killed = True


def test_vespa_fts_feed_batch_writes_jsonl_operations():
    vespa = object.__new__(Vespa)
    vespa.schema_name = "vdbbench"
    vespa._reset_fts_feed_client()
    vespa._feed_proc = FakeFeedProc()

    texts = ["hello world", "vespa feed"]
    doc_ids = ["doc-1", "doc-2"]
    vespa._write_fts_feed_batch(texts, doc_ids)

    operations = [json.loads(line) for line in vespa._feed_proc.stdin.getvalue().splitlines()]
    assert operations == [
        {
            "put": "id:vdbbench:vdbbench::doc-1",
            "fields": {"id": "doc-1", "text": "hello world"},
        },
        {
            "put": "id:vdbbench:vdbbench::doc-2",
            "fields": {"id": "doc-2", "text": "vespa feed"},
        },
    ]


def test_vespa_feed_target_appends_port_once():
    vespa = object.__new__(Vespa)
    vespa.db_config = {"url": "http://127.0.0.1", "port": 8080}
    assert vespa._feed_target() == "http://127.0.0.1:8080"

    vespa.db_config = {"url": "http://127.0.0.1:8080", "port": 8080}
    assert vespa._feed_target() == "http://127.0.0.1:8080"


class BatchFtsDb:
    name = "BatchFtsDb"
    thread_safe = True

    def __init__(self):
        self.calls = []

    @contextmanager
    def init(self):
        yield

    def insert_documents(self, texts, doc_ids):
        self.calls.append({"texts": texts, "doc_ids": doc_ids})
        return len(texts), None


def test_concurrent_runner_keeps_batch_fts_insert_contract():
    db = BatchFtsDb()
    dataset = FakeFtsDataset()
    runner = ConcurrentInsertRunner(
        db=db,
        dataset=dataset,
        normalize=False,
        max_workers=3,
        batch_size=2,
        duration=9,
        workload_kind=WorkloadKind.FULL_TEXT,
    )

    assert runner.run() == 2
    assert db.calls == [{"texts": ["hello world", "vespa feed"], "doc_ids": ["doc-1", "doc-2"]}]
    assert dataset.iterated is True


def test_vespa_insert_documents_streams_batches_to_one_feed_process():
    popen_calls = []
    fake_proc = FakeFeedProc()

    def fake_popen(cmd, **kwargs):
        popen_calls.append({"cmd": cmd, "kwargs": kwargs})
        return fake_proc

    old_which = vespa_mod.shutil.which
    old_popen = vespa_mod.subprocess.Popen
    vespa_mod.shutil.which = lambda command: command
    vespa_mod.subprocess.Popen = fake_popen
    try:
        vespa = object.__new__(Vespa)
        vespa.schema_name = "vdbbench"
        vespa.db_config = {"url": "http://127.0.0.1", "port": 8080}
        vespa.case_config = VespaFtsConfig()
        vespa._is_fts = True
        vespa.client = object()
        vespa._reset_fts_feed_client()

        count, err = vespa.insert_documents(["hello world"], ["doc-1"])
        assert count == 1
        assert err is None
        count, err = vespa.insert_documents(["vespa feed"], ["doc-2"])
        assert count == 1
        assert err is None

        vespa._finish_fts_feed_client()
        vespa._cleanup_fts_feed_client()
    finally:
        vespa_mod.shutil.which = old_which
        vespa_mod.subprocess.Popen = old_popen

    assert len(popen_calls) == 1
    assert popen_calls[0]["cmd"][:5] == ["vespa", "feed", "-", "--target", "http://127.0.0.1:8080"]
    operations = [json.loads(line) for line in fake_proc.stdin.getvalue().splitlines()]
    assert operations == [
        {
            "put": "id:vdbbench:vdbbench::doc-1",
            "fields": {"id": "doc-1", "text": "hello world"},
        },
        {
            "put": "id:vdbbench:vdbbench::doc-2",
            "fields": {"id": "doc-2", "text": "vespa feed"},
        },
    ]


def test_vespa_fts_config_defaults_to_cli_feed():
    config = VespaFtsConfig()
    assert config.feed_client_command == "vespa"
    assert config.feed_client_connections is None
