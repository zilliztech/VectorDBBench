import datetime
import json
import logging
import math
import shutil
import subprocess
import tempfile
import threading
import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from vespa import application

from vectordb_bench.backend.payload import PayloadProfile

from ..api import VectorDB
from . import util
from .config import VespaFtsConfig, VespaHNSWConfig

log = logging.getLogger(__name__)

VESPA_DOC_COUNT_POLL_INTERVAL_SEC = 2
VESPA_DOC_COUNT_TIMEOUT_SEC = 1800
VESPA_FEED_OUTPUT_TAIL_CHARS = 4000


def _document_id_from_hit(hit: dict) -> str | None:
    fields = hit.get("fields")
    if isinstance(fields, dict) and "id" in fields:
        return str(fields["id"])

    document_id = hit.get("id")
    if document_id is None:
        return None

    document_id = str(document_id)
    if document_id.startswith("id:") and "::" in document_id:
        return document_id.rsplit("::", 1)[-1]
    return document_id


def _tail_text(value: str, limit: int = VESPA_FEED_OUTPUT_TAIL_CHARS) -> str:
    return value[-limit:] if len(value) > limit else value


class Vespa(VectorDB):
    def __init__(
        self,
        dim: int,
        db_config: dict[str, str],
        db_case_config: VespaHNSWConfig | VespaFtsConfig | None = None,
        collection_name: str = "VectorDBBenchCollection",
        drop_old: bool = False,
        **kwargs,
    ) -> None:
        self.dim = dim
        self.db_config = db_config
        self.case_config = db_case_config or VespaHNSWConfig()
        self._is_fts = isinstance(self.case_config, VespaFtsConfig)
        self.schema_name = collection_name
        self._text_field = "text"

        client = self.deploy_http()
        client.wait_for_application_up()

        if drop_old:
            try:
                client.delete_all_docs("vectordbbench_content", self.schema_name)
            except Exception:
                if self._is_fts:
                    raise
                drop_old = False
                log.exception(f"Vespa client drop_old schema: {self.schema_name}")
            else:
                if self._is_fts:
                    self._wait_for_document_count(client, 0, "drop_old")

    @contextmanager
    def init(self) -> Generator[None, None, None]:
        """create and destory connections to database.
        Why contextmanager:

            In multiprocessing search tasks, vectordbbench might init
            totally hundreds of thousands of connections with DB server.

            Too many connections may drain local FDs or server connection resources.
            If the DB client doesn't have `close()` method, just set the object to None.

        Examples:
            >>> with self.init():
            >>>     self.insert_embeddings()
        """
        self.client = application.Vespa(self.db_config["url"], port=self.db_config["port"])
        self._reset_fts_feed_client()
        try:
            yield
            self._finish_fts_feed_client()
        finally:
            self._cleanup_fts_feed_client()
            self.client = None

    @classmethod
    def supports_full_text_search(cls) -> bool:
        return True

    def has_text_field(self) -> bool:
        return bool(getattr(self, "_is_fts", False) and getattr(self, "_text_field", None))

    def need_normalize_cosine(self) -> bool:
        """Wheather this database need to normalize dataset to support COSINE"""
        return False

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        **kwargs,
    ) -> tuple[int, Exception | None]:
        """Insert the embeddings to the vector database. The default number of embeddings for
        each insert_embeddings is 5000.

        Args:
            embeddings(list[list[float]]): list of embedding to add to the vector database.
            metadatas(list[int]): metadata associated with the embeddings, for filtering.
            **kwargs(Any): vector database specific parameters.

        Returns:
            int: inserted data count
        """
        assert self.client is not None

        data = ({"id": str(i), "fields": {"id": i, "embedding": e}} for i, e in zip(metadata, embeddings, strict=True))
        self.client.feed_iterable(data, self.schema_name)
        return len(embeddings), None

    def insert_documents(
        self,
        texts: list[str],
        doc_ids: list[str],
        **kwargs,
    ) -> tuple[int, Exception | None]:
        if not self._is_fts:
            msg = "Vespa full-text insert requires VespaFtsConfig"
            raise RuntimeError(msg)
        assert self.client is not None

        if len(texts) != len(doc_ids):
            msg = f"Mismatch between texts ({len(texts)}) and doc_ids ({len(doc_ids)}) lengths"
            raise ValueError(msg)

        try:
            self._write_fts_feed_batch(texts, doc_ids)
        except Exception as exc:
            log.warning("Vespa feed failed for schema %s", self.schema_name, exc_info=True)
            return 0, exc

        return len(texts), None

    def _reset_fts_feed_client(self) -> None:
        self._feed_proc = None
        self._feed_stdout_file = None
        self._feed_stderr_file = None
        self._feed_written_count = 0
        self._feed_lock = threading.Lock()

    def _ensure_fts_feed_client(self) -> subprocess.Popen:
        if self._feed_proc is not None:
            return self._feed_proc
        command = self.case_config.feed_client_command
        if shutil.which(command) is None:
            msg = (
                f"Vespa feed client command {command!r} was not found. "
                "Install the Vespa CLI or set VespaFtsConfig.feed_client_command."
            )
            raise RuntimeError(msg)

        connections = self.case_config.feed_client_connections or 8
        cmd = [
            command,
            "feed",
            "-",
            "--target",
            self._feed_target(),
            "--connections",
            str(connections),
            "--inflight",
            "0",
            "--progress",
            "0",
        ]
        log.info("Start Vespa feed client: %s", " ".join(cmd))

        self._feed_stdout_file = tempfile.TemporaryFile(mode="w+", encoding="utf-8")
        self._feed_stderr_file = tempfile.TemporaryFile(mode="w+", encoding="utf-8")
        self._feed_proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=self._feed_stdout_file,
            stderr=self._feed_stderr_file,
            text=True,
        )
        return self._feed_proc

    def _write_fts_feed_batch(self, texts: list[str], doc_ids: list[str]) -> None:
        lines = []
        for doc_id, text in zip(doc_ids, texts, strict=True):
            operation = {
                "put": self._vespa_document_id(str(doc_id)),
                "fields": {"id": str(doc_id), "text": text},
            }
            lines.append(json.dumps(operation, ensure_ascii=False, separators=(",", ":")))

        with self._feed_lock:
            proc = self._ensure_fts_feed_client()
            returncode = proc.poll()
            if returncode is not None:
                msg = f"Vespa feed client exited before all documents were written: returncode={returncode}"
                raise RuntimeError(msg)
            assert proc.stdin is not None
            proc.stdin.write("\n".join(lines))
            proc.stdin.write("\n")
            proc.stdin.flush()
            self._feed_written_count += len(lines)

    def _finish_fts_feed_client(self) -> None:
        if self._feed_proc is None:
            return
        with self._feed_lock:
            proc = self._feed_proc
            if proc.stdin is not None and not proc.stdin.closed:
                proc.stdin.close()
            returncode = proc.wait()
            assert self._feed_stdout_file is not None
            assert self._feed_stderr_file is not None
            self._feed_stdout_file.seek(0)
            self._feed_stderr_file.seek(0)
            stdout = self._feed_stdout_file.read()
            stderr = self._feed_stderr_file.read()

        count = self._feed_written_count
        metrics = self._parse_feed_metrics(stdout)
        ok_count = int(metrics.get("feeder.ok.count", count)) if metrics else count
        error_count = int(metrics.get("feeder.error.count", 0)) if metrics else 0
        response_error_count = int(metrics.get("http.response.error.count", 0)) if metrics else 0

        if returncode != 0 or error_count or response_error_count or ok_count != count:
            msg = (
                "Vespa feed client failed "
                f"returncode={returncode}, written={count}, ok={ok_count}, "
                f"feed_errors={error_count}, response_errors={response_error_count}, "
                f"stdout_tail={_tail_text(stdout)!r}, stderr_tail={_tail_text(stderr)!r}"
            )
            raise RuntimeError(msg)

        log.info("Vespa feed client inserted %d docs; metrics=%s", ok_count, metrics)
        self._feed_proc = None

    def _cleanup_fts_feed_client(self) -> None:
        proc = getattr(self, "_feed_proc", None)
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait()
        for file_attr in ("_feed_stdout_file", "_feed_stderr_file"):
            file_obj = getattr(self, file_attr, None)
            if file_obj is not None:
                file_obj.close()
                setattr(self, file_attr, None)
        self._feed_proc = None

    def _vespa_document_id(self, doc_id: str) -> str:
        return f"id:{self.schema_name}:{self.schema_name}::{doc_id}"

    def _feed_target(self) -> str:
        target = str(self.db_config["url"]).rstrip("/")
        port = self.db_config.get("port")
        if port is not None and f":{port}" not in target.rsplit("/", 1)[-1]:
            target = f"{target}:{port}"
        return target

    def _parse_feed_metrics(self, output: str) -> dict[str, Any]:
        text = output.strip()
        if not text:
            return {}
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.rfind("\n{")
            if start >= 0:
                try:
                    return json.loads(text[start + 1 :])
                except json.JSONDecodeError:
                    pass
        log.warning("Failed to parse Vespa feed client metrics: %s", _tail_text(output))
        return {}

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
    ) -> list[int]:
        """Get k most similar embeddings to query vector.

        Args:
            query(list[float]): query embedding to look up documents similar to.
            k(int): Number of most similar embeddings to return. Defaults to 100.
            filters(dict, optional): filtering expression to filter the data while searching.

        Returns:
            list[int]: list of k most similar embeddings IDs to the query embedding.
        """
        assert self.client is not None

        ef = self.case_config.ef
        extra_ef = max(0, ef - k)
        embedding_field = "embedding" if self.case_config.quantization_type == "none" else "embedding_binary"

        yql = (
            f"select id from {self.schema_name} where "
            f"{{targetHits: {k}, hnsw.exploreAdditionalHits: {extra_ef}}}"
            f"nearestNeighbor({embedding_field}, query_embedding)"
        )

        if filters:
            id_filter = filters.get("id")
            yql += f" and id >= {id_filter}"

        query_embedding = query if self.case_config.quantization_type == "none" else util.binarize_tensor(query)

        ranking = self.case_config.quantization_type

        result = self.client.query({"yql": yql, "input.query(query_embedding)": query_embedding, "ranking": ranking})
        return [child["fields"]["id"] for child in result.get_json()["root"]["children"]]

    def search_documents(
        self,
        query: str,
        k: int = 100,
        payload_profile: PayloadProfile = PayloadProfile.IDS_ONLY,
        **kwargs,
    ) -> list[str]:
        if not self._is_fts:
            msg = "Vespa full-text search requires VespaFtsConfig"
            raise RuntimeError(msg)
        if not self.supports_document_payload_profile(payload_profile):
            msg = f"Vespa does not support document payload_profile={payload_profile.value}"
            raise NotImplementedError(msg)
        assert self.client is not None

        selected_fields = "id"
        if payload_profile == PayloadProfile.TEXT:
            selected_fields = f"id, {self._text_field}"
        yql = f"select {selected_fields} from {self.schema_name} where userQuery()"
        result = self.client.query(
            {
                "yql": yql,
                "query": query,
                "type": "any",
                "ranking": "bm25",
                "hits": k,
                "default-index": self._text_field,
            }
        )
        children = result.get_json().get("root", {}).get("children", []) or []
        ids = []
        for child in children:
            if not isinstance(child, dict):
                continue
            doc_id = _document_id_from_hit(child)
            if doc_id is not None:
                ids.append(doc_id)
        return ids

    def optimize(self, data_size: int | None = None):
        """optimize will be called between insertion and search in performance cases.

        Should be blocked until the vectorDB is ready to be tested on
        heavy performance cases.

        Time(insert the dataset) + Time(optimize) will be recorded as "load_duration" metric
        Optimize's execution time is limited, the limited time is based on cases.
        """
        if self._is_fts and data_size is not None:
            assert self.client is not None
            self._wait_for_document_count(self.client, data_size, "post_insert")

    def _document_count(self, client: application.Vespa) -> int:
        result = client.query({"yql": f"select id from {self.schema_name} where true", "hits": 0})
        root = result.get_json().get("root", {})
        fields = root.get("fields", {})
        if "totalCount" not in fields:
            msg = f"Vespa count query did not return totalCount for schema {self.schema_name}: {root}"
            raise RuntimeError(msg)
        return int(fields["totalCount"])

    def _wait_for_document_count(self, client: application.Vespa, expected_count: int, stage: str) -> None:
        deadline = time.monotonic() + VESPA_DOC_COUNT_TIMEOUT_SEC
        last_count = None
        while True:
            last_count = self._document_count(client)
            if last_count == expected_count:
                log.info("Vespa %s document count reached %d", stage, expected_count)
                return
            if time.monotonic() >= deadline:
                msg = (
                    f"Timed out waiting for Vespa {stage} document count to reach "
                    f"{expected_count}; last_count={last_count}"
                )
                raise TimeoutError(msg)
            log.info("Waiting for Vespa %s document count: current=%d expected=%d", stage, last_count, expected_count)
            time.sleep(VESPA_DOC_COUNT_POLL_INTERVAL_SEC)

    @property
    def application_package(self):
        if getattr(self, "_application_package", None) is None:
            self._application_package = self._create_application_package()
        return self._application_package

    def _create_application_package(self):
        from vespa.package import (
            HNSW,
            ApplicationPackage,
            Document,
            Field,
            RankProfile,
            Schema,
            Validation,
            ValidationID,
        )

        tomorrow = datetime.date.today() + datetime.timedelta(days=1)

        if self._is_fts:
            fields = [
                Field("id", "string", indexing=["summary", "attribute"]),
                Field(
                    "text",
                    "string",
                    indexing=["index", "summary"],
                    index="enable-bm25",
                    stemming="none",
                ),
            ]
            return ApplicationPackage(
                "vectordbbench",
                [
                    Schema(
                        self.schema_name,
                        Document(fields),
                        rank_profiles=[
                            RankProfile(
                                name="bm25",
                                first_phase="bm25(text)",
                                inherits="default",
                                rank_properties=self.case_config.rank_properties(),
                            ),
                        ],
                    )
                ],
                validations=[Validation(ValidationID.fieldTypeChange, until=tomorrow)],
            )

        fields = [
            Field(
                "id",
                "int",
                indexing=["summary", "attribute"],
            ),
            Field(
                "embedding",
                f"tensor<float>(x[{self.dim}])",
                indexing=["summary", "attribute", "index"],
                ann=HNSW(**self.case_config.index_param()),
            ),
        ]

        if self.case_config.quantization_type == "binary":
            fields.append(
                Field(
                    "embedding_binary",
                    f"tensor<int8>(x[{math.ceil(self.dim / 8)}])",
                    indexing=[
                        "input embedding",
                        # convert 32 bit float to 1 bit
                        "binarize",
                        # pack 8 bits into one int8
                        "pack_bits",
                        "summary",
                        "attribute",
                        "index",
                    ],
                    ann=HNSW(**{**self.case_config.index_param(), "distance_metric": "hamming"}),
                    is_document_field=False,
                )
            )

        return ApplicationPackage(
            "vectordbbench",
            [
                Schema(
                    self.schema_name,
                    Document(
                        fields,
                    ),
                    rank_profiles=[
                        RankProfile(
                            name="none",
                            first_phase="",
                            inherits="default",
                            inputs=[("query(query_embedding)", f"tensor<float>(x[{self.dim}])")],
                        ),
                        RankProfile(
                            name="binary",
                            first_phase="",
                            inherits="default",
                            inputs=[("query(query_embedding)", f"tensor<int8>(x[{math.ceil(self.dim / 8)}])")],
                        ),
                    ],
                )
            ],
            validations=[
                Validation(ValidationID.tensorTypeChange, until=tomorrow),
                Validation(ValidationID.fieldTypeChange, until=tomorrow),
            ],
        )

    def deploy_http(self) -> application.Vespa:
        """
        Deploy a Vespa application package via HTTP REST API.

        Returns:
            application.Vespa: The deployed Vespa application instance
        """
        import requests

        url = self.db_config["url"] + ":19071/application/v2/tenant/default/prepareandactivate"
        package_data = self.application_package.to_zip()
        headers = {"Content-Type": "application/zip"}

        try:
            response = requests.post(url=url, data=package_data, headers=headers, timeout=10)

            response.raise_for_status()
            result = response.json()
            return application.Vespa(
                url=self.db_config["url"],
                port=self.db_config["port"],
                deployment_message=result.get("message"),
                application_package=self.application_package,
            )

        except requests.exceptions.RequestException as e:
            error_msg = f"Failed to deploy Vespa application: {e!s}"
            if hasattr(e, "response") and e.response is not None:
                error_msg += f" - Response: {e.response.text}"
            raise RuntimeError(error_msg) from e
