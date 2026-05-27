import datetime
import logging
import math
import threading
from collections.abc import Generator
from contextlib import contextmanager

from vespa import application

from ..api import VectorDB
from . import util
from .config import VespaFtsConfig, VespaHNSWConfig

log = logging.getLogger(__name__)


def _is_successful_response(response) -> bool:
    if hasattr(response, "is_successful"):
        return response.is_successful()
    if hasattr(response, "get_status_code"):
        return response.get_status_code() == 200
    return getattr(response, "status_code", None) == 200


def _response_json(response) -> dict:
    if hasattr(response, "get_json"):
        return response.get_json()
    return getattr(response, "json", {})


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

        client = self.deploy_http()
        client.wait_for_application_up()

        if drop_old:
            try:
                client.delete_all_docs("vectordbbench_content", self.schema_name)
            except Exception:
                drop_old = False
                log.exception(f"Vespa client drop_old schema: {self.schema_name}")

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
        yield
        self.client = None

    @classmethod
    def supports_full_text_search(cls) -> bool:
        return True

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

        data = (
            {"id": str(doc_id), "fields": {"id": str(doc_id), "text": text}}
            for doc_id, text in zip(doc_ids, texts, strict=True)
        )

        successful_count = 0
        failures: list[str] = []
        lock = threading.Lock()

        def callback(response, doc_id):
            nonlocal successful_count
            with lock:
                if _is_successful_response(response):
                    successful_count += 1
                    return
                failures.append(f"{doc_id}: {_response_json(response)}")

        try:
            self.client.feed_iterable(data, self.schema_name, callback=callback)
        except Exception as exc:
            log.warning("Vespa feed failed for schema %s", self.schema_name, exc_info=True)
            return successful_count, exc

        if failures:
            msg = f"Vespa feed failed for {len(failures)} documents in schema {self.schema_name}: {failures[:3]}"
            err = RuntimeError(msg)
            log.warning(msg)
            return successful_count, err

        return len(texts), None

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
        **kwargs,
    ) -> list[str]:
        if not self._is_fts:
            msg = "Vespa full-text search requires VespaFtsConfig"
            raise RuntimeError(msg)
        assert self.client is not None

        yql = f"select id from {self.schema_name} where userQuery()"  # noqa: S608
        result = self.client.query(
            {
                "yql": yql,
                "query": query,
                "type": "any",
                "ranking": "bm25",
                "hits": k,
                "default-index": "text",
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
                Field("text", "string", indexing=["index", "summary"], index="enable-bm25"),
            ]
            return ApplicationPackage(
                "vectordbbench",
                [
                    Schema(
                        self.schema_name,
                        Document(fields),
                        rank_profiles=[
                            RankProfile(name="bm25", first_phase="bm25(text)", inherits="default"),
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
