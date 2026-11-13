"""VectorDBBench client for DXMT, over its REST API (`/api/v1`)."""

import logging
import time
from contextlib import contextmanager
from urllib.parse import quote

import httpx
import orjson

from ...filter import Filter, FilterOp
from ..api import VectorDB
from .config import DXMTIndexConfig

log = logging.getLogger(__name__)

# The server refuses a request carrying more than 10,000 records.
MAX_RECORDS_PER_REQUEST = 5_000

# Payload fields written with every record, for the filtered cases. Record ids are strings
# in DXMT, so the integer the NumGE cases compare against is stored separately.
ID_FIELD = "id_num"
LABEL_FIELD = "label"

# How long an insert keeps retrying while the server answers 429 (write backpressure).
MAX_BACKPRESSURE_WAIT_SEC = 600


class DXMTError(RuntimeError):
    pass


class DXMT(VectorDB):
    supported_filter_types: list[FilterOp] = [FilterOp.NonFilter, FilterOp.NumGE, FilterOp.StrEqual]
    name = "DXMT"

    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: DXMTIndexConfig | None,
        collection_name: str = "VectorDBBenchCollection",
        drop_old: bool = False,
        with_scalar_labels: bool = False,
        **kwargs,
    ):
        self.dim = dim
        self.db_config = db_config
        self.case_config = db_case_config
        self.table_name = collection_name
        self.with_scalar_labels = with_scalar_labels
        self.base_url = db_config.get("url", "http://localhost:9999").rstrip("/") + "/api/v1"
        self.timeout = float(db_config.get("timeout_sec") or 600)
        self.headers = {"content-type": "application/json"}
        if db_config.get("api_key"):
            self.headers["api-key"] = db_config["api_key"]

        self.index_param = db_case_config.index_param() if db_case_config else {"distance_metric": "cosine"}
        search_param = db_case_config.search_param() if db_case_config else {}
        self.search_mode = search_param.get("search_mode")
        self.query_filter = None
        self.client: httpx.Client | None = None

        with self._connect() as client:
            self.client = client
            try:
                if drop_old:
                    self._drop_table()
                if not self._table_exists():
                    self._create_table()
            finally:
                self.client = None

    def _connect(self) -> httpx.Client:
        return httpx.Client(base_url=self.base_url, headers=self.headers, timeout=self.timeout)

    @property
    def _table_path(self) -> str:
        return f"/tables/{quote(self.table_name, safe='')}"

    @staticmethod
    def _data(resp: httpx.Response) -> dict:
        try:
            body = resp.json()
        except ValueError:
            body = None
        if resp.is_success and isinstance(body, dict) and body.get("success", True):
            return body.get("data") or {}
        error = body.get("error") if isinstance(body, dict) else None
        msg = f"DXMT {resp.request.method} {resp.request.url.path} -> {resp.status_code}: {error or resp.text}"
        raise DXMTError(msg)

    def _table_exists(self) -> bool:
        resp = self.client.get(self._table_path)
        if resp.status_code == httpx.codes.NOT_FOUND:
            return False
        self._data(resp)
        return True

    def _drop_table(self):
        resp = self.client.delete(self._table_path)
        if resp.status_code == httpx.codes.NOT_FOUND:
            return
        self._data(resp)
        log.info("dropped table %s", self.table_name)

    def _create_table(self):
        body = {"name": self.table_name, "dimension": self.dim, **self.index_param}
        table = self._data(self.client.post("/tables", content=orjson.dumps(body)))
        # A server that ignored the requested graph parameters would build a different index
        # than the one this case is labelled with.
        wanted = self.index_param.get("hnsw")
        if wanted and table.get("hnsw") != wanted:
            msg = f"DXMT created {self.table_name} with hnsw={table.get('hnsw')}, requested {wanted}"
            raise DXMTError(msg)
        log.info("created table %s: %s", self.table_name, body)

    @contextmanager
    def init(self):
        self.client = self._connect()
        try:
            yield
        finally:
            self.client.close()
            self.client = None

    def need_normalize_cosine(self) -> bool:
        return False

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        labels_data: list[str] | None = None,
        **kwargs,
    ) -> tuple[int, Exception | None]:
        assert self.client is not None, "call init() first"
        inserted = 0
        try:
            for start in range(0, len(embeddings), MAX_RECORDS_PER_REQUEST):
                end = start + MAX_RECORDS_PER_REQUEST
                ids = metadata[start:end]
                vectors = embeddings[start:end]
                labels = labels_data[start:end] if self.with_scalar_labels and labels_data else None
                records = []
                for i, (pk, vec) in enumerate(zip(ids, vectors, strict=True)):
                    payload = {ID_FIELD: int(pk)}
                    if labels is not None:
                        payload[LABEL_FIELD] = labels[i]
                    records.append({"id": str(pk), "vector": vec, "payload": payload})
                inserted += self._insert_batch(records)
        except Exception as e:
            log.warning("insert into %s failed after %d rows: %s", self.table_name, inserted, e)
            return inserted, e
        return inserted, None

    def _insert_batch(self, records: list[dict]) -> int:
        # OPT_SERIALIZE_NUMPY encodes numpy rows in C; plain lists pass through unchanged.
        content = orjson.dumps({"records": records}, option=orjson.OPT_SERIALIZE_NUMPY)
        deadline = time.monotonic() + MAX_BACKPRESSURE_WAIT_SEC
        while True:
            resp = self.client.post(f"{self._table_path}/records", content=content)
            if resp.status_code != httpx.codes.TOO_MANY_REQUESTS or time.monotonic() > deadline:
                break
            time.sleep(float(resp.headers.get("retry-after") or 1))
        data = self._data(resp)
        if data.get("failed_count"):
            msg = f"DXMT rejected {data['failed_count']} of {len(records)} records: {data.get('failed')}"
            raise DXMTError(msg)
        return int(data.get("count", len(records)))

    def prepare_filter(self, filters: Filter):
        if filters.type == FilterOp.NonFilter:
            self.query_filter = None
        elif filters.type == FilterOp.NumGE:
            self.query_filter = {
                "must": [{"key": ID_FIELD, "operation": {"type": "GreaterThanOrEqual", "value": filters.int_value}}]
            }
        elif filters.type == FilterOp.StrEqual:
            self.query_filter = {
                "must": [{"key": LABEL_FIELD, "operation": {"type": "Equal", "value": filters.label_value}}]
            }
        else:
            msg = f"Not support Filter for DXMT - {filters}"
            raise ValueError(msg)

    def search_embedding(self, query: list[float], k: int = 100, **kwargs) -> list[int]:
        assert self.client is not None, "call init() first"
        body = {"vector": query, "top_k": k, "with_payload": False}
        if self.query_filter is not None:
            body["filter"] = self.query_filter
        if self.search_mode:
            body["search_mode"] = self.search_mode
        resp = self.client.post(
            f"{self._table_path}/search",
            content=orjson.dumps(body, option=orjson.OPT_SERIALIZE_NUMPY),
        )
        return [int(hit["id"]) for hit in self._data(resp).get("hits", [])]

    def optimize(self, data_size: int | None = None):
        # Ask for the index the server would choose for the table's final size. It otherwise
        # re-evaluates that choice on a timer, and the search stages would measure whichever
        # index happened to be current. A rebuild that would change nothing returns at once.
        with self._connect() as client:
            resp = client.post(f"{self._table_path}/index/rebuild", timeout=None)
            self._data(resp)
        log.info("index of %s is final", self.table_name)
