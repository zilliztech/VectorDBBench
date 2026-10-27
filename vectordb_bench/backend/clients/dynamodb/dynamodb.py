"""Wrapper around Amazon DynamoDB vector search over VectorDB.

Uses only the public, GA DynamoDB vector API via the standard boto3 SDK:
  - create_table(..., VectorIndexes=[...]) with a SearchSchema to create a
    table and vector index
  - batch_write_item to load vectors
  - search_vectors to run approximate nearest-neighbour queries

Credentials come from the standard boto3 credential chain unless explicit keys
are given.

SearchSchema (see the DynamoDB vector search docs). The vector attribute is
named by VectorAttribute, not listed in SearchSchema; the schema holds only:
  - HASH:          an optional vector index partition key. When present it
                   scopes each SearchVectors call to vectors that share its
                   value, which is how search throughput scales over a large
                   index. Its value is REQUIRED in SearchConditionExpression
                   on every search. Enabled here via num_partitions > 1.
  - INLINE_FILTER: an optional non-vector attribute stored next to the vector
                   so equality filters are applied during the search. Used for
                   the StrEqual (label) benchmark filter. Inline filters and
                   the HASH key support ONLY the equality operator (=).
"""

import logging
import random
import time
from collections.abc import Iterable
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from vectordb_bench.backend.filter import Filter, FilterOp

from ..api import VectorDB
from .config import DynamoDBIndexConfig

if TYPE_CHECKING:
    from botocore.client import BaseClient

log = logging.getLogger(__name__)

WRITE_BATCH_MAX_SIZE = 25  # DynamoDB BatchWriteItem hard limit
UNPROCESSED_MAX_RETRIES = 10  # extra re-drives for throttled UnprocessedItems
UNPROCESSED_BASE_DELAY = 0.05  # seconds; exponential-backoff base
UNPROCESSED_MAX_DELAY = 20.0  # seconds; backoff ceiling
TOP_K_MAX = 100  # SearchVectors TopK valid range is 1..100

_ID_FIELD = "id"
_LABEL_FIELD = "label"
_VECTOR_FIELD = "vector"
_PK_FIELD = "pk"
_PARTITION_FIELD = "part"  # vector index partition key (SearchSchema HASH)


class DynamoDB(VectorDB):
    supported_filter_types: list[FilterOp] = [
        FilterOp.NonFilter,
        FilterOp.StrEqual,
    ]

    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: DynamoDBIndexConfig,
        drop_old: bool = False,
        with_scalar_labels: bool = False,
        **kwargs,
    ):
        self.dim = dim
        self.db_config = db_config
        self.case_config = db_case_config
        self.with_scalar_labels = with_scalar_labels

        self.table_name = db_config.get("table_name")
        self.index_name = db_config.get("index_name")
        self.num_partitions = max(1, int(db_case_config.num_partitions))
        self.use_partition_key = db_case_config.use_partition_key()

        # Prepared search condition, set by prepare_filter().
        self._condition_expr: str | None = None
        self._expr_values: dict[str, Any] = {}

        client = self._new_client()
        try:
            if drop_old:
                self._drop_table(client)
            self._create_table(client, dim)
        finally:
            client.close()

    # ------------------------------------------------------------------ #
    # Connection
    # ------------------------------------------------------------------ #
    def _new_client(self) -> "BaseClient":
        cfg = self.db_config
        kwargs: dict[str, Any] = {
            "service_name": "dynamodb",
            "region_name": cfg.get("region_name"),
            "config": Config(
                retries={"max_attempts": 8, "mode": "standard"},
                max_pool_connections=100,
            ),
        }
        if cfg.get("access_key_id") and cfg.get("secret_access_key"):
            kwargs["aws_access_key_id"] = cfg["access_key_id"]
            kwargs["aws_secret_access_key"] = cfg["secret_access_key"]
            if cfg.get("session_token"):
                kwargs["aws_session_token"] = cfg["session_token"]
        return boto3.client(**kwargs)

    # ------------------------------------------------------------------ #
    # Table / index lifecycle
    # ------------------------------------------------------------------ #
    def _drop_table(self, client: "BaseClient") -> None:
        try:
            log.info(f"DynamoDB dropping old table: {self.table_name}")
            client.delete_table(TableName=self.table_name)
            client.get_waiter("table_not_exists").wait(TableName=self.table_name)
            log.info(f"DynamoDB dropped table: {self.table_name}")
        except ClientError as error:
            if error.response["Error"]["Code"] == "ResourceNotFoundException":
                log.info(f"DynamoDB table does not exist, nothing to drop: {self.table_name}")
            else:
                raise

    def _search_schema_elements(self) -> list[dict[str, str]]:
        """SearchSchema holds only the optional partition key (HASH) and inline
        filters. The vector attribute itself is named by VectorAttribute, not
        listed here; there is no VECTOR element type in the API."""
        elements: list[dict[str, str]] = []
        if self.use_partition_key:
            # At most one HASH (vector index partition key) is allowed.
            elements.append({"AttributeName": _PARTITION_FIELD, "SearchSchemaElementType": "HASH"})
        if self.with_scalar_labels:
            elements.append({"AttributeName": _LABEL_FIELD, "SearchSchemaElementType": "INLINE_FILTER"})
        return elements

    def _create_table(self, client: "BaseClient", dim: int) -> None:
        vector_index: dict[str, Any] = {
            "IndexName": self.index_name,
            "VectorAttribute": {"AttributeName": _VECTOR_FIELD},
            "Dimensions": dim,
            "DistanceFunction": self.case_config.parse_metric(),
            "Projection": {"ProjectionType": "ALL"},
        }
        search_schema = self._search_schema_elements()
        if search_schema:
            vector_index["SearchSchema"] = search_schema

        attribute_definitions = [{"AttributeName": _PK_FIELD, "AttributeType": "N"}]
        if self.use_partition_key:
            # A SearchSchema HASH attribute must also be declared in the table's
            # AttributeDefinitions (same rule as secondary-index key attributes).
            attribute_definitions.append({"AttributeName": _PARTITION_FIELD, "AttributeType": "N"})

        try:
            log.info(
                f"DynamoDB creating table: {self.table_name} "
                f"(dim={dim}, partitions={self.num_partitions})"
            )
            client.create_table(
                TableName=self.table_name,
                KeySchema=[{"AttributeName": _PK_FIELD, "KeyType": "HASH"}],
                AttributeDefinitions=attribute_definitions,
                BillingMode="PAY_PER_REQUEST",  # vector indexes require on-demand
                VectorIndexes=[vector_index],
            )
            client.get_waiter("table_exists").wait(TableName=self.table_name)
            log.info(f"DynamoDB table active: {self.table_name}")
        except ClientError as error:
            if error.response["Error"]["Code"] == "ResourceInUseException":
                log.info(f"DynamoDB table already exists: {self.table_name}")
                client.get_waiter("table_exists").wait(TableName=self.table_name)
            else:
                raise

    @contextmanager
    def init(self):
        """Create and destroy the boto3 client for a worker process.

        Examples:
            >>> with self.init():
            >>>     self.insert_embeddings()
            >>>     self.search_embedding()
        """
        self.client = self._new_client()
        yield
        self.client.close()

    def optimize(self, **kwargs):
        """No-op: DynamoDB builds the vector index server-side."""
        return

    def need_normalize_cosine(self) -> bool:
        return False

    # ------------------------------------------------------------------ #
    # Load
    # ------------------------------------------------------------------ #
    def insert_embeddings(
        self,
        embeddings: Iterable[list[float]],
        metadata: list[int],
        labels_data: list[str] | None = None,
        **kwargs,
    ) -> tuple[int, Exception | None]:
        assert self.client is not None
        embeddings = list(embeddings)
        assert len(embeddings) == len(metadata)

        insert_count = 0
        try:
            for start in range(0, len(embeddings), WRITE_BATCH_MAX_SIZE):
                end = min(start + WRITE_BATCH_MAX_SIZE, len(embeddings))
                request_items = {
                    self.table_name: [
                        {"PutRequest": {"Item": self._build_item(metadata[i], embeddings[i], labels_data, i)}}
                        for i in range(start, end)
                    ]
                }
                self._write_batch_with_retry(request_items)
                insert_count += end - start
        except Exception as e:
            log.warning(f"DynamoDB failed to insert data: {e}")
            return insert_count, e
        return insert_count, None

    def _build_item(
        self,
        row_id: int,
        embedding: list[float],
        labels_data: list[str] | None,
        i: int,
    ) -> dict[str, Any]:
        item: dict[str, Any] = {
            _PK_FIELD: {"N": str(row_id)},
            _ID_FIELD: {"N": str(row_id)},
            _VECTOR_FIELD: {"L": [{"N": str(v)} for v in embedding]},
        }
        if self.use_partition_key:
            # Uniform (balanced-baseline) distribution: id % N spreads rows
            # evenly across partitions. Real-world keys are usually skewed, so
            # this is the best-case, no-hot-partition scenario.
            item[_PARTITION_FIELD] = {"N": str(row_id % self.num_partitions)}
        if self.with_scalar_labels and labels_data is not None:
            item[_LABEL_FIELD] = {"S": labels_data[i]}
        return item

    def _write_batch_with_retry(self, request_items: dict[str, list[dict]]) -> None:
        """Write one batch, re-driving UnprocessedItems with bounded exponential
        backoff.

        Two layers of throttling defence:
          1. boto3 ``standard`` retry mode (max_attempts=8) retries throttling
             exceptions and 5xx on each BatchWriteItem call with its own
             backoff + jitter.
          2. BatchWriteItem can also succeed (HTTP 200) while returning some
             rows in UnprocessedItems when the table/index is being throttled.
             Those are NOT retried by the SDK, so we re-drive them here. Under
             heavy throttling a large fraction comes back unprocessed, so we
             back off exponentially (with jitter) between re-drives instead of
             hot-looping, and give up after UNPROCESSED_MAX_RETRIES.
        """
        response = self.client.batch_write_item(RequestItems=request_items)
        unprocessed = response.get("UnprocessedItems") or {}

        attempt = 0
        while unprocessed:
            pending = sum(len(v) for v in unprocessed.values())
            if attempt >= UNPROCESSED_MAX_RETRIES:
                msg = (
                    f"BatchWriteItem still has {pending} unprocessed items after "
                    f"{UNPROCESSED_MAX_RETRIES} backoff retries; table is throttling "
                    f"faster than it can drain."
                )
                raise RuntimeError(msg)

            delay = min(UNPROCESSED_MAX_DELAY, UNPROCESSED_BASE_DELAY * (2**attempt))
            delay += random.uniform(0, delay)  # full jitter
            log.debug(
                f"DynamoDB BatchWriteItem throttled: {pending} unprocessed, "
                f"re-drive {attempt + 1}/{UNPROCESSED_MAX_RETRIES} after {delay:.2f}s"
            )
            time.sleep(delay)

            response = self.client.batch_write_item(RequestItems=unprocessed)
            unprocessed = response.get("UnprocessedItems") or {}
            attempt += 1

    # ------------------------------------------------------------------ #
    # Search
    # ------------------------------------------------------------------ #
    def prepare_filter(self, filters: Filter):
        """Build the SearchConditionExpression for the next searches.

        DynamoDB vector search supports ONLY equality (=) in a
        SearchConditionExpression, on HASH and INLINE_FILTER attributes. The
        StrEqual benchmark filter maps to an equality on the inline-filter
        label attribute; NonFilter clears it. NumGE (id >= N) is intentionally
        not advertised in supported_filter_types because the API has no range
        operator for the search condition.
        """
        if filters.type == FilterOp.NonFilter:
            self._condition_expr = None
            self._expr_values = {}
        elif filters.type == FilterOp.StrEqual:
            self._condition_expr = f"{_LABEL_FIELD} = :label"
            self._expr_values = {":label": {"S": filters.label_value}}
        else:
            msg = f"Unsupported filter for DynamoDB vector search - {filters}"
            raise ValueError(msg)

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        timeout: int | None = None,
    ) -> list[int]:
        assert self.client is not None
        top_k = min(k, TOP_K_MAX)

        condition = self._condition_expr
        values = dict(self._expr_values)
        if self.use_partition_key:
            # A SearchVectors call is scoped to ONE partition-key value, which
            # is how the feature is used in practice: the caller searches a
            # single known partition rather than the whole index. We pick a
            # random partition value per query to spread search load evenly
            # across partitions (mirroring the id % N write distribution).
            #
            # NOTE: this searches only ~1/num_partitions of the dataset, while
            # VectorDBBench computes recall against WHOLE-dataset ground truth.
            # Recall is therefore expected to be ~1/num_partitions here; the
            # QPS/latency figures reflect the realistic single-partition access
            # pattern. Use num_partitions=1 for a whole-index recall benchmark.
            part = random.randrange(self.num_partitions)
            part_cond = f"{_PARTITION_FIELD} = :part"
            values[":part"] = {"N": str(part)}
            condition = f"{part_cond} AND {condition}" if condition else part_cond

        results = self._search_one(query, top_k, condition=condition, values=values)
        return [row_id for row_id, _ in results]

    def _search_one(
        self,
        query: list[float],
        top_k: int,
        condition: str | None,
        values: dict[str, Any],
    ) -> list[tuple[int, float]]:
        params: dict[str, Any] = {
            "TableName": self.table_name,
            "IndexName": self.index_name,
            "SearchVector": [{"N": str(v)} for v in query],
            "TopK": top_k,
            "ProjectionExpression": _ID_FIELD,
        }
        if condition:
            params["SearchConditionExpression"] = condition
            params["ExpressionAttributeValues"] = values

        response = self.client.search_vectors(**params)
        return [
            (int(r["Item"][_ID_FIELD]["N"]), float(r.get("Score", 0.0)))
            for r in response.get("SearchResults", [])
        ]
