"""VolcanoMilvus client implementation.

Extends the base Milvus client with Volcano-specific behaviors:
- Expose ``load_reqs_size`` to tune insert batch size (avoid gRPC payload limits).
- Expose ``load_after_compaction`` to choose collection load timing:
  - ``False`` (default): ``load_collection`` is called right after collection creation;
    ``optimize()`` ends with ``refresh_load``.
  - ``True``: ``load_collection`` is deferred until after compaction & index wait
    in ``optimize()``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pymilvus import DataType, MilvusClient

from ..milvus.milvus import MILVUS_FORCE_MERGE_TARGET_SIZE_MB, MILVUS_LOAD_REQS_SIZE, Milvus

if TYPE_CHECKING:
    from .config import MilvusIndexConfig

log = logging.getLogger(__name__)


class VolcanoMilvus(Milvus):
    # Insert concurrency is handled by the framework (ConcurrentInsertRunner).
    # This client does not manage its own multiprocessing pool.
    thread_safe: bool = True

    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: MilvusIndexConfig,
        collection_name: str = "VDBBench",
        drop_old: bool = False,
        name: str = "VolcanoMilvus",
        with_scalar_labels: bool = False,
        **kwargs,
    ):
        """Initialize wrapper around the volcano milvus vector database.

        The vector index is created right after collection creation (same as the
        base ``Milvus`` client). Whether ``load_collection`` is called here or
        deferred to ``optimize()`` is controlled by ``load_after_compaction``.
        """
        self.name = name
        self.db_config = db_config
        self.case_config = db_case_config
        self.collection_name = collection_name
        self.load_reqs_size = int(self.db_config.get("load_reqs_size", MILVUS_LOAD_REQS_SIZE))
        self.batch_size = max(1, int(self.load_reqs_size / (dim * 4)))
        self.with_scalar_labels = with_scalar_labels
        self.load_after_compaction = bool(self.db_config.get("load_after_compaction", False))

        self._primary_field = "pk"
        self._scalar_id_field = "id"
        self._scalar_label_field = "label"
        self._scalar_payload_label_field = self._scalar_label_field
        self._multitenant_partition_key_field = self._scalar_label_field
        self.multitenant_tenant_labels: list[str] = kwargs.get("multitenant_tenant_labels", [])
        if self.multitenant_tenant_labels:
            self._multitenant_partition_key_field = "labels"
            if self.with_scalar_labels:
                self._scalar_payload_label_field = "scalar_label"
        self._vector_field = "vector"
        self._vector_index_name = "vector_idx"
        self._scalar_id_index_name = "id_sort_idx"
        self._scalar_labels_index_name = "labels_idx"

        client = MilvusClient(
            uri=self.db_config.get("uri"),
            user=self.db_config.get("user"),
            password=self.db_config.get("password"),
            # VolcanoMilvusConfig does not expose `token` (Volcano uses uri/user/password
            # auth). Pass through `db_config["token"]` if a future config adds it; default
            # to "" to stay compatible with the upstream Milvus client signature.
            token=self.db_config.get("token", ""),
            timeout=30,
        )

        if drop_old and client.has_collection(self.collection_name):
            log.info(f"{self.name} client drop_old collection: {self.collection_name}")
            client.drop_collection(self.collection_name)

        if not client.has_collection(self.collection_name):
            schema = MilvusClient.create_schema()
            schema.add_field(self._primary_field, DataType.INT64, is_primary=True)
            schema.add_field(self._scalar_id_field, DataType.INT64)
            schema.add_field(self._vector_field, DataType.FLOAT_VECTOR, dim=dim)

            if self.multitenant_tenant_labels:
                schema.add_field(
                    self._multitenant_partition_key_field,
                    DataType.VARCHAR,
                    max_length=256,
                    is_partition_key=True,
                )

            if self.with_scalar_labels:
                is_partition_key = db_case_config.use_partition_key
                log.info(f"with_scalar_labels, add a new varchar field, as partition_key: {is_partition_key}")
                if not self.multitenant_tenant_labels or (
                    self._scalar_payload_label_field != self._multitenant_partition_key_field
                ):
                    schema.add_field(
                        self._scalar_payload_label_field,
                        DataType.VARCHAR,
                        max_length=256,
                        is_partition_key=is_partition_key and not self.multitenant_tenant_labels,
                    )

            log.info(f"{self.name} create collection: {self.collection_name}")

            index_params = self._build_index_params()
            client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                num_shards=self.db_config.get("num_shards", 1),
                consistency_level="Session",
            )
            client.create_index(self.collection_name, index_params)
            if not self.load_after_compaction:
                self._apply_load_with_properties(client)

        client.close()

    def need_normalize_cosine(self) -> bool:
        """Wheather this database need to normalize dataset to support COSINE"""
        log.info("cosine dataset need normalize.")
        return True

    def _apply_load_with_properties(self, client: MilvusClient) -> None:
        """Apply Volcano-specific knowhere.* properties and load the collection.

        The properties (currently ``knowhere.enable_thp`` / ``knowhere.enable_prefetch``)
        have to be set via ``alter_collection_properties`` before ``load_collection``
        so the segment loader picks them up. We keep the alter+load pair in one helper
        to make the intent explicit and avoid name-shadowing with
        ``MilvusClient.load_collection``.
        """
        properties = self.case_config.load_param()
        if properties:
            client.alter_collection_properties(
                collection_name=self.collection_name,
                properties=properties,
            )

        collection_properties = client.describe_collection(self.collection_name).get("properties")
        log.info(f"set collection properties to: {collection_properties}")

        client.load_collection(
            self.collection_name,
            replica_number=self.db_config.get("replica_number", 1),
        )

    def _optimize(self):
        """Volcano-specific optimize.

        Overrides the parent ``_optimize`` to control collection-load timing:
          - ``load_after_compaction=True``: collection is loaded here (after compaction
            and index wait) via ``_apply_load_with_properties``;
          - ``load_after_compaction=False`` (default): collection was already loaded in
            ``__init__``, so we end with ``refresh_load`` like the upstream Milvus.

        Otherwise mirrors the upstream control flow (flush -> wait segments sorted ->
        wait index -> compact -> wait compaction -> wait index again), including the
        ``is_gpu_index`` skip branch and the ``PERMISSION_DENIED`` fallback.
        """
        log.info(f"{self.name} optimizing before search")
        try:
            self.client.flush(self.collection_name)

            try:
                self._wait_for_segments_sorted()
                compaction_id = self.client.compact(
                    self.collection_name,
                    target_size=MILVUS_FORCE_MERGE_TARGET_SIZE_MB,
                )
                if compaction_id > 0:
                    self._wait_for_compaction(compaction_id)
                log.info(f"{self.name} force merge compaction completed.")
            except Exception as e:
                log.warning(f"{self.name} compact or list segments error: {e}")
                if getattr(getattr(e, "code", None), "name", None) == "PERMISSION_DENIED":
                    log.warning("Skip compact due to list segments or compact permission denied.")
                else:
                    raise e from None

            # wait for index no matter what
            self._wait_for_index()
            if self.load_after_compaction:
                self._apply_load_with_properties(self.client)
            else:
                self.client.refresh_load(self.collection_name)
        except Exception as e:
            log.warning(f"{self.name} optimize error: {e}")
            raise e from None
