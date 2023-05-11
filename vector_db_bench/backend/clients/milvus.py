"""Wrapper around the Milvus vector database over VectorDB"""

import logging
from uuid import uuid4
from typing import Any, Optional, Iterable, Union

from pydantic import BaseModel
from . import (
    IndexType,
    MetricType,
    DBCaseConfig,
    VectorDB,
)

log = logging.getLogger(__name__)

DEFAULT_MILVUS_CONNECTION = {
    "host": "localhost",
    "port": "19530",
    "user": "",
    "password": "",
    "secure": False,
}


class Milvus(VectorDB):
    # TODO normalize vectors for COSIN metric and use L2 for COSIN

    def __init__(
        self,
        collection_name: str = "VectorDBBenchCollection",
        connection_args: dict[str, Any] | None = None,
        consistency_level: str = "Session",
        index_params: dict | None = None,
        search_params: dict | None = None,
        drop_old: bool = False,
    ):
        """Initialize wrapper around the milvus vector database.

        In order to use this you need to have `pymilvus` installed and a
        running Milvus/Zilliz Cloud instance.
        """
        try:
            from pymilvus import Collection, utility
        except ImportError:
            raise ValueError(
                "Could not import pymilvus python package. "
                "Please install it with `pip install pymilvus`."
            ) from None

        # Default search params when one is not provided.
        self.default_search_params = {
            "IVF_FLAT": {"metric_type": "L2", "params": {"nprobe": 10}},
            "IVF_SQ8": {"metric_type": "L2", "params": {"nprobe": 10}},
            "IVF_PQ": {"metric_type": "L2", "params": {"nprobe": 10}},
            "HNSW": {"metric_type": "L2", "params": {"ef": 10}},
            "RHNSW_FLAT": {"metric_type": "L2", "params": {"ef": 10}},
            "RHNSW_SQ": {"metric_type": "L2", "params": {"ef": 10}},
            "RHNSW_PQ": {"metric_type": "L2", "params": {"ef": 10}},
            "IVF_HNSW": {"metric_type": "L2", "params": {"nprobe": 10, "ef": 10}},
            "ANNOY": {"metric_type": "L2", "params": {"search_k": 10}},
            "AUTOINDEX": {"metric_type": "L2", "params": {}},
        }

        self.collection_name = collection_name
        self.index_params = index_params
        self.search_params = search_params
        self.consistency_level = consistency_level

        # In order for a collection to be compatible, pk needs to be auto'id and int
        self._primary_field = "pk"
        # In orer for compatbility, the vector field needs to be called "vector"
        self._vector_field = "vector"
        self.fields: list[str] = []
        # Create the connection to the server
        if connection_args is None:
            connection_args = DEFAULT_MILVUS_CONNECTION
        self.alias = self._create_connection_alias(connection_args)
        self.col: Optional[Collection] = None

        # Grab the existing colection if it exists
        if utility.has_collection(self.collection_name, using=self.alias):
            self.col = Collection(
                self.collection_name,
                using=self.alias,
            )
        # If need to drop old, drop it
        if drop_old and isinstance(self.col, Collection):
            self.col.drop()
            self.col = None

        # Initialize the milvus client
        self._init()

    def _create_connection_alias(self, connection_args: dict) -> str:
        """Create the connection to the Milvus server."""
        from pymilvus import MilvusException, connections

        # Grab the connection arguments that are used for checking existing connection
        host: str = connection_args.get("host", None)
        port: Union[str, int] = connection_args.get("port", None)
        address: str = connection_args.get("address", None)
        uri: str = connection_args.get("uri", None)
        user = connection_args.get("user", None)

        # Order of use is host/port, uri, address
        if host is not None and port is not None:
            given_address = str(host) + ":" + str(port)
        elif uri is not None:
            given_address = uri.split("https://")[1]
        elif address is not None:
            given_address = address
        else:
            given_address = None
            log.debug("Missing standard address type for reuse atttempt")

        # User defaults to empty string when getting connection info
        if user is not None:
            tmp_user = user
        else:
            tmp_user = ""

        # If a valid address was given, then check if a connection exists
        if given_address is not None:
            for con in connections.list_connections():
                addr = connections.get_connection_addr(con[0])
                if (
                    con[1]
                    and ("address" in addr)
                    and (addr["address"] == given_address)
                    and ("user" in addr)
                    and (addr["user"] == tmp_user)
                ):
                    log.debug("Using previous connection: %s", con[0])
                    return con[0]

        # Generate a new connection if one doesnt exist
        alias = uuid4().hex
        try:
            connections.connect(alias=alias, **connection_args)
            log.debug("Created new connection using: %s", alias)
            return alias
        except MilvusException as e:
            log.warn("Failed to create new connection using: %s", alias)
            raise e

    def _init(
        self,
        embeddings: list[list[float]] | None = None,
        metadatas: list[dict] | None = None,
    ) -> None:
        if embeddings is not None:
            self._create_collection(embeddings, metadatas)
        self._extract_fields()
        self._create_index()
        self._create_search_params()
        self._load()

    def _create_collection(
        self, embeddings: list[list[float]], metadatas: Optional[list[dict]] = None
    ) -> None:
        from pymilvus import (
            Collection,
            CollectionSchema,
            DataType,
            FieldSchema,
            MilvusException,
        )
        from pymilvus.orm.types import infer_dtype_bydata

        # Determine embedding dim
        dim = len(embeddings[0])
        fields = []
        # Determine metadata schema
        if metadatas:
            # Create FieldSchema for each entry in metadata.
            for key, value in metadatas[0].items():
                # Infer the corresponding datatype of the metadata
                dtype = infer_dtype_bydata(value)
                # Datatype isnt compatible
                if dtype == DataType.UNKNOWN or dtype == DataType.NONE:
                    log.warn(
                        "Failure to create collection, unrecognized dtype for key: %s",
                        key,
                    )
                    raise ValueError(f"Unrecognized datatype for {key}.")
                # Dataype is a string/varchar equivalent
                elif dtype == DataType.VARCHAR:
                    fields.append(FieldSchema(key, DataType.VARCHAR, max_length=65_535))
                else:
                    fields.append(FieldSchema(key, dtype))

        # Create the primary key field
        fields.append(
            FieldSchema(
                self._primary_field, DataType.INT64, is_primary=True, auto_id=True
            )
        )
        # Create the vector field, supports binary or float vectors
        fields.append(
            FieldSchema(self._vector_field, infer_dtype_bydata(embeddings[0]), dim=dim)
        )

        # Create the schema for the collection
        schema = CollectionSchema(fields)
        log.info(f"collection schema: {schema}")

        # Create the collection
        try:
            self.col = Collection(
                name=self.collection_name,
                schema=schema,
                consistency_level=self.consistency_level,
                using=self.alias,
            )
        except MilvusException as e:
            log.warn(
                "Failed to create collection: %s error: %s", self.collection_name, e
            )
            raise e

    def _extract_fields(self) -> None:
        """Grab the existing fields from the Collection"""
        from pymilvus import Collection

        if isinstance(self.col, Collection):
            schema = self.col.schema
            for x in schema.fields:
                self.fields.append(x.name)
            # Since primary field is auto-id, no need to track it
            self.fields.remove(self._primary_field)

    def _get_index(self) -> Optional[dict[str, Any]]:
        """Return the vector index information if it exists"""
        from pymilvus import Collection

        if isinstance(self.col, Collection):
            for x in self.col.indexes:
                if x.field_name == self._vector_field:
                    return x.to_dict()
        return None

    def _create_index(self) -> None:
        """Create a index on the collection"""
        from pymilvus import Collection, MilvusException

        if isinstance(self.col, Collection) and self._get_index() is None:
            try:
                # If no index params, use a default HNSW based one
                if self.index_params is None:
                    self.index_params = {
                        "metric_type": "COSINE", # todo
                        "index_type": "HNSW",
                        "params": {"M": 8, "efConstruction": 64},
                    }

                self.col.create_index(
                    self._vector_field,
                    index_params=self.index_params,
                    using=self.alias,
                )

            except MilvusException as e:
                log.warn(
                    "Failed to create an index on collection: %s", self.collection_name
                )
                raise e

    def _create_search_params(self) -> None:
        """Generate search params based on the current index type"""
        from pymilvus import Collection

        if isinstance(self.col, Collection) and self.search_params is None:
            index = self._get_index()
            if index is not None:
                index_type: str = index["index_param"]["index_type"]
                metric_type: str = index["index_param"]["metric_type"]
                self.search_params = self.default_search_params[index_type]
                self.search_params["metric_type"] = metric_type

    def _load(self) -> None:
        """Load the collection if available."""
        from pymilvus import Collection

        if isinstance(self.col, Collection) and self._get_index() is not None:
            self.col.load()

    def insert_embeddings(
        self,
        embeddings: Iterable[list[float]],
        metadatas: Optional[list[dict]] = None,
        **kwargs: Any,
    ) -> list[str]:
        """Insert embeddings into Milvus.

        Inserting data when the collection has not be made yet will result
        in creating a new Collection. The data of the first entity decides
        the schema of the new collection, the dim is extracted from the first
        embedding and the columns are decided by the first metadata dict.
        Metada keys will need to be present for all inserted values. At
        the moment there is no None equivalent in Milvus.

        Args:
            embeddings(Iterable[list[float]]): list of embedding to add to the vector database.
            metadatas(list[dict], Optional): metadatas associated with the embeddings. Defaults to None.
            kwargs(Any): vector database specific parameters.

        Raises:
            MilvusException: Failure to add embeddings

        Returns:
            list[str]: The resulting ids for each inserted element.
        """
        from pymilvus import Collection, MilvusException
        # If the collection hasnt been initialized yet, perform all steps to do so
        if not isinstance(self.col, Collection):
            self._init(embeddings, metadatas)

        # Dict to hold all insert columns
        insert_dict: dict[str, list] = {
            self._vector_field: embeddings,
        }

        # Collect the metadata into the insert dict.
        if metadatas is not None:
            for d in metadatas:
                for key, value in d.items():
                    if key in self.fields:
                        insert_dict.setdefault(key, []).append(value)

        pks: list[str] = []
        assert isinstance(self.col, Collection)
        insert_list = [insert_dict[x] for x in self.fields]
        try:
            res = self.col.insert(insert_list, **kwargs)
            pks.extend(res.primary_keys)
        except MilvusException as e:
            log.warn("Failed to insert data")
            raise e
        return pks

    def search_embedding_with_score(
        self,
        query: list[float],
        k: int = 100,
        filters: Any | None = None,
        param: dict | None  = None,
        timeout: int | None = None,
        **kwargs: Any,
    ) -> list[tuple[int, float]]:
        """Perform a search on a query embedding and return results with score.

        For more information about the search parameters, take a look at the pymilvus
        documentation found here:
        https://milvus.io/api-reference/pymilvus/v2.2.8/Collection/search().md

        Args:
            query(list[float]): query embedding to look up documents similar to.
            k(int): Number of most similar embeddings to return. Defaults to 100.
            filters(Any, optional): filtering expression to filter the data while searching.
            param (dict): The search params for the specified index. Defaults to None.
            timeout (int, optional): How long to wait before timeout error. Defaults to None.
            kwargs: Collection.search() keyword arguments.

        Returns:
            list[tuple[int, float]]: Result embedding's id and score.
        """
        if self.col is None:
            log.debug("No existing collection to search.")
            return []

        if param is None:
            param = self.search_params

        # Determine result metadata fields.
        output_fields = self.fields[:]
        output_fields.remove(self._vector_field)
        param["metric_type"]= "COSINE"
        log.debug(f"search param: {param}")

        # Perform the search.
        res = self.col.search(
            data=[query],
            anns_field=self._vector_field,
            param=param,
            limit=k,
            expr=filters,
            output_fields=output_fields,
            timeout=timeout,
            **kwargs,
        )
        # Organize results.
        ret = [(result.id, result.score) for result in res[0]]
        return ret

    def ready_to_search(self):
        # TODO wait for create index done
        return


class MilvusIndexConfig(BaseModel):
    index: IndexType
    metric_type: MetricType


class HNSWConfig(MilvusIndexConfig, DBCaseConfig):
    M: int
    efConstruction: int
    ef: int | None = None
    index: IndexType = IndexType.HNSW

    def index_param(self) -> dict:
        return {
            "metric_type": self.metric_type,
            "index_type": self.index,
            "params": {"M": self.M, "efConstruction": self.efConstruction},
        }

    def search_param(self) -> dict:
        return {
            "metric_type": self.metric_type,
            "params": {"ef": self.ef},
        }

class DISKANNConfig(MilvusIndexConfig, DBCaseConfig):
    search_list: int | None = None
    index: IndexType = IndexType.DISKANN

    def index_param(self) -> dict:
        return {
            "metric_type": self.metric_type,
            "index_type": self.index,
            "params": {},
        }

    def search_param(self) -> dict:
        return {
            "metric_type": self.metric_type,
            "params": {"search_list": self.search_list},
        }

class IVFFlatConfig(DBCaseConfig, DBCaseConfig):
    nlist: int
    nprobe: int | None = None
    index: IndexType = IndexType.IVFFlat

    def index_param(self) -> dict:
        return {
            "metric_type": self.metric_type,
            "index_type": self.index,
            "params": {"nlist": self.nlist},
        }

    def search_param(self) -> dict:
        return {
            "metric_type": self.metric_type,
            "params": {"nprobe": self.nprobe},
        }


class FLATConfig(DBCaseConfig, DBCaseConfig):
    index: IndexType = IndexType.IVFFlat

    def index_param(self) -> dict:
        return {
            "metric_type": self.metric_type,
            "index_type": self.index,
            "params": {},
        }

    def search_param(self) -> dict:
        return {
            "metric_type": self.metric_type,
            "params": {},
        }


