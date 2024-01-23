import logging
from contextlib import contextmanager
from typing import Type
from ..api import VectorDB, DBCaseConfig, DBConfig, IndexType
from .config import KnowhereConfig, KnowhereIndexConfig
import pathlib
import json

log = logging.getLogger(__name__)


class Knowhere(VectorDB):
    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: KnowhereIndexConfig,
        drop_old: bool = False,
        name: str = "Knowhere",
        tmp_dir_path="./vectordb_bench/results/tmp_knowhere/",
        **kwargs,
    ):
        self.name = name
        self.db_config = db_config
        self.case_config = db_case_config
        self.dim = dim
        self.config = json.loads(f'{{{self.db_config.get("config")}}}')
        self.config["dim"] = dim
        self.build_threads = db_config.get("build_threads", 4)
        self.search_threads = db_config.get("search_threads", 4)

        import knowhere
        
        knowhere.SetBuildThreadPool(self.build_threads)
        knowhere.SetSearchThreadPool(self.search_threads)

        self.version = knowhere.GetCurrentVersion()
        tmp_dir = pathlib.Path(tmp_dir_path)
        if not tmp_dir.exists():
            tmp_dir.mkdir(parents=True)
        self.indexFile = tmp_dir_path + (
            self.db_config.get("index_type")
            + f"_{db_case_config.metric_type.value}_{dim}d_"
            + db_config.get("config")
            .replace('"', "")
            .replace(" ", "")
            .replace(":", "_")
            .replace(",", "_")
            + ".index"
        )

        self.index = None
        self.bitset = None

    @classmethod
    def config_cls(cls) -> Type[DBConfig]:
        return KnowhereConfig

    @classmethod
    def case_config_cls(cls, index_type: IndexType | None = None) -> Type[DBCaseConfig]:
        return KnowhereIndexConfig

    @contextmanager
    def init(self) -> None:
        import knowhere
        
        knowhere.SetBuildThreadPool(self.build_threads)
        knowhere.SetSearchThreadPool(self.search_threads)

        index = knowhere.CreateIndex(self.db_config.get("index_type"), self.version)
        filePath = pathlib.Path(self.indexFile)
        if filePath.exists():
            log.info(
                f"Index file existed; Load the index file and Deserialize; {self.indexFile}"
            )
            indexBinarySet = knowhere.GetBinarySet()
            knowhere.Load(indexBinarySet, self.indexFile)
            index.Deserialize(indexBinarySet)
            log.info(f"Index ready")
        self.index = index

        yield
        self.index = None
        self.bitset = None

    def insert_embeddings(
        self,
        embeddings,
        metadata: list[int],
        **kwargs,
    ) -> (int, Exception):
        log.info(f"Start building index with {len(embeddings)} vectors")
        import knowhere
        
        knowhere.SetBuildThreadPool(self.build_threads)
        knowhere.SetSearchThreadPool(self.search_threads)

        data = knowhere.ArrayToDataSet(embeddings)
        self.config.update(self.case_config.index_param())
        log.info(
            f"Build config: {self.config}, {self.db_config.get('index_type')}, {self.version}"
        )
        index = knowhere.CreateIndex(self.db_config.get("index_type"), self.version)
        index.Build(data, json.dumps(self.config))
        indexBinarySet = knowhere.GetBinarySet()
        log.info(f"Serialize the trained index to BinarySet")
        index.Serialize(indexBinarySet)
        log.info(f"Dump the BinarySet to file - {self.indexFile}")
        knowhere.Dump(indexBinarySet, self.indexFile)
        log.info(f"Dump Finished")

        return len(embeddings), None

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
    ) -> list[int]:
        raise NotImplementedError

    def search_embeddings(
        self,
        queryData,  # ndarray
        k: int = 100,
        filters: dict | None = None,
    ) -> list[int]:
        import knowhere

        query = knowhere.ArrayToDataSet(queryData)
        self.config.update(self.case_config.search_param())
        self.config["k"] = k
        bitset = (
            self.bitset.GetBitSetView() if self.bitset else knowhere.GetNullBitSetView()
        )
        log.info(f"search config: {self.config}")
        ans, _ = self.index.Search(query, json.dumps(self.config), bitset)
        k_dis, k_ids = knowhere.DataSetToArray(ans)
        return k_ids.tolist()

    def convert_to_bitset(self, valid_ids: list[int]):
        import knowhere

        log.info("Convert valid ids to knowher-bitset")
        rowCount = self.index.Count()
        bitset = knowhere.CreateBitSet(rowCount)
        invalid_ids = set(range(rowCount)) - set(valid_ids)
        for invalid_id in invalid_ids:
            bitset.SetBit(invalid_id)
        self.bitset = bitset
        log.info("Convert Finished")

    def optimize(self):
        pass

    def ready_to_load(self):
        pass
