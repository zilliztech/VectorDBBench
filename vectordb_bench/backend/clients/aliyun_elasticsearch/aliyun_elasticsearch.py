from ..elastic_cloud.config import ElasticCloudIndexConfig
from ..elastic_cloud.elastic_cloud import ElasticCloud


class AliyunElasticsearch(ElasticCloud):
    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: ElasticCloudIndexConfig,
        indice: str = "vdb_bench_indice",  # must be lowercase
        id_col_name: str = "id",
        vector_col_name: str = "vector",
        drop_old: bool = False,
        **kwargs,
    ):
        super().__init__(
            dim=dim,
            db_config=db_config,
            db_case_config=db_case_config,
            indice=indice,
            id_col_name=id_col_name,
            vector_col_name=vector_col_name,
            drop_old=drop_old,
            **kwargs,
        )
