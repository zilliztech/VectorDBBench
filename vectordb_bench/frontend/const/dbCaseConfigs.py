from enum import IntEnum
import typing
from pydantic import BaseModel
from vectordb_bench.backend.cases import CaseLabel, CaseType
from vectordb_bench.backend.clients import DB
from vectordb_bench.backend.clients.api import IndexType

from vectordb_bench.models import CaseConfigParamType

MAX_STREAMLIT_INT = (1 << 53) - 1

DB_LIST = [d for d in DB if d != DB.Test]

DIVIDER = "DIVIDER"
CASE_LIST_WITH_DIVIDER = [
    CaseType.Performance768D100M,
    CaseType.Performance768D10M,
    CaseType.Performance768D1M,
    DIVIDER,
    CaseType.Performance1536D5M,
    CaseType.Performance1536D500K,
    DIVIDER,
    CaseType.Performance768D10M1P,
    CaseType.Performance768D1M1P,
    DIVIDER,
    CaseType.Performance1536D5M1P,
    CaseType.Performance1536D500K1P,
    DIVIDER,
    CaseType.Performance768D10M99P,
    CaseType.Performance768D1M99P,
    DIVIDER,
    CaseType.Performance1536D5M99P,
    CaseType.Performance1536D500K99P,
    DIVIDER,
    CaseType.CapacityDim960,
    CaseType.CapacityDim128,
]

CASE_LIST = [item for item in CASE_LIST_WITH_DIVIDER if isinstance(item, CaseType)]


class InputType(IntEnum):
    Text = 20001
    Number = 20002
    Option = 20003


class CaseConfigInput(BaseModel):
    label: CaseConfigParamType
    inputType: InputType = InputType.Text
    inputConfig: dict = {}
    inputHelp: str = ""
    displayLabel: str = ""
    # todo type should be a function
    isDisplayed: typing.Any = lambda x: True


CaseConfigParamInput_IndexType = CaseConfigInput(
    label=CaseConfigParamType.IndexType,
    inputType=InputType.Option,
    inputConfig={
        "options": [
            IndexType.HNSW.value,
            IndexType.IVFFlat.value,
            IndexType.IVFSQ8.value,
            IndexType.DISKANN.value,
            IndexType.Flat.value,
            IndexType.AUTOINDEX.value,
            IndexType.GPU_IVF_FLAT.value,
            IndexType.GPU_IVF_PQ.value,
            IndexType.GPU_CAGRA.value,
        ],
    },
)

CaseConfigParamInput_IndexType_PgVector = CaseConfigInput(
    label=CaseConfigParamType.IndexType,
    inputHelp="Select Index Type",
    inputType=InputType.Option,
    inputConfig={
        "options": [
            IndexType.HNSW.value,
            IndexType.IVFFlat.value,
        ],
    },
)

CaseConfigParamInput_M = CaseConfigInput(
    label=CaseConfigParamType.M,
    inputType=InputType.Number,
    inputConfig={
        "min": 4,
        "max": 64,
        "value": 30,
    },
    isDisplayed=lambda config: config.get(CaseConfigParamType.IndexType, None)
    == IndexType.HNSW.value,
)

CaseConfigParamInput_m = CaseConfigInput(
    label=CaseConfigParamType.m,
    inputType=InputType.Number,
    inputConfig={
        "min": 4,
        "max": 64,
        "value": 16,
    },
    isDisplayed=lambda config: config.get(CaseConfigParamType.IndexType, None)
    == IndexType.HNSW.value,
)


CaseConfigParamInput_EFConstruction_Milvus = CaseConfigInput(
    label=CaseConfigParamType.EFConstruction,
    inputType=InputType.Number,
    inputConfig={
        "min": 8,
        "max": 512,
        "value": 360,
    },
    isDisplayed=lambda config: config[CaseConfigParamType.IndexType]
    == IndexType.HNSW.value,
)

CaseConfigParamInput_EFConstruction_Weaviate = CaseConfigInput(
    label=CaseConfigParamType.EFConstruction,
    inputType=InputType.Number,
    inputConfig={
        "min": 8,
        "max": 512,
        "value": 128,
    },
)

CaseConfigParamInput_EFConstruction_ES = CaseConfigInput(
    label=CaseConfigParamType.EFConstruction,
    inputType=InputType.Number,
    inputConfig={
        "min": 8,
        "max": 512,
        "value": 360,
    },
)

CaseConfigParamInput_maintenance_work_mem_PgVector = CaseConfigInput(
    label=CaseConfigParamType.maintenance_work_mem,
    inputHelp="Recommended value: 1.33x the index size, not to exceed the available free memory."
              "Specify in gigabytes. e.g. 8GB",
    inputType=InputType.Text,
    inputConfig={
        "value": "8GB",
    },
)

CaseConfigParamInput_max_parallel_workers_PgVector = CaseConfigInput(
    label=CaseConfigParamType.max_parallel_workers,
    displayLabel="Max parallel workers",
    inputHelp="Recommended value: (cpu cores - 1). This will set the parameters: max_parallel_maintenance_workers,"
              " max_parallel_workers & table(parallel_workers)",
    inputType=InputType.Number,
    inputConfig={
        "min": 0,
        "max": 1024,
        "value": 16,
    },
)


CaseConfigParamInput_EFConstruction_PgVectoRS = CaseConfigInput(
    label=CaseConfigParamType.EFConstruction,
    inputType=InputType.Number,
    inputConfig={
        "min": 8,
        "max": 512,
        "value": 360,
    },
    isDisplayed=lambda config: config[CaseConfigParamType.IndexType]
    == IndexType.HNSW.value,
)

CaseConfigParamInput_EFConstruction_PgVector = CaseConfigInput(
    label=CaseConfigParamType.ef_construction,
    inputType=InputType.Number,
    inputConfig={
        "min": 8,
        "max": 1024,
        "value": 256,
    },
    isDisplayed=lambda config: config[CaseConfigParamType.IndexType]
    == IndexType.HNSW.value,
)


CaseConfigParamInput_M_ES = CaseConfigInput(
    label=CaseConfigParamType.M,
    inputType=InputType.Number,
    inputConfig={
        "min": 4,
        "max": 64,
        "value": 30,
    },
)

CaseConfigParamInput_NumCandidates_ES = CaseConfigInput(
    label=CaseConfigParamType.numCandidates,
    inputType=InputType.Number,
    inputConfig={
        "min": 1,
        "max": 10000,
        "value": 100,
    },
)

CaseConfigParamInput_EF_Milvus = CaseConfigInput(
    label=CaseConfigParamType.EF,
    inputType=InputType.Number,
    inputConfig={
        "min": 100,
        "max": MAX_STREAMLIT_INT,
        "value": 100,
    },
    isDisplayed=lambda config: config[CaseConfigParamType.IndexType]
    == IndexType.HNSW.value,
)

CaseConfigParamInput_EF_Weaviate = CaseConfigInput(
    label=CaseConfigParamType.EF,
    inputType=InputType.Number,
    inputConfig={
        "min": -1,
        "max": MAX_STREAMLIT_INT,
        "value": -1,
    },
)

CaseConfigParamInput_MaxConnections = CaseConfigInput(
    label=CaseConfigParamType.MaxConnections,
    inputType=InputType.Number,
    inputConfig={"min": 1, "max": MAX_STREAMLIT_INT, "value": 64},
)

CaseConfigParamInput_SearchList = CaseConfigInput(
    label=CaseConfigParamType.SearchList,
    inputType=InputType.Number,
    inputConfig={
        "min": 100,
        "max": MAX_STREAMLIT_INT,
        "value": 100,
    },
    isDisplayed=lambda config: config.get(CaseConfigParamType.IndexType, None)
    == IndexType.DISKANN.value,
)

CaseConfigParamInput_Nlist = CaseConfigInput(
    label=CaseConfigParamType.Nlist,
    inputType=InputType.Number,
    inputConfig={
        "min": 1,
        "max": 65536,
        "value": 1024,
    },
    isDisplayed=lambda config: config.get(CaseConfigParamType.IndexType, None)
    in [
        IndexType.IVFFlat.value,
        IndexType.IVFSQ8.value,
        IndexType.GPU_IVF_FLAT.value,
        IndexType.GPU_IVF_PQ.value,
    ],
)

CaseConfigParamInput_Nprobe = CaseConfigInput(
    label=CaseConfigParamType.Nprobe,
    inputType=InputType.Number,
    inputConfig={
        "min": 1,
        "max": 65536,
        "value": 64,
    },
    isDisplayed=lambda config: config.get(CaseConfigParamType.IndexType, None)
    in [
        IndexType.IVFFlat.value,
        IndexType.IVFSQ8.value,
        IndexType.GPU_IVF_FLAT.value,
        IndexType.GPU_IVF_PQ.value,
    ],
)

CaseConfigParamInput_M_PQ = CaseConfigInput(
    label=CaseConfigParamType.m,
    inputType=InputType.Number,
    inputConfig={
        "min": 0,
        "max": 65536,
        "value": 0,
    },
    isDisplayed=lambda config: config.get(CaseConfigParamType.IndexType, None)
    in [IndexType.GPU_IVF_PQ.value],
)

CaseConfigParamInput_Nbits_PQ = CaseConfigInput(
    label=CaseConfigParamType.nbits,
    inputType=InputType.Number,
    inputConfig={
        "min": 1,
        "max": 65536,
        "value": 8,
    },
    isDisplayed=lambda config: config.get(CaseConfigParamType.IndexType, None)
    in [IndexType.GPU_IVF_PQ.value],
)

CaseConfigParamInput_intermediate_graph_degree = CaseConfigInput(
    label=CaseConfigParamType.intermediate_graph_degree,
    inputType=InputType.Number,
    inputConfig={
        "min": 1,
        "max": 65536,
        "value": 64,
    },
    isDisplayed=lambda config: config.get(CaseConfigParamType.IndexType, None)
    in [IndexType.GPU_CAGRA.value],
)

CaseConfigParamInput_graph_degree = CaseConfigInput(
    label=CaseConfigParamType.graph_degree,
    inputType=InputType.Number,
    inputConfig={
        "min": 1,
        "max": 65536,
        "value": 32,
    },
    isDisplayed=lambda config: config.get(CaseConfigParamType.IndexType, None)
    in [IndexType.GPU_CAGRA.value],
)

CaseConfigParamInput_itopk_size = CaseConfigInput(
    label=CaseConfigParamType.itopk_size,
    inputType=InputType.Number,
    inputConfig={
        "min": 1,
        "max": 65536,
        "value": 128,
    },
    isDisplayed=lambda config: config.get(CaseConfigParamType.IndexType, None)
    in [IndexType.GPU_CAGRA.value],
)

CaseConfigParamInput_team_size = CaseConfigInput(
    label=CaseConfigParamType.team_size,
    inputType=InputType.Number,
    inputConfig={
        "min": 0,
        "max": 65536,
        "value": 0,
    },
    isDisplayed=lambda config: config.get(CaseConfigParamType.IndexType, None)
    in [IndexType.GPU_CAGRA.value],
)

CaseConfigParamInput_search_width = CaseConfigInput(
    label=CaseConfigParamType.search_width,
    inputType=InputType.Number,
    inputConfig={
        "min": 0,
        "max": 65536,
        "value": 4,
    },
    isDisplayed=lambda config: config.get(CaseConfigParamType.IndexType, None)
    in [IndexType.GPU_CAGRA.value],
)

CaseConfigParamInput_min_iterations = CaseConfigInput(
    label=CaseConfigParamType.min_iterations,
    inputType=InputType.Number,
    inputConfig={
        "min": 0,
        "max": 65536,
        "value": 0,
    },
    isDisplayed=lambda config: config.get(CaseConfigParamType.IndexType, None)
    in [IndexType.GPU_CAGRA.value],
)

CaseConfigParamInput_max_iterations = CaseConfigInput(
    label=CaseConfigParamType.max_iterations,
    inputType=InputType.Number,
    inputConfig={
        "min": 0,
        "max": 65536,
        "value": 0,
    },
    isDisplayed=lambda config: config.get(CaseConfigParamType.IndexType, None)
    in [IndexType.GPU_CAGRA.value],
)

CaseConfigParamInput_build_algo = CaseConfigInput(
    label=CaseConfigParamType.build_algo,
    inputType=InputType.Option,
    inputConfig={
        "options": ["IVF_PQ", "NN_DESCENT"],
    },
    isDisplayed=lambda config: config.get(CaseConfigParamType.IndexType, None)
    in [IndexType.GPU_CAGRA.value],
)


CaseConfigParamInput_cache_dataset_on_device = CaseConfigInput(
    label=CaseConfigParamType.cache_dataset_on_device,
    inputType=InputType.Option,
    inputConfig={
        "options": ["false", "true"],
    },
    isDisplayed=lambda config: config.get(CaseConfigParamType.IndexType, None)
    in [
        IndexType.GPU_CAGRA.value,
        IndexType.GPU_IVF_PQ.value,
        IndexType.GPU_IVF_FLAT.value,
    ],
)

CaseConfigParamInput_refine_ratio = CaseConfigInput(
    label=CaseConfigParamType.refine_ratio,
    inputType=InputType.Number,
    inputConfig={
        "min": 1.0,
        "max": 2.0,
        "value": 1.0,
        "step": 0.01,
    },
    isDisplayed=lambda config: config.get(CaseConfigParamType.IndexType, None)
    in [
        IndexType.GPU_CAGRA.value,
        IndexType.GPU_IVF_PQ.value,
        IndexType.GPU_IVF_FLAT.value,
    ],
)

CaseConfigParamInput_Lists = CaseConfigInput(
    label=CaseConfigParamType.lists,
    inputType=InputType.Number,
    inputConfig={
        "min": 1,
        "max": 65536,
        "value": 10,
    },
    isDisplayed=lambda config: config.get(CaseConfigParamType.IndexType, None)
    in [IndexType.IVFFlat.value],
)

CaseConfigParamInput_Probes = CaseConfigInput(
    label=CaseConfigParamType.probes,
    inputType=InputType.Number,
    inputConfig={
        "min": 1,
        "max": 65536,
        "value": 1,
    },
)

CaseConfigParamInput_Lists_PgVector = CaseConfigInput(
    label=CaseConfigParamType.lists,
    inputType=InputType.Number,
    inputConfig={
        "min": 1,
        "max": 65536,
        "value": 10,
    },
    isDisplayed=lambda config: config.get(CaseConfigParamType.IndexType, None)
    == IndexType.IVFFlat.value,
)

CaseConfigParamInput_Probes_PgVector = CaseConfigInput(
    label=CaseConfigParamType.probes,
    inputType=InputType.Number,
    inputConfig={
        "min": 1,
        "max": 65536,
        "value": 1,
    },
    isDisplayed=lambda config: config.get(CaseConfigParamType.IndexType, None)
    == IndexType.IVFFlat.value,
)

CaseConfigParamInput_EFSearch_PgVector = CaseConfigInput(
    label=CaseConfigParamType.ef_search,
    inputType=InputType.Number,
    inputConfig={
        "min": 1,
        "max": 2048,
        "value": 256,
    },
    isDisplayed=lambda config: config.get(CaseConfigParamType.IndexType, None)
    == IndexType.HNSW.value,
)

CaseConfigParamInput_QuantizationType_PgVectoRS = CaseConfigInput(
    label=CaseConfigParamType.quantizationType,
    inputType=InputType.Option,
    inputConfig={
        "options": ["trivial", "scalar", "product"],
    },
    isDisplayed=lambda config: config.get(CaseConfigParamType.IndexType, None)
    in [
        IndexType.HNSW.value,
        IndexType.IVFFlat.value,
    ],
)

CaseConfigParamInput_QuantizationRatio_PgVectoRS = CaseConfigInput(
    label=CaseConfigParamType.quantizationRatio,
    inputType=InputType.Option,
    inputConfig={
        "options": ["x4", "x8", "x16", "x32", "x64"],
    },
    isDisplayed=lambda config: config.get(CaseConfigParamType.quantizationType, None)
    == "product" and config.get(CaseConfigParamType.IndexType, None)
    in [
        IndexType.HNSW.value,
        IndexType.IVFFlat.value,
    ],
)

CaseConfigParamInput_ZillizLevel = CaseConfigInput(
    label=CaseConfigParamType.level,
    inputType=InputType.Number,
    inputConfig={
        "min": 1,
        "max": 3,
        "value": 1,
    },
)

MilvusLoadConfig = [
    CaseConfigParamInput_IndexType,
    CaseConfigParamInput_M,
    CaseConfigParamInput_EFConstruction_Milvus,
    CaseConfigParamInput_Nlist,
    CaseConfigParamInput_M_PQ,
    CaseConfigParamInput_Nbits_PQ,
    CaseConfigParamInput_intermediate_graph_degree,
    CaseConfigParamInput_graph_degree,
    CaseConfigParamInput_build_algo,
    CaseConfigParamInput_cache_dataset_on_device,
]
MilvusPerformanceConfig = [
    CaseConfigParamInput_IndexType,
    CaseConfigParamInput_M,
    CaseConfigParamInput_EFConstruction_Milvus,
    CaseConfigParamInput_EF_Milvus,
    CaseConfigParamInput_SearchList,
    CaseConfigParamInput_Nlist,
    CaseConfigParamInput_Nprobe,
    CaseConfigParamInput_M_PQ,
    CaseConfigParamInput_Nbits_PQ,
    CaseConfigParamInput_intermediate_graph_degree,
    CaseConfigParamInput_graph_degree,
    CaseConfigParamInput_itopk_size,
    CaseConfigParamInput_team_size,
    CaseConfigParamInput_search_width,
    CaseConfigParamInput_min_iterations,
    CaseConfigParamInput_max_iterations,
    CaseConfigParamInput_build_algo,
    CaseConfigParamInput_cache_dataset_on_device,
    CaseConfigParamInput_refine_ratio,
]

WeaviateLoadConfig = [
    CaseConfigParamInput_MaxConnections,
    CaseConfigParamInput_EFConstruction_Weaviate,
]
WeaviatePerformanceConfig = [
    CaseConfigParamInput_MaxConnections,
    CaseConfigParamInput_EFConstruction_Weaviate,
    CaseConfigParamInput_EF_Weaviate,
]

ESLoadingConfig = [CaseConfigParamInput_EFConstruction_ES, CaseConfigParamInput_M_ES]
ESPerformanceConfig = [
    CaseConfigParamInput_EFConstruction_ES,
    CaseConfigParamInput_M_ES,
    CaseConfigParamInput_NumCandidates_ES,
]

PgVectorLoadingConfig = [CaseConfigParamInput_IndexType_PgVector,
                         CaseConfigParamInput_Lists_PgVector,
                         CaseConfigParamInput_m,
                         CaseConfigParamInput_EFConstruction_PgVector,
                         CaseConfigParamInput_maintenance_work_mem_PgVector,
                         CaseConfigParamInput_max_parallel_workers_PgVector,
                         ]
PgVectorPerformanceConfig = [CaseConfigParamInput_IndexType_PgVector,
                             CaseConfigParamInput_m,
                             CaseConfigParamInput_EFConstruction_PgVector,
                             CaseConfigParamInput_EFSearch_PgVector,
                             CaseConfigParamInput_Lists_PgVector,
                             CaseConfigParamInput_Probes_PgVector,
                             CaseConfigParamInput_maintenance_work_mem_PgVector,
                             CaseConfigParamInput_max_parallel_workers_PgVector,
                             ]

PgVectoRSLoadingConfig = [
    CaseConfigParamInput_IndexType,
    CaseConfigParamInput_M,
    CaseConfigParamInput_EFConstruction_PgVectoRS,
    CaseConfigParamInput_Nlist,
    CaseConfigParamInput_QuantizationType_PgVectoRS,
    CaseConfigParamInput_QuantizationRatio_PgVectoRS,
]

PgVectoRSPerformanceConfig = [
    CaseConfigParamInput_IndexType,
    CaseConfigParamInput_M,
    CaseConfigParamInput_EFConstruction_PgVectoRS,
    CaseConfigParamInput_Nlist,
    CaseConfigParamInput_Nprobe,
    CaseConfigParamInput_QuantizationType_PgVectoRS,
    CaseConfigParamInput_QuantizationRatio_PgVectoRS,
]

ZillizCloudPerformanceConfig = [
    CaseConfigParamInput_ZillizLevel,
]

CASE_CONFIG_MAP = {
    DB.Milvus: {
        CaseLabel.Load: MilvusLoadConfig,
        CaseLabel.Performance: MilvusPerformanceConfig,
    },
    DB.ZillizCloud: {
        CaseLabel.Performance: ZillizCloudPerformanceConfig,
    },
    DB.WeaviateCloud: {
        CaseLabel.Load: WeaviateLoadConfig,
        CaseLabel.Performance: WeaviatePerformanceConfig,
    },
    DB.ElasticCloud: {
        CaseLabel.Load: ESLoadingConfig,
        CaseLabel.Performance: ESPerformanceConfig,
    },
    DB.PgVector: {
        CaseLabel.Load: PgVectorLoadingConfig,
        CaseLabel.Performance: PgVectorPerformanceConfig,
    },
    DB.PgVectoRS: {
        CaseLabel.Load: PgVectoRSLoadingConfig,
        CaseLabel.Performance: PgVectoRSPerformanceConfig,
    },
}
