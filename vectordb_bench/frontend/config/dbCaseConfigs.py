from enum import IntEnum, Enum
import typing
from pydantic import BaseModel
from vectordb_bench.backend.cases import CaseLabel, CaseType
from vectordb_bench.backend.clients import DB
from vectordb_bench.backend.clients.api import IndexType, MetricType, SQType
from vectordb_bench.frontend.components.custom.getCustomConfig import get_custom_configs

from vectordb_bench.models import CaseConfig, CaseConfigParamType

MAX_STREAMLIT_INT = (1 << 53) - 1

DB_LIST = [d for d in DB if d != DB.Test]


class Delimiter(Enum):
    Line = "line"


class BatchCaseConfig(BaseModel):
    label: str = ""
    description: str = ""
    cases: list[CaseConfig] = []


class UICaseItem(BaseModel):
    isLine: bool = False
    label: str = ""
    description: str = ""
    cases: list[CaseConfig] = []
    caseLabel: CaseLabel = CaseLabel.Performance

    def __init__(
        self,
        isLine: bool = False,
        case_id: CaseType | None = None,
        custom_case: dict | None = None,
        cases: list[CaseConfig] | None = None,
        label: str = "",
        description: str = "",
        caseLabel: CaseLabel = CaseLabel.Performance,
    ):
        if isLine is True:
            super().__init__(isLine=True)
        elif case_id is not None and isinstance(case_id, CaseType):
            c = case_id.case_cls(custom_case)
            super().__init__(
                label=c.name,
                description=c.description,
                cases=[CaseConfig(case_id=case_id, custom_case=custom_case)],
                caseLabel=c.label,
            )
        else:
            super().__init__(
                label=label,
                description=description,
                cases=cases,
                caseLabel=caseLabel,
            )

    def __hash__(self) -> int:
        return hash(self.json())


class UICaseItemCluster(BaseModel):
    label: str = ""
    uiCaseItems: list[UICaseItem] = []


def get_custom_case_items() -> list[UICaseItem]:
    custom_configs = get_custom_configs()
    return [
        UICaseItem(case_id=CaseType.PerformanceCustomDataset, custom_case=custom_config.dict())
        for custom_config in custom_configs
    ]


def get_custom_case_cluter() -> UICaseItemCluster:
    return UICaseItemCluster(label="Custom Search Performance Test", uiCaseItems=get_custom_case_items())


UI_CASE_CLUSTERS: list[UICaseItemCluster] = [
    UICaseItemCluster(
        label="Search Performance Test",
        uiCaseItems=[
            UICaseItem(case_id=CaseType.Performance768D100M),
            UICaseItem(case_id=CaseType.Performance768D10M),
            UICaseItem(case_id=CaseType.Performance768D1M),
            UICaseItem(isLine=True),
            UICaseItem(case_id=CaseType.Performance1536D5M),
            UICaseItem(case_id=CaseType.Performance1536D500K),
            UICaseItem(case_id=CaseType.Performance1536D50K),
        ],
    ),
    UICaseItemCluster(
        label="Filter Search Performance Test",
        uiCaseItems=[
            UICaseItem(case_id=CaseType.Performance768D10M1P),
            UICaseItem(case_id=CaseType.Performance768D10M99P),
            UICaseItem(case_id=CaseType.Performance768D1M1P),
            UICaseItem(case_id=CaseType.Performance768D1M99P),
            UICaseItem(isLine=True),
            UICaseItem(case_id=CaseType.Performance1536D5M1P),
            UICaseItem(case_id=CaseType.Performance1536D5M99P),
            UICaseItem(case_id=CaseType.Performance1536D500K1P),
            UICaseItem(case_id=CaseType.Performance1536D500K99P),
        ],
    ),
    UICaseItemCluster(
        label="Capacity Test",
        uiCaseItems=[
            UICaseItem(case_id=CaseType.CapacityDim960),
            UICaseItem(case_id=CaseType.CapacityDim128),
        ],
    ),
]

# DIVIDER = "DIVIDER"
DISPLAY_CASE_ORDER: list[CaseType] = [
    CaseType.Performance768D100M,
    CaseType.Performance768D10M,
    CaseType.Performance768D1M,
    CaseType.Performance1536D5M,
    CaseType.Performance1536D500K,
    CaseType.Performance1536D50K,
    CaseType.Performance768D10M1P,
    CaseType.Performance768D1M1P,
    CaseType.Performance1536D5M1P,
    CaseType.Performance1536D500K1P,
    CaseType.Performance768D10M99P,
    CaseType.Performance768D1M99P,
    CaseType.Performance1536D5M99P,
    CaseType.Performance1536D500K99P,
    CaseType.CapacityDim960,
    CaseType.CapacityDim128,
]
CASE_NAME_ORDER = [case.case_cls().name for case in DISPLAY_CASE_ORDER]

# CASE_LIST = [
#     item for item in CASE_LIST_WITH_DIVIDER if isinstance(item, CaseType)]


class InputType(IntEnum):
    Text = 20001
    Number = 20002
    Option = 20003
    Float = 20004
    Bool = 20005


class CaseConfigInput(BaseModel):
    label: CaseConfigParamType
    inputType: InputType = InputType.Text
    inputConfig: dict = {}
    inputHelp: str = ""
    displayLabel: str = ""
    # todo type should be a function
    isDisplayed: typing.Any = lambda config: True


CaseConfigParamInput_IndexType = CaseConfigInput(
    label=CaseConfigParamType.IndexType,
    inputType=InputType.Option,
    inputConfig={
        "options": [
            IndexType.HNSW.value,
            IndexType.HNSW_SQ.value,
            IndexType.HNSW_PQ.value,
            IndexType.HNSW_PRQ.value,
            IndexType.IVFFlat.value,
            IndexType.IVFPQ.value,
            IndexType.IVFSQ8.value,
            IndexType.IVF_RABITQ.value,
            IndexType.DISKANN.value,
            IndexType.Flat.value,
            IndexType.AUTOINDEX.value,
            IndexType.GPU_IVF_FLAT.value,
            IndexType.GPU_IVF_PQ.value,
            IndexType.GPU_CAGRA.value,
            IndexType.GPU_BRUTE_FORCE.value,
        ],
    },
)

CaseConfigParamInput_IndexType_PgDiskANN = CaseConfigInput(
    label=CaseConfigParamType.IndexType,
    inputHelp="Select Index Type",
    inputType=InputType.Option,
    inputConfig={
        "options": [
            IndexType.DISKANN.value,
        ],
    },
)

CaseConfigParamInput_IndexType_PgVectorScale = CaseConfigInput(
    label=CaseConfigParamType.IndexType,
    inputHelp="Select Index Type",
    inputType=InputType.Option,
    inputConfig={
        "options": [
            IndexType.STREAMING_DISKANN.value,
        ],
    },
)


CaseConfigParamInput_storage_layout = CaseConfigInput(
    label=CaseConfigParamType.storage_layout,
    inputHelp="Select Storage Layout",
    inputType=InputType.Option,
    inputConfig={
        "options": [
            "memory_optimized",
            "plain",
        ],
    },
)

CaseConfigParamInput_max_neighbors = CaseConfigInput(
    label=CaseConfigParamType.max_neighbors,
    inputType=InputType.Number,
    inputConfig={
        "min": 10,
        "max": 300,
        "value": 32,
    },
    isDisplayed=lambda config: config.get(CaseConfigParamType.IndexType, None) == IndexType.DISKANN.value,
)

CaseConfigParamInput_l_value_ib = CaseConfigInput(
    label=CaseConfigParamType.l_value_ib,
    inputType=InputType.Number,
    inputConfig={
        "min": 10,
        "max": 300,
        "value": 50,
    },
    isDisplayed=lambda config: config.get(CaseConfigParamType.IndexType, None) == IndexType.DISKANN.value,
)

CaseConfigParamInput_l_value_is = CaseConfigInput(
    label=CaseConfigParamType.l_value_is,
    inputType=InputType.Number,
    inputConfig={
        "min": 10,
        "max": 300,
        "value": 40,
    },
    isDisplayed=lambda config: config.get(CaseConfigParamType.IndexType, None) == IndexType.DISKANN.value,
)

CaseConfigParamInput_num_neighbors = CaseConfigInput(
    label=CaseConfigParamType.num_neighbors,
    inputType=InputType.Number,
    inputConfig={
        "min": 10,
        "max": 300,
        "value": 50,
    },
    isDisplayed=lambda config: config.get(CaseConfigParamType.IndexType, None) == IndexType.STREAMING_DISKANN.value,
)

CaseConfigParamInput_search_list_size = CaseConfigInput(
    label=CaseConfigParamType.search_list_size,
    inputType=InputType.Number,
    inputConfig={
        "min": 10,
        "max": 300,
        "value": 100,
    },
    isDisplayed=lambda config: config.get(CaseConfigParamType.IndexType, None) == IndexType.STREAMING_DISKANN.value,
)

CaseConfigParamInput_max_alpha = CaseConfigInput(
    label=CaseConfigParamType.max_alpha,
    inputType=InputType.Float,
    inputConfig={
        "min": 0.1,
        "max": 2.0,
        "value": 1.2,
    },
    isDisplayed=lambda config: config.get(CaseConfigParamType.IndexType, None) == IndexType.STREAMING_DISKANN.value,
)

CaseConfigParamInput_num_dimensions = CaseConfigInput(
    label=CaseConfigParamType.num_dimensions,
    inputType=InputType.Number,
    inputConfig={
        "min": 0,
        "max": 2000,
        "value": 0,
    },
    isDisplayed=lambda config: config.get(CaseConfigParamType.IndexType, None) == IndexType.STREAMING_DISKANN.value,
)

CaseConfigParamInput_query_search_list_size = CaseConfigInput(
    label=CaseConfigParamType.query_search_list_size,
    inputType=InputType.Number,
    inputConfig={
        "min": 50,
        "max": 150,
        "value": 100,
    },
    isDisplayed=lambda config: config.get(CaseConfigParamType.IndexType, None) == IndexType.STREAMING_DISKANN.value,
)


CaseConfigParamInput_query_rescore = CaseConfigInput(
    label=CaseConfigParamType.query_rescore,
    inputType=InputType.Number,
    inputConfig={
        "min": 0,
        "max": 150,
        "value": 50,
    },
    isDisplayed=lambda config: config.get(CaseConfigParamType.IndexType, None) == IndexType.STREAMING_DISKANN.value,
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

CaseConfigParamInput_IndexType_PgVectoRS = CaseConfigInput(
    label=CaseConfigParamType.IndexType,
    inputHelp="Select Index Type",
    inputType=InputType.Option,
    inputConfig={
        "options": [
            IndexType.HNSW.value,
            IndexType.IVFFlat.value,
            IndexType.Flat.value,
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
    in [
        IndexType.HNSW.value,
        IndexType.HNSW_SQ.value,
        IndexType.HNSW_PQ.value,
        IndexType.HNSW_PRQ.value,
    ],
)


CaseConfigParamInput_m = CaseConfigInput(
    label=CaseConfigParamType.m,
    inputType=InputType.Number,
    inputConfig={
        "min": 4,
        "max": 64,
        "value": 16,
    },
    isDisplayed=lambda config: config.get(CaseConfigParamType.IndexType, None) == IndexType.HNSW.value,
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
    in [
        IndexType.HNSW.value,
        IndexType.HNSW_SQ.value,
        IndexType.HNSW_PQ.value,
        IndexType.HNSW_PRQ.value,
    ],
)

CaseConfigParamInput_SQType = CaseConfigInput(
    label=CaseConfigParamType.sq_type,
    inputType=InputType.Option,
    inputHelp="Scalar quantizer type.",
    inputConfig={
        "options": [SQType.SQ6.value, SQType.SQ8.value, SQType.BF16.value, SQType.FP16.value, SQType.FP32.value]
    },
    isDisplayed=lambda config: config.get(CaseConfigParamType.IndexType, None) in [IndexType.HNSW_SQ.value],
)

CaseConfigParamInput_Refine = CaseConfigInput(
    label=CaseConfigParamType.refine,
    inputType=InputType.Option,
    inputHelp="Whether refined data is reserved during index building.",
    inputConfig={"options": [True, False]},
    isDisplayed=lambda config: config.get(CaseConfigParamType.IndexType, None)
    in [IndexType.HNSW_SQ.value, IndexType.HNSW_PQ.value, IndexType.HNSW_PRQ.value, IndexType.IVF_RABITQ.value],
)

CaseConfigParamInput_RefineType = CaseConfigInput(
    label=CaseConfigParamType.refine_type,
    inputType=InputType.Option,
    inputHelp="The data type of the refine index.",
    inputConfig={
        "options": [SQType.FP32.value, SQType.FP16.value, SQType.BF16.value, SQType.SQ8.value, SQType.SQ6.value]
    },
    isDisplayed=lambda config: config.get(CaseConfigParamType.IndexType, None)
    in [IndexType.HNSW_SQ.value, IndexType.HNSW_PQ.value, IndexType.HNSW_PRQ.value, IndexType.IVF_RABITQ.value]
    and config.get(CaseConfigParamType.refine, True),
)

CaseConfigParamInput_RefineK = CaseConfigInput(
    label=CaseConfigParamType.refine_k,
    inputType=InputType.Float,
    inputHelp="The magnification factor of refine compared to k.",
    inputConfig={"min": 1.0, "max": 10000.0, "value": 1.0},
    isDisplayed=lambda config: config.get(CaseConfigParamType.IndexType, None)
    in [IndexType.HNSW_SQ.value, IndexType.HNSW_PQ.value, IndexType.HNSW_PRQ.value, IndexType.IVF_RABITQ.value]
    and config.get(CaseConfigParamType.refine, True),
)

CaseConfigParamInput_RBQBitsQuery = CaseConfigInput(
    label=CaseConfigParamType.rbq_bits_query,
    inputType=InputType.Number,
    inputHelp="The magnification factor of refine compared to k.",
    inputConfig={"min": 0, "max": 8, "value": 0},
    isDisplayed=lambda config: config.get(CaseConfigParamType.IndexType, None) in [IndexType.IVF_RABITQ.value],
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

CaseConfigParamInput_EFConstruction_AWSOpensearch = CaseConfigInput(
    label=CaseConfigParamType.EFConstruction,
    inputType=InputType.Number,
    inputConfig={
        "min": 100,
        "max": 1024,
        "value": 256,
    },
)

CaseConfigParamInput_M_AWSOpensearch = CaseConfigInput(
    label=CaseConfigParamType.M,
    inputType=InputType.Number,
    inputConfig={
        "min": 4,
        "max": 64,
        "value": 16,
    },
)

CaseConfigParamInput_EF_SEARCH_AWSOpensearch = CaseConfigInput(
    label=CaseConfigParamType.ef_search,
    inputType=InputType.Number,
    inputConfig={
        "min": 100,
        "max": 1024,
        "value": 256,
    },
)

CaseConfigParamInput_EF_SEARCH_AliyunOpensearch = CaseConfigInput(
    label=CaseConfigParamType.ef_search,
    inputType=InputType.Number,
    inputConfig={
        "min": 1,
        "max": 1000000,
        "value": 40,
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
    label=CaseConfigParamType.ef_construction,
    inputType=InputType.Number,
    inputConfig={
        "min": 10,
        "max": 2000,
        "value": 300,
    },
    isDisplayed=lambda config: config.get(CaseConfigParamType.IndexType, None) == IndexType.HNSW.value,
)

CaseConfigParamInput_EFSearch_PgVectoRS = CaseConfigInput(
    label=CaseConfigParamType.ef_search,
    inputType=InputType.Number,
    inputConfig={
        "min": 1,
        "max": 65535,
        "value": 100,
    },
    isDisplayed=lambda config: config.get(CaseConfigParamType.IndexType, None) == IndexType.HNSW.value,
)

CaseConfigParamInput_EFConstruction_PgVector = CaseConfigInput(
    label=CaseConfigParamType.ef_construction,
    inputType=InputType.Number,
    inputConfig={
        "min": 8,
        "max": 1024,
        "value": 256,
    },
    isDisplayed=lambda config: config[CaseConfigParamType.IndexType] == IndexType.HNSW.value,
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
    in [
        IndexType.HNSW.value,
        IndexType.HNSW_SQ.value,
        IndexType.HNSW_PQ.value,
        IndexType.HNSW_PRQ.value,
    ],
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
    isDisplayed=lambda config: config.get(CaseConfigParamType.IndexType, None) == IndexType.DISKANN.value,
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
        IndexType.IVFPQ.value,
        IndexType.IVFSQ8.value,
        IndexType.IVF_RABITQ.value,
        IndexType.GPU_IVF_FLAT.value,
        IndexType.GPU_IVF_PQ.value,
        IndexType.GPU_BRUTE_FORCE.value,
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
        IndexType.IVFPQ.value,
        IndexType.IVFSQ8.value,
        IndexType.IVF_RABITQ.value,
        IndexType.GPU_IVF_FLAT.value,
        IndexType.GPU_IVF_PQ.value,
        IndexType.GPU_BRUTE_FORCE.value,
    ],
)

CaseConfigParamInput_M_PQ = CaseConfigInput(
    label=CaseConfigParamType.m,
    inputType=InputType.Number,
    inputConfig={
        "min": 1,
        "max": 65536,
        "value": 32,
    },
    isDisplayed=lambda config: config.get(CaseConfigParamType.IndexType, None)
    in [IndexType.GPU_IVF_PQ.value, IndexType.HNSW_PQ.value, IndexType.HNSW_PRQ.value, IndexType.IVFPQ.value],
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
    in [IndexType.GPU_IVF_PQ.value, IndexType.HNSW_PQ.value, IndexType.HNSW_PRQ.value, IndexType.IVFPQ.value],
)

CaseConfigParamInput_NRQ = CaseConfigInput(
    label=CaseConfigParamType.nrq,
    inputType=InputType.Number,
    inputHelp="The number of residual subquantizers.",
    inputConfig={
        "min": 1,
        "max": 16,
        "value": 2,
    },
    isDisplayed=lambda config: config.get(CaseConfigParamType.IndexType, None) in [IndexType.HNSW_PRQ.value],
)

CaseConfigParamInput_intermediate_graph_degree = CaseConfigInput(
    label=CaseConfigParamType.intermediate_graph_degree,
    inputType=InputType.Number,
    inputConfig={
        "min": 1,
        "max": 65536,
        "value": 64,
    },
    isDisplayed=lambda config: config.get(CaseConfigParamType.IndexType, None) in [IndexType.GPU_CAGRA.value],
)

CaseConfigParamInput_graph_degree = CaseConfigInput(
    label=CaseConfigParamType.graph_degree,
    inputType=InputType.Number,
    inputConfig={
        "min": 1,
        "max": 65536,
        "value": 32,
    },
    isDisplayed=lambda config: config.get(CaseConfigParamType.IndexType, None) in [IndexType.GPU_CAGRA.value],
)

CaseConfigParamInput_itopk_size = CaseConfigInput(
    label=CaseConfigParamType.itopk_size,
    inputType=InputType.Number,
    inputConfig={
        "min": 1,
        "max": 65536,
        "value": 128,
    },
    isDisplayed=lambda config: config.get(CaseConfigParamType.IndexType, None) in [IndexType.GPU_CAGRA.value],
)

CaseConfigParamInput_team_size = CaseConfigInput(
    label=CaseConfigParamType.team_size,
    inputType=InputType.Number,
    inputConfig={
        "min": 0,
        "max": 65536,
        "value": 0,
    },
    isDisplayed=lambda config: config.get(CaseConfigParamType.IndexType, None) in [IndexType.GPU_CAGRA.value],
)

CaseConfigParamInput_search_width = CaseConfigInput(
    label=CaseConfigParamType.search_width,
    inputType=InputType.Number,
    inputConfig={
        "min": 0,
        "max": 65536,
        "value": 4,
    },
    isDisplayed=lambda config: config.get(CaseConfigParamType.IndexType, None) in [IndexType.GPU_CAGRA.value],
)

CaseConfigParamInput_min_iterations = CaseConfigInput(
    label=CaseConfigParamType.min_iterations,
    inputType=InputType.Number,
    inputConfig={
        "min": 0,
        "max": 65536,
        "value": 0,
    },
    isDisplayed=lambda config: config.get(CaseConfigParamType.IndexType, None) in [IndexType.GPU_CAGRA.value],
)

CaseConfigParamInput_max_iterations = CaseConfigInput(
    label=CaseConfigParamType.max_iterations,
    inputType=InputType.Number,
    inputConfig={
        "min": 0,
        "max": 65536,
        "value": 0,
    },
    isDisplayed=lambda config: config.get(CaseConfigParamType.IndexType, None) in [IndexType.GPU_CAGRA.value],
)

CaseConfigParamInput_build_algo = CaseConfigInput(
    label=CaseConfigParamType.build_algo,
    inputType=InputType.Option,
    inputConfig={
        "options": ["IVF_PQ", "NN_DESCENT"],
    },
    isDisplayed=lambda config: config.get(CaseConfigParamType.IndexType, None) in [IndexType.GPU_CAGRA.value],
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
        IndexType.GPU_BRUTE_FORCE.value,
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
        IndexType.GPU_BRUTE_FORCE.value,
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
    isDisplayed=lambda config: config.get(CaseConfigParamType.IndexType, None) in [IndexType.IVFFlat.value],
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
    isDisplayed=lambda config: config.get(CaseConfigParamType.IndexType, None) == IndexType.IVFFlat.value,
)

CaseConfigParamInput_Probes_PgVector = CaseConfigInput(
    label=CaseConfigParamType.probes,
    inputType=InputType.Number,
    inputConfig={
        "min": 1,
        "max": 65536,
        "value": 1,
    },
    isDisplayed=lambda config: config.get(CaseConfigParamType.IndexType, None) == IndexType.IVFFlat.value,
)

CaseConfigParamInput_EFSearch_PgVector = CaseConfigInput(
    label=CaseConfigParamType.ef_search,
    inputType=InputType.Number,
    inputConfig={
        "min": 1,
        "max": 2048,
        "value": 256,
    },
    isDisplayed=lambda config: config.get(CaseConfigParamType.IndexType, None) == IndexType.HNSW.value,
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

CaseConfigParamInput_QuantizationType_PgVector = CaseConfigInput(
    label=CaseConfigParamType.quantizationType,
    inputType=InputType.Option,
    inputConfig={
        "options": ["none", "bit", "halfvec"],
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
    isDisplayed=lambda config: config.get(CaseConfigParamType.quantizationType, None) == "product"
    and config.get(CaseConfigParamType.IndexType, None)
    in [
        IndexType.HNSW.value,
        IndexType.IVFFlat.value,
    ],
)

CaseConfigParamInput_TableQuantizationType_PgVector = CaseConfigInput(
    label=CaseConfigParamType.tableQuantizationType,
    inputType=InputType.Option,
    inputConfig={
        "options": ["none", "bit", "halfvec"],
    },
    isDisplayed=lambda config: config.get(CaseConfigParamType.IndexType, None)
    in [
        IndexType.HNSW.value,
        IndexType.IVFFlat.value,
    ],
)

CaseConfigParamInput_max_parallel_workers_PgVectorRS = CaseConfigInput(
    label=CaseConfigParamType.max_parallel_workers,
    displayLabel="Max parallel workers",
    inputHelp="Recommended value: (cpu cores - 1). This will set the parameters: [optimizing.optimizing_threads]",
    inputType=InputType.Number,
    inputConfig={
        "min": 0,
        "max": 1024,
        "value": 16,
    },
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

CaseConfigParamInput_reranking_PgVector = CaseConfigInput(
    label=CaseConfigParamType.reranking,
    inputType=InputType.Bool,
    displayLabel="Enable Reranking",
    inputHelp="Enable if you want to use reranking while performing \
        similarity search in binary quantization",
    inputConfig={
        "value": False,
    },
    isDisplayed=lambda config: config.get(CaseConfigParamType.quantizationType, None) == "bit",
)

CaseConfigParamInput_quantized_fetch_limit_PgVector = CaseConfigInput(
    label=CaseConfigParamType.quantizedFetchLimit,
    displayLabel="Quantized vector fetch limit",
    inputHelp="Limit top-k vectors using the quantized vector comparison --bound by ef_search",
    inputType=InputType.Number,
    inputConfig={
        "min": 20,
        "max": 1000,
        "value": 200,
    },
    isDisplayed=lambda config: config.get(CaseConfigParamType.quantizationType, None) == "bit"
    and config.get(CaseConfigParamType.reranking, False),
)


CaseConfigParamInput_reranking_metric_PgVector = CaseConfigInput(
    label=CaseConfigParamType.rerankingMetric,
    inputType=InputType.Option,
    inputConfig={
        "options": [metric.value for metric in MetricType if metric.value not in ["HAMMING", "JACCARD"]],
    },
    isDisplayed=lambda config: config.get(CaseConfigParamType.quantizationType, None) == "bit"
    and config.get(CaseConfigParamType.reranking, False),
)


CaseConfigParamInput_IndexType_AlloyDB = CaseConfigInput(
    label=CaseConfigParamType.IndexType,
    inputHelp="Select Index Type",
    inputType=InputType.Option,
    inputConfig={
        "options": [
            IndexType.SCANN.value,
        ],
    },
)

CaseConfigParamInput_num_leaves_AlloyDB = CaseConfigInput(
    label=CaseConfigParamType.numLeaves,
    displayLabel="Num Leaves",
    inputHelp="The number of partition to apply to this index",
    inputType=InputType.Number,
    inputConfig={
        "min": 1,
        "max": 1048576,
        "value": 200,
    },
)

CaseConfigParamInput_quantizer_AlloyDB = CaseConfigInput(
    label=CaseConfigParamType.quantizer,
    inputType=InputType.Option,
    inputConfig={
        "options": ["SQ8", "Flat"],
    },
)

CaseConfigParamInput_max_num_levels_AlloyDB = CaseConfigInput(
    label=CaseConfigParamType.maxNumLevels,
    inputType=InputType.Option,
    inputConfig={
        "options": [1, 2],
    },
)

CaseConfigParamInput_enable_pca_AlloyDB = CaseConfigInput(
    label=CaseConfigParamType.enablePca,
    inputType=InputType.Option,
    inputConfig={
        "options": ["on", "off"],
    },
)

CaseConfigParamInput_num_leaves_to_search_AlloyDB = CaseConfigInput(
    label=CaseConfigParamType.numLeavesToSearch,
    displayLabel="Num leaves to search",
    inputHelp="The database flag controls the trade off between recall and QPS",
    inputType=InputType.Number,
    inputConfig={
        "min": 20,
        "max": 10486,
        "value": 20,
    },
)

CaseConfigParamInput_max_top_neighbors_buffer_size_AlloyDB = CaseConfigInput(
    label=CaseConfigParamType.maxTopNeighborsBufferSize,
    displayLabel="Max top neighbors buffer size",
    inputHelp="The database flag specifies the size of cache used to improve the \
        performance for filtered queries by scoring or ranking the scanned candidate \
        neighbors in memory instead of the disk",
    inputType=InputType.Number,
    inputConfig={
        "min": 10000,
        "max": 60000,
        "value": 20000,
    },
)

CaseConfigParamInput_pre_reordering_num_neighbors_AlloyDB = CaseConfigInput(
    label=CaseConfigParamType.preReorderingNumNeigbors,
    displayLabel="Pre reordering num neighbors",
    inputHelp="Specifies the number of candidate neighbors to consider during the reordering \
        stages after initial search identifies a set of candidates",
    inputType=InputType.Number,
    inputConfig={
        "min": 20,
        "max": 10486,
        "value": 80,
    },
)

CaseConfigParamInput_num_search_threads_AlloyDB = CaseConfigInput(
    label=CaseConfigParamType.numSearchThreads,
    displayLabel="Num of searcher threads",
    inputHelp="The number of searcher threads for multi-thread search.",
    inputType=InputType.Number,
    inputConfig={
        "min": 1,
        "max": 100,
        "value": 2,
    },
)

CaseConfigParamInput_max_num_prefetch_datasets_AlloyDB = CaseConfigInput(
    label=CaseConfigParamType.maxNumPrefetchDatasets,
    displayLabel="Max num prefetch datasets",
    inputHelp="The maximum number of data batches to prefetch during index search, where batch is a group of buffer pages",
    inputType=InputType.Number,
    inputConfig={
        "min": 10,
        "max": 150,
        "value": 100,
    },
)

CaseConfigParamInput_maintenance_work_mem_AlloyDB = CaseConfigInput(
    label=CaseConfigParamType.maintenance_work_mem,
    inputHelp="Recommended value: 1.33x the index size, not to exceed the available free memory."
    "Specify in gigabytes. e.g. 8GB",
    inputType=InputType.Text,
    inputConfig={
        "value": "8GB",
    },
)

CaseConfigParamInput_max_parallel_workers_AlloyDB = CaseConfigInput(
    label=CaseConfigParamType.max_parallel_workers,
    displayLabel="Max parallel workers",
    inputHelp="Recommended value: (cpu cores - 1). This will set the parameters: max_parallel_maintenance_workers,"
    " max_parallel_workers & table(parallel_workers)",
    inputType=InputType.Number,
    inputConfig={
        "min": 0,
        "max": 1024,
        "value": 7,
    },
)

CaseConfigParamInput_EFConstruction_AliES = CaseConfigInput(
    label=CaseConfigParamType.EFConstruction,
    inputType=InputType.Number,
    inputConfig={
        "min": 8,
        "max": 512,
        "value": 360,
    },
)

CaseConfigParamInput_M_AliES = CaseConfigInput(
    label=CaseConfigParamType.M,
    inputType=InputType.Number,
    inputConfig={
        "min": 4,
        "max": 64,
        "value": 30,
    },
)
CaseConfigParamInput_NumCandidates_AliES = CaseConfigInput(
    label=CaseConfigParamType.numCandidates,
    inputType=InputType.Number,
    inputConfig={
        "min": 1,
        "max": 10000,
        "value": 100,
    },
)

CaseConfigParamInput_IndexType_MariaDB = CaseConfigInput(
    label=CaseConfigParamType.IndexType,
    inputHelp="Select Index Type",
    inputType=InputType.Option,
    inputConfig={
        "options": [
            IndexType.HNSW.value,
        ],
    },
)

CaseConfigParamInput_StorageEngine_MariaDB = CaseConfigInput(
    label=CaseConfigParamType.storage_engine,
    inputHelp="Select Storage Engine",
    inputType=InputType.Option,
    inputConfig={
        "options": ["InnoDB", "MyISAM"],
    },
)

CaseConfigParamInput_M_MariaDB = CaseConfigInput(
    label=CaseConfigParamType.M,
    inputHelp="M parameter in MHNSW vector indexing",
    inputType=InputType.Number,
    inputConfig={
        "min": 3,
        "max": 200,
        "value": 6,
    },
    isDisplayed=lambda config: config.get(CaseConfigParamType.IndexType, None) == IndexType.HNSW.value,
)

CaseConfigParamInput_EFSearch_MariaDB = CaseConfigInput(
    label=CaseConfigParamType.ef_search,
    inputHelp="mhnsw_ef_search",
    inputType=InputType.Number,
    inputConfig={
        "min": 1,
        "max": 10000,
        "value": 20,
    },
    isDisplayed=lambda config: config.get(CaseConfigParamType.IndexType, None) == IndexType.HNSW.value,
)

CaseConfigParamInput_CacheSize_MariaDB = CaseConfigInput(
    label=CaseConfigParamType.max_cache_size,
    inputHelp="mhnsw_max_cache_size",
    inputType=InputType.Number,
    inputConfig={
        "min": 1048576,
        "max": (1 << 53) - 1,
        "value": 16 * 1024**3,
    },
    isDisplayed=lambda config: config.get(CaseConfigParamType.IndexType, None) == IndexType.HNSW.value,
)

CaseConfigParamInput_MongoDBQuantizationType = CaseConfigInput(
    label=CaseConfigParamType.mongodb_quantization_type,
    inputType=InputType.Option,
    inputConfig={
        "options": ["none", "scalar", "binary"],
    },
)


CaseConfigParamInput_MongoDBNumCandidatesRatio = CaseConfigInput(
    label=CaseConfigParamType.mongodb_num_candidates_ratio,
    inputType=InputType.Number,
    inputConfig={
        "min": 10,
        "max": 20,
        "value": 10,
    },
)


CaseConfigParamInput_M_Vespa = CaseConfigInput(
    label=CaseConfigParamType.M,
    inputType=InputType.Number,
    inputConfig={
        "min": 4,
        "max": 64,
        "value": 16,
    },
    isDisplayed=lambda config: config.get(CaseConfigParamType.IndexType, None) == IndexType.HNSW.value,
)

CaseConfigParamInput_IndexType_Vespa = CaseConfigInput(
    label=CaseConfigParamType.IndexType,
    inputType=InputType.Option,
    inputConfig={
        "options": [
            IndexType.HNSW.value,
        ],
    },
)

CaseConfigParamInput_QuantizationType_Vespa = CaseConfigInput(
    label=CaseConfigParamType.quantizationType,
    inputType=InputType.Option,
    inputConfig={
        "options": ["none", "binary"],
    },
)

CaseConfigParamInput_EFConstruction_Vespa = CaseConfigInput(
    label=CaseConfigParamType.EFConstruction,
    inputType=InputType.Number,
    inputConfig={
        "min": 8,
        "max": 512,
        "value": 200,
    },
    isDisplayed=lambda config: config[CaseConfigParamType.IndexType] == IndexType.HNSW.value,
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
    CaseConfigParamInput_SQType,
    CaseConfigParamInput_Refine,
    CaseConfigParamInput_RefineType,
    CaseConfigParamInput_NRQ,
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
    CaseConfigParamInput_RBQBitsQuery,
    CaseConfigParamInput_NRQ,
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
    CaseConfigParamInput_SQType,
    CaseConfigParamInput_Refine,
    CaseConfigParamInput_RefineType,
    CaseConfigParamInput_RefineK,
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

AWSOpensearchLoadingConfig = [
    CaseConfigParamInput_EFConstruction_AWSOpensearch,
    CaseConfigParamInput_M_AWSOpensearch,
]
AWSOpenSearchPerformanceConfig = [
    CaseConfigParamInput_EFConstruction_AWSOpensearch,
    CaseConfigParamInput_M_AWSOpensearch,
    CaseConfigParamInput_EF_SEARCH_AWSOpensearch,
]

AliyunOpensearchLoadingConfig = []
AliyunOpenSearchPerformanceConfig = [
    CaseConfigParamInput_EF_SEARCH_AliyunOpensearch,
]

PgVectorLoadingConfig = [
    CaseConfigParamInput_IndexType_PgVector,
    CaseConfigParamInput_Lists_PgVector,
    CaseConfigParamInput_m,
    CaseConfigParamInput_EFConstruction_PgVector,
    CaseConfigParamInput_QuantizationType_PgVector,
    CaseConfigParamInput_TableQuantizationType_PgVector,
    CaseConfigParamInput_maintenance_work_mem_PgVector,
    CaseConfigParamInput_max_parallel_workers_PgVector,
]
PgVectorPerformanceConfig = [
    CaseConfigParamInput_IndexType_PgVector,
    CaseConfigParamInput_m,
    CaseConfigParamInput_EFConstruction_PgVector,
    CaseConfigParamInput_EFSearch_PgVector,
    CaseConfigParamInput_Lists_PgVector,
    CaseConfigParamInput_Probes_PgVector,
    CaseConfigParamInput_QuantizationType_PgVector,
    CaseConfigParamInput_TableQuantizationType_PgVector,
    CaseConfigParamInput_maintenance_work_mem_PgVector,
    CaseConfigParamInput_max_parallel_workers_PgVector,
    CaseConfigParamInput_reranking_PgVector,
    CaseConfigParamInput_reranking_metric_PgVector,
    CaseConfigParamInput_quantized_fetch_limit_PgVector,
]

PgVectoRSLoadingConfig = [
    CaseConfigParamInput_IndexType_PgVectoRS,
    CaseConfigParamInput_m,
    CaseConfigParamInput_EFConstruction_PgVectoRS,
    CaseConfigParamInput_Nlist,
    CaseConfigParamInput_QuantizationType_PgVectoRS,
    CaseConfigParamInput_QuantizationRatio_PgVectoRS,
    CaseConfigParamInput_max_parallel_workers_PgVectorRS,
]

PgVectoRSPerformanceConfig = [
    CaseConfigParamInput_IndexType_PgVectoRS,
    CaseConfigParamInput_m,
    CaseConfigParamInput_EFConstruction_PgVectoRS,
    CaseConfigParamInput_EFSearch_PgVectoRS,
    CaseConfigParamInput_Nlist,
    CaseConfigParamInput_Nprobe,
    CaseConfigParamInput_QuantizationType_PgVectoRS,
    CaseConfigParamInput_QuantizationRatio_PgVectoRS,
    CaseConfigParamInput_max_parallel_workers_PgVectorRS,
]

ZillizCloudPerformanceConfig = [
    CaseConfigParamInput_ZillizLevel,
]

PgVectorScaleLoadingConfig = [
    CaseConfigParamInput_IndexType_PgVectorScale,
    CaseConfigParamInput_num_neighbors,
    CaseConfigParamInput_storage_layout,
    CaseConfigParamInput_search_list_size,
    CaseConfigParamInput_max_alpha,
]

PgVectorScalePerformanceConfig = [
    CaseConfigParamInput_IndexType_PgVectorScale,
    CaseConfigParamInput_num_neighbors,
    CaseConfigParamInput_storage_layout,
    CaseConfigParamInput_search_list_size,
    CaseConfigParamInput_max_alpha,
    CaseConfigParamInput_query_rescore,
    CaseConfigParamInput_query_search_list_size,
]

PgDiskANNLoadConfig = [
    CaseConfigParamInput_IndexType_PgDiskANN,
    CaseConfigParamInput_max_neighbors,
    CaseConfigParamInput_l_value_ib,
]

PgDiskANNPerformanceConfig = [
    CaseConfigParamInput_IndexType_PgDiskANN,
    CaseConfigParamInput_max_neighbors,
    CaseConfigParamInput_l_value_ib,
    CaseConfigParamInput_l_value_is,
]


AlloyDBLoadConfig = [
    CaseConfigParamInput_IndexType_AlloyDB,
    CaseConfigParamInput_num_leaves_AlloyDB,
    CaseConfigParamInput_max_num_levels_AlloyDB,
    CaseConfigParamInput_enable_pca_AlloyDB,
    CaseConfigParamInput_quantizer_AlloyDB,
    CaseConfigParamInput_maintenance_work_mem_AlloyDB,
    CaseConfigParamInput_max_parallel_workers_AlloyDB,
]

AlloyDBPerformanceConfig = [
    CaseConfigParamInput_IndexType_AlloyDB,
    CaseConfigParamInput_num_leaves_AlloyDB,
    CaseConfigParamInput_max_num_levels_AlloyDB,
    CaseConfigParamInput_enable_pca_AlloyDB,
    CaseConfigParamInput_quantizer_AlloyDB,
    CaseConfigParamInput_num_search_threads_AlloyDB,
    CaseConfigParamInput_num_leaves_to_search_AlloyDB,
    CaseConfigParamInput_max_num_prefetch_datasets_AlloyDB,
    CaseConfigParamInput_max_top_neighbors_buffer_size_AlloyDB,
    CaseConfigParamInput_pre_reordering_num_neighbors_AlloyDB,
    CaseConfigParamInput_maintenance_work_mem_AlloyDB,
    CaseConfigParamInput_max_parallel_workers_AlloyDB,
]

AliyunElasticsearchLoadingConfig = [
    CaseConfigParamInput_EFConstruction_AliES,
    CaseConfigParamInput_M_AliES,
]
AliyunElasticsearchPerformanceConfig = [
    CaseConfigParamInput_EFConstruction_AliES,
    CaseConfigParamInput_M_AliES,
    CaseConfigParamInput_NumCandidates_AliES,
]

MongoDBLoadingConfig = [
    CaseConfigParamInput_MongoDBQuantizationType,
]
MongoDBPerformanceConfig = [
    CaseConfigParamInput_MongoDBQuantizationType,
    CaseConfigParamInput_MongoDBNumCandidatesRatio,
]

MariaDBLoadingConfig = [
    CaseConfigParamInput_IndexType_MariaDB,
    CaseConfigParamInput_StorageEngine_MariaDB,
    CaseConfigParamInput_M_MariaDB,
    CaseConfigParamInput_CacheSize_MariaDB,
]
MariaDBPerformanceConfig = [
    CaseConfigParamInput_IndexType_MariaDB,
    CaseConfigParamInput_StorageEngine_MariaDB,
    CaseConfigParamInput_M_MariaDB,
    CaseConfigParamInput_CacheSize_MariaDB,
    CaseConfigParamInput_EFSearch_MariaDB,
]

VespaLoadingConfig = [
    CaseConfigParamInput_IndexType_Vespa,
    CaseConfigParamInput_QuantizationType_Vespa,
    CaseConfigParamInput_M_Vespa,
    CaseConfigParamInput_EF_Milvus,
    CaseConfigParamInput_EFConstruction_Vespa,
]
VespaPerformanceConfig = VespaLoadingConfig

CaseConfigParamInput_IndexType_LanceDB = CaseConfigInput(
    label=CaseConfigParamType.IndexType,
    inputHelp="AUTOINDEX = IVFPQ with default parameters",
    inputType=InputType.Option,
    inputConfig={
        "options": [
            IndexType.NONE.value,
            IndexType.AUTOINDEX.value,
            IndexType.IVFPQ.value,
            IndexType.HNSW.value,
        ],
    },
)

CaseConfigParamInput_num_partitions_LanceDB = CaseConfigInput(
    label=CaseConfigParamType.num_partitions,
    displayLabel="Number of Partitions",
    inputHelp="Number of partitions (clusters) for IVF_PQ. Default (when 0): sqrt(num_rows)",
    inputType=InputType.Number,
    inputConfig={
        "min": 0,
        "max": 10000,
        "value": 0,
    },
    isDisplayed=lambda config: config.get(CaseConfigParamType.IndexType, None) == IndexType.IVFPQ.value
    or config.get(CaseConfigParamType.IndexType, None) == IndexType.HNSW.value,
)

CaseConfigParamInput_num_sub_vectors_LanceDB = CaseConfigInput(
    label=CaseConfigParamType.num_sub_vectors,
    displayLabel="Number of Sub-vectors",
    inputHelp="Number of sub-vectors for PQ. Default (when 0): dim/16 or dim/8",
    inputType=InputType.Number,
    inputConfig={
        "min": 0,
        "max": 1000,
        "value": 0,
    },
    isDisplayed=lambda config: config.get(CaseConfigParamType.IndexType, None) == IndexType.IVFPQ.value
    or config.get(CaseConfigParamType.IndexType, None) == IndexType.HNSW.value,
)

CaseConfigParamInput_num_bits_LanceDB = CaseConfigInput(
    label=CaseConfigParamType.nbits,
    displayLabel="Number of Bits",
    inputHelp="Number of bits per sub-vector.",
    inputType=InputType.Option,
    inputConfig={
        "options": [4, 8],
    },
    isDisplayed=lambda config: config.get(CaseConfigParamType.IndexType, None) == IndexType.IVFPQ.value
    or config.get(CaseConfigParamType.IndexType, None) == IndexType.HNSW.value,
)

CaseConfigParamInput_sample_rate_LanceDB = CaseConfigInput(
    label=CaseConfigParamType.sample_rate,
    displayLabel="Sample Rate",
    inputHelp="Sample rate for training. Higher values are more accurate but slower",
    inputType=InputType.Number,
    inputConfig={
        "min": 16,
        "max": 1024,
        "value": 256,
    },
    isDisplayed=lambda config: config.get(CaseConfigParamType.IndexType, None) == IndexType.IVFPQ.value
    or config.get(CaseConfigParamType.IndexType, None) == IndexType.HNSW.value,
)

CaseConfigParamInput_max_iterations_LanceDB = CaseConfigInput(
    label=CaseConfigParamType.max_iterations,
    displayLabel="Max Iterations",
    inputHelp="Maximum iterations for k-means clustering",
    inputType=InputType.Number,
    inputConfig={
        "min": 10,
        "max": 200,
        "value": 50,
    },
    isDisplayed=lambda config: config.get(CaseConfigParamType.IndexType, None) == IndexType.IVFPQ.value
    or config.get(CaseConfigParamType.IndexType, None) == IndexType.HNSW.value,
)

CaseConfigParamInput_m_LanceDB = CaseConfigInput(
    label=CaseConfigParamType.m,
    displayLabel="m",
    inputHelp="m parameter in HNSW",
    inputType=InputType.Number,
    inputConfig={
        "min": 0,
        "max": 1000,
        "value": 0,
    },
    isDisplayed=lambda config: config.get(CaseConfigParamType.IndexType, None) == IndexType.HNSW.value,
)

CaseConfigParamInput_ef_construction_LanceDB = CaseConfigInput(
    label=CaseConfigParamType.ef_construction,
    displayLabel="ef_construction",
    inputHelp="ef_construction parameter in HNSW",
    inputType=InputType.Number,
    inputConfig={
        "min": 0,
        "max": 1000,
        "value": 0,
    },
    isDisplayed=lambda config: config.get(CaseConfigParamType.IndexType, None) == IndexType.HNSW.value,
)

LanceDBLoadConfig = [
    CaseConfigParamInput_IndexType_LanceDB,
    CaseConfigParamInput_num_partitions_LanceDB,
    CaseConfigParamInput_num_sub_vectors_LanceDB,
    CaseConfigParamInput_num_bits_LanceDB,
    CaseConfigParamInput_sample_rate_LanceDB,
    CaseConfigParamInput_max_iterations_LanceDB,
    CaseConfigParamInput_m_LanceDB,
    CaseConfigParamInput_ef_construction_LanceDB,
]

LanceDBPerformanceConfig = LanceDBLoadConfig

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
    DB.AWSOpenSearch: {
        CaseLabel.Load: AWSOpensearchLoadingConfig,
        CaseLabel.Performance: AWSOpenSearchPerformanceConfig,
    },
    DB.PgVector: {
        CaseLabel.Load: PgVectorLoadingConfig,
        CaseLabel.Performance: PgVectorPerformanceConfig,
    },
    DB.PgVectoRS: {
        CaseLabel.Load: PgVectoRSLoadingConfig,
        CaseLabel.Performance: PgVectoRSPerformanceConfig,
    },
    DB.PgVectorScale: {
        CaseLabel.Load: PgVectorScaleLoadingConfig,
        CaseLabel.Performance: PgVectorScalePerformanceConfig,
    },
    DB.PgDiskANN: {
        CaseLabel.Load: PgDiskANNLoadConfig,
        CaseLabel.Performance: PgDiskANNPerformanceConfig,
    },
    DB.AlloyDB: {
        CaseLabel.Load: AlloyDBLoadConfig,
        CaseLabel.Performance: AlloyDBPerformanceConfig,
    },
    DB.AliyunElasticsearch: {
        CaseLabel.Load: AliyunElasticsearchLoadingConfig,
        CaseLabel.Performance: AliyunElasticsearchPerformanceConfig,
    },
    DB.AliyunOpenSearch: {
        CaseLabel.Load: AliyunOpensearchLoadingConfig,
        CaseLabel.Performance: AliyunOpenSearchPerformanceConfig,
    },
    DB.MongoDB: {
        CaseLabel.Load: MongoDBLoadingConfig,
        CaseLabel.Performance: MongoDBPerformanceConfig,
    },
    DB.MariaDB: {
        CaseLabel.Load: MariaDBLoadingConfig,
        CaseLabel.Performance: MariaDBPerformanceConfig,
    },
    DB.Vespa: {
        CaseLabel.Load: VespaLoadingConfig,
        CaseLabel.Performance: VespaPerformanceConfig,
    },
    DB.LanceDB: {
        CaseLabel.Load: LanceDBLoadConfig,
        CaseLabel.Performance: LanceDBPerformanceConfig,
    },
}
