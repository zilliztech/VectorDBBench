from enum import IntEnum, Enum
import typing
from pydantic import BaseModel
from vectordb_bench.backend.cases import CaseLabel, CaseType
from vectordb_bench.backend.clients import DB
from vectordb_bench.backend.clients.api import IndexType, MetricType
from vectordb_bench.backend.dataset import DatasetWithSizeType
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


class InputType(IntEnum):
    Text = 20001
    Number = 20002
    Option = 20003
    Float = 20004
    Bool = 20005


class ConfigInput(BaseModel):
    label: CaseConfigParamType
    inputType: InputType = InputType.Text
    inputConfig: dict = {}
    inputHelp: str = ""
    displayLabel: str = ""


class CaseConfigInput(ConfigInput):
    # todo type should be a function
    isDisplayed: typing.Any = lambda config: True


class UICaseItem(BaseModel):
    isLine: bool = False
    key: str = ""
    label: str = ""
    description: str = ""
    cases: list[CaseConfig] = []
    caseLabel: CaseLabel = CaseLabel.Performance
    extra_custom_case_config_inputs: list[ConfigInput] = []
    tmp_custom_config: dict = dict()

    def __init__(
        self,
        isLine: bool = False,
        cases: list[CaseConfig] = None,
        label: str = "",
        description: str = "",
        caseLabel: CaseLabel = CaseLabel.Performance,
        **kwargs,
    ):
        if isLine is True:
            super().__init__(isLine=True, **kwargs)
        if cases is None:
            cases = []
        elif len(cases) == 1:
            c = cases[0].case
            super().__init__(
                label=label if label else c.name,
                description=description if description else c.description,
                cases=cases,
                caseLabel=caseLabel,
                **kwargs,
            )
        else:
            super().__init__(
                label=label,
                description=description,
                cases=cases,
                caseLabel=caseLabel,
                **kwargs,
            )

    def __hash__(self) -> int:
        return hash(self.key if self.key else self.label)

    def get_cases(self) -> list[CaseConfig]:
        # return self.cases
        if len(self.extra_custom_case_config_inputs) == 0:
            return self.cases
        cases = [
            CaseConfig(
                case_id=c.case_id,
                k=c.k,
                concurrency_search_config=c.concurrency_search_config,
                custom_case={**c.custom_case, **self.tmp_custom_config},
            )
            for c in self.cases
        ]
        return cases


class UICaseItemCluster(BaseModel):
    label: str = ""
    uiCaseItems: list[UICaseItem] = []


def get_custom_case_items() -> list[UICaseItem]:
    custom_configs = get_custom_configs()
    return [
        UICaseItem(
            cases=[
                CaseConfig(
                    case_id=CaseType.PerformanceCustomDataset,
                    custom_case=custom_config.dict(),
                )
            ]
        )
        for custom_config in custom_configs
    ]


def generate_normal_cases(case_id: CaseType, custom_case: dict | None = None) -> list[CaseConfig]:
    return [CaseConfig(case_id=case_id, custom_case=custom_case)]


def get_custom_case_cluter() -> UICaseItemCluster:
    return UICaseItemCluster(label="Custom Search Performance Test", uiCaseItems=get_custom_case_items())


def generate_custom_streaming_case() -> CaseConfig:
    return CaseConfig(
        case_id=CaseType.StreamingPerformanceCase,
        custom_case=dict(),
    )


custom_streaming_config: list[ConfigInput] = [
    ConfigInput(
        label=CaseConfigParamType.dataset_with_size_type,
        displayLabel="dataset",
        inputType=InputType.Option,
        inputConfig=dict(options=[dataset.value for dataset in DatasetWithSizeType]),
    ),
    ConfigInput(
        label=CaseConfigParamType.insert_rate,
        inputType=InputType.Number,
        inputConfig=dict(step=100, min=100, max=4_000, value=1000),
        inputHelp="fixed insertion rate (rows/s), must be divisible by 100",
    ),
    ConfigInput(
        label=CaseConfigParamType.search_stages,
        inputType=InputType.Text,
        inputConfig=dict(value="[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]"),
        inputHelp="0<=stage<1.0; do search test when inserting a specified amount of data.",
    ),
    ConfigInput(
        label=CaseConfigParamType.concurrencies,
        inputType=InputType.Text,
        inputConfig=dict(value="[2, 3]"),
        inputHelp="concurrent num of search test while insertion; record max-qps.",
    ),
    ConfigInput(
        label=CaseConfigParamType.optimize_after_write,
        inputType=InputType.Option,
        inputConfig=dict(options=[True, False]),
        inputHelp="whether to optimize after inserting all data",
    ),
    ConfigInput(
        label=CaseConfigParamType.read_dur_after_write,
        inputType=InputType.Number,
        inputConfig=dict(step=10, min=30, max=360_000, value=30),
        inputHelp="search test duration after inserting all data",
    ),
]


def generate_label_filter_cases(dataset_with_size_type: DatasetWithSizeType) -> list[CaseConfig]:
    label_percentages = dataset_with_size_type.get_manager().data.scalar_label_percentages
    return [
        CaseConfig(
            case_id=CaseType.LabelFilterPerformanceCase,
            custom_case=dict(dataset_with_size_type=dataset_with_size_type, label_percentage=label_percentage),
        )
        for label_percentage in label_percentages
    ]


UI_CASE_CLUSTERS: list[UICaseItemCluster] = [
    UICaseItemCluster(
        label="Search Performance Test",
        uiCaseItems=[
            UICaseItem(cases=generate_normal_cases(CaseType.Performance768D100M)),
            UICaseItem(cases=generate_normal_cases(CaseType.Performance768D10M)),
            UICaseItem(cases=generate_normal_cases(CaseType.Performance768D1M)),
            UICaseItem(isLine=True),
            UICaseItem(cases=generate_normal_cases(CaseType.Performance1536D5M)),
            UICaseItem(cases=generate_normal_cases(CaseType.Performance1536D500K)),
            UICaseItem(cases=generate_normal_cases(CaseType.Performance1536D50K)),
        ],
    ),
    UICaseItemCluster(
        label="Int-Filter Search Performance Test",
        uiCaseItems=[
            UICaseItem(cases=generate_normal_cases(CaseType.Performance768D10M1P)),
            UICaseItem(cases=generate_normal_cases(CaseType.Performance768D10M99P)),
            UICaseItem(cases=generate_normal_cases(CaseType.Performance768D1M1P)),
            UICaseItem(cases=generate_normal_cases(CaseType.Performance768D1M99P)),
            UICaseItem(isLine=True),
            UICaseItem(cases=generate_normal_cases(CaseType.Performance1536D5M1P)),
            UICaseItem(cases=generate_normal_cases(CaseType.Performance1536D5M99P)),
            UICaseItem(cases=generate_normal_cases(CaseType.Performance1536D500K1P)),
            UICaseItem(cases=generate_normal_cases(CaseType.Performance1536D500K99P)),
        ],
    ),
    UICaseItemCluster(
        label="Label-Filter Search Performance Test",
        uiCaseItems=[
            UICaseItem(
                label=f"Label-Filter Search Performance Test - {dataset_with_size_type.value}",
                description=(
                    f'[Batch Cases] These cases evaluate search performance under filtering constraints like "color==red." '
                    "Vdbbench provides an additional column of randomly distributed labels with fixed proportions, "
                    f"such as {dataset_with_size_type.get_manager().data.scalar_label_percentages}. "
                    f"Essentially, vdbbench will test each filter label in {dataset_with_size_type.value} to "
                    "assess the vector database's search performance across different filtering conditions. "
                ),
                cases=generate_label_filter_cases(dataset_with_size_type),
            )
            for dataset_with_size_type in DatasetWithSizeType
        ],
    ),
    UICaseItemCluster(
        label="Capacity Test",
        uiCaseItems=[
            UICaseItem(cases=generate_normal_cases(CaseType.CapacityDim960)),
            UICaseItem(cases=generate_normal_cases(CaseType.CapacityDim128)),
        ],
    ),
    UICaseItemCluster(
        label="Streaming Test",
        uiCaseItems=[
            UICaseItem(
                label="Customize Streaming Test",
                description=(
                    "This case test the search performance during insertion. "
                    "VDBB will send insert requests to VectorDB at a fixed rate and "
                    "conduct a search test once the insert count reaches the search_stages. "
                    "After all data is inserted, optimization and search tests can be "
                    "optionally performed."
                ),
                cases=[generate_custom_streaming_case()],
                extra_custom_case_config_inputs=custom_streaming_config,
            )
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
    CaseType.CapacityDim960,
    CaseType.CapacityDim128,
]
CASE_NAME_ORDER = [case.case_cls().name for case in DISPLAY_CASE_ORDER]

# CASE_LIST = [
#     item for item in CASE_LIST_WITH_DIVIDER if isinstance(item, CaseType)]


CaseConfigParamInput_IndexType = CaseConfigInput(
    label=CaseConfigParamType.IndexType,
    inputType=InputType.Option,
    inputConfig={
        "options": [
            IndexType.HNSW.value,
            IndexType.IVFFlat.value,
            IndexType.IVFSQ8.value,
            IndexType.DISKANN.value,
            IndexType.STREAMING_DISKANN.value,
            IndexType.Flat.value,
            IndexType.AUTOINDEX.value,
            IndexType.GPU_IVF_FLAT.value,
            IndexType.GPU_IVF_PQ.value,
            IndexType.GPU_CAGRA.value,
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
    isDisplayed=lambda config: config.get(CaseConfigParamType.IndexType, None) == IndexType.HNSW.value,
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
    isDisplayed=lambda config: config[CaseConfigParamType.IndexType] == IndexType.HNSW.value,
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
    isDisplayed=lambda config: config[CaseConfigParamType.IndexType] == IndexType.HNSW.value,
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
    isDisplayed=lambda config: config.get(CaseConfigParamType.IndexType, None) in [IndexType.GPU_IVF_PQ.value],
)


CaseConfigParamInput_Nbits_PQ = CaseConfigInput(
    label=CaseConfigParamType.nbits,
    inputType=InputType.Number,
    inputConfig={
        "min": 1,
        "max": 65536,
        "value": 8,
    },
    isDisplayed=lambda config: config.get(CaseConfigParamType.IndexType, None) in [IndexType.GPU_IVF_PQ.value],
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

CaseConfigParamInput_Milvus_use_partition_key = CaseConfigInput(
    label=CaseConfigParamType.use_partition_key,
    inputType=InputType.Option,
    inputHelp="whether to use partition_key for label-filter cases. only works in label-filter cases",
    inputConfig={"options": [True, False]},
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
    CaseConfigParamInput_Milvus_use_partition_key,
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
    CaseConfigParamInput_Milvus_use_partition_key,
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

CASE_CONFIG_MAP = {
    DB.Milvus: {
        CaseLabel.Load: MilvusLoadConfig,
        CaseLabel.Performance: MilvusPerformanceConfig,
        CaseLabel.Streaming: MilvusPerformanceConfig,
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
}


def get_case_config_inputs(db: DB, case_label: CaseLabel) -> list[CaseConfigInput]:
    if db not in CASE_CONFIG_MAP:
        return []
    if case_label == CaseLabel.Load:
        return CASE_CONFIG_MAP[db][CaseLabel.Load]
    elif case_label == CaseLabel.Performance or case_label == CaseLabel.Streaming:
        return CASE_CONFIG_MAP[db][CaseLabel.Performance]
    return []
