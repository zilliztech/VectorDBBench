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

# AWS OpenSearch specific inputs
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
        "min": 1,
        "max": 1024,
        "value": 256,
    },
)

CaseConfigParamInput_INDEX_THREAD_QTY_DURING_FORCE_MERGE_AWSOpensearch = CaseConfigInput(
    label=CaseConfigParamType.index_thread_qty_during_force_merge,
    displayLabel="Index Thread Qty During Force Merge",
    inputHelp="Thread count during force merge operations",
    inputType=InputType.Number,
    inputConfig={
        "min": 1,
        "max": 32,
        "value": 4,
    },
)

CaseConfigParamInput_NUMBER_OF_INDEXING_CLIENTS_AWSOpensearch = CaseConfigInput(
    label=CaseConfigParamType.number_of_indexing_clients,
    displayLabel="Number of Indexing Clients",
    inputHelp="Number of concurrent clients for data insertion",
    inputType=InputType.Number,
    inputConfig={
        "min": 1,
        "max": 32,
        "value": 1,
    },
)

CaseConfigParamInput_NUMBER_OF_SHARDS_AWSOpensearch = CaseConfigInput(
    label=CaseConfigParamType.number_of_shards,
    displayLabel="Number of Shards",
    inputHelp="Number of primary shards for the index",
    inputType=InputType.Number,
    inputConfig={
        "min": 1,
        "max": 32,
        "value": 1,
    },
)

CaseConfigParamInput_NUMBER_OF_REPLICAS_AWSOpensearch = CaseConfigInput(
    label=CaseConfigParamType.number_of_replicas,
    displayLabel="Number of Replicas",
    inputHelp="Number of replica copies for each primary shard",
    inputType=InputType.Number,
    inputConfig={
        "min": 0,
        "max": 10,
        "value": 1,
    },
)

CaseConfigParamInput_INDEX_THREAD_QTY_AWSOpensearch = CaseConfigInput(
    label=CaseConfigParamType.index_thread_qty,
    displayLabel="Index Thread Qty",
    inputHelp="Thread count for native engine indexing",
    inputType=InputType.Number,
    inputConfig={
        "min": 1,
        "max": 32,
        "value": 4,
    },
)

# AWS OpenSearch config lists
AWSOpensearchLoadingConfig = [
    CaseConfigParamInput_EFConstruction_AWSOpensearch,
    CaseConfigParamInput_M_AWSOpensearch,
    CaseConfigParamInput_INDEX_THREAD_QTY_DURING_FORCE_MERGE_AWSOpensearch,
    CaseConfigParamInput_NUMBER_OF_INDEXING_CLIENTS_AWSOpensearch,
    CaseConfigParamInput_NUMBER_OF_SHARDS_AWSOpensearch,
    CaseConfigParamInput_NUMBER_OF_REPLICAS_AWSOpensearch,
    CaseConfigParamInput_INDEX_THREAD_QTY_AWSOpensearch,
]

AWSOpenSearchPerformanceConfig = [
    CaseConfigParamInput_EFConstruction_AWSOpensearch,
    CaseConfigParamInput_M_AWSOpensearch,
    CaseConfigParamInput_EF_SEARCH_AWSOpensearch,
    CaseConfigParamInput_INDEX_THREAD_QTY_DURING_FORCE_MERGE_AWSOpensearch,
    CaseConfigParamInput_NUMBER_OF_INDEXING_CLIENTS_AWSOpensearch,
    CaseConfigParamInput_NUMBER_OF_SHARDS_AWSOpensearch,
    CaseConfigParamInput_NUMBER_OF_REPLICAS_AWSOpensearch,
    CaseConfigParamInput_INDEX_THREAD_QTY_AWSOpensearch,
]

# Map DB to config
CASE_CONFIG_MAP = {
    DB.AWSOpenSearch: {
        CaseLabel.Load: AWSOpensearchLoadingConfig,
        CaseLabel.Performance: AWSOpenSearchPerformanceConfig,
    },
}
