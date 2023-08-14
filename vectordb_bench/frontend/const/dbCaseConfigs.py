from enum import IntEnum
import typing
from pydantic import BaseModel
from vectordb_bench.backend.cases import CaseType
from vectordb_bench.backend.clients import DB
from vectordb_bench.backend.clients.api import IndexType

from vectordb_bench.models import CaseConfigParamType

MAX_STREAMLIT_INT = (1 << 53) - 1

DB_LIST = [d for d in DB]

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
    # todo type should be a function
    isDisplayed: typing.Any = lambda x: True


CaseConfigParamInput_IndexType = CaseConfigInput(
    label=CaseConfigParamType.IndexType,
    inputType=InputType.Option,
    inputConfig={
        "options": [
            IndexType.HNSW.value,
            IndexType.IVFFlat.value,
            IndexType.DISKANN.value,
            IndexType.Flat.value,
            IndexType.AUTOINDEX.value,
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
        "value": 1000,
    },
    isDisplayed=lambda config: config.get(CaseConfigParamType.IndexType, None)
    == IndexType.IVFFlat.value,
)

CaseConfigParamInput_Nprobe = CaseConfigInput(
    label=CaseConfigParamType.Nprobe,
    inputType=InputType.Number,
    inputConfig={
        "min": 1,
        "max": 65536,
        "value": 10,
    },
    isDisplayed=lambda config: config.get(CaseConfigParamType.IndexType, None)
    == IndexType.IVFFlat.value,
)

CaseConfigParamInput_Lists = CaseConfigInput(
    label=CaseConfigParamType.lists,
    inputType=InputType.Number,
    inputConfig={
        "min": 1,
        "max": 65536,
        "value": 10,
    },
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


MilvusLoadConfig = [
    CaseConfigParamInput_IndexType,
    CaseConfigParamInput_M,
    CaseConfigParamInput_EFConstruction_Milvus,
    CaseConfigParamInput_Nlist,
]
MilvusPerformanceConfig = [
    CaseConfigParamInput_IndexType,
    CaseConfigParamInput_M,
    CaseConfigParamInput_EFConstruction_Milvus,
    CaseConfigParamInput_EF_Milvus,
    CaseConfigParamInput_SearchList,
    CaseConfigParamInput_Nlist,
    CaseConfigParamInput_Nprobe,
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

PgVectorLoadingConfig = [CaseConfigParamInput_Lists]
PgVectorPerformanceConfig = [CaseConfigParamInput_Lists, CaseConfigParamInput_Probes]

CASE_CONFIG_MAP = {
    DB.Milvus: {
        CaseType.CapacityDim960: MilvusLoadConfig,
        CaseType.CapacityDim128: MilvusLoadConfig,
        CaseType.Performance768D100M: MilvusPerformanceConfig,
        CaseType.Performance768D10M: MilvusPerformanceConfig,
        CaseType.Performance768D1M: MilvusPerformanceConfig,
        CaseType.Performance768D10M1P: MilvusPerformanceConfig,
        CaseType.Performance768D1M1P: MilvusPerformanceConfig,
        CaseType.Performance768D10M99P: MilvusPerformanceConfig,
        CaseType.Performance768D1M99P: MilvusPerformanceConfig,
        CaseType.Performance1536D5M: MilvusPerformanceConfig,
        CaseType.Performance1536D500K: MilvusPerformanceConfig,
        CaseType.Performance1536D5M1P: MilvusPerformanceConfig,
        CaseType.Performance1536D500K1P: MilvusPerformanceConfig,
        CaseType.Performance1536D5M99P: MilvusPerformanceConfig,
        CaseType.Performance1536D500K99P: MilvusPerformanceConfig,        
    },
    DB.WeaviateCloud: {
        CaseType.CapacityDim960: WeaviateLoadConfig,
        CaseType.CapacityDim128: WeaviateLoadConfig,
        CaseType.Performance768D100M: WeaviatePerformanceConfig,
        CaseType.Performance768D10M: WeaviatePerformanceConfig,
        CaseType.Performance768D1M: WeaviatePerformanceConfig,
        CaseType.Performance768D10M1P: WeaviatePerformanceConfig,
        CaseType.Performance768D1M1P: WeaviatePerformanceConfig,
        CaseType.Performance768D10M99P: WeaviatePerformanceConfig,
        CaseType.Performance768D1M99P: WeaviatePerformanceConfig,
        CaseType.Performance1536D5M: WeaviatePerformanceConfig,
        CaseType.Performance1536D500K: WeaviatePerformanceConfig,
        CaseType.Performance1536D5M1P: WeaviatePerformanceConfig,
        CaseType.Performance1536D500K1P: WeaviatePerformanceConfig,
        CaseType.Performance1536D5M99P: WeaviatePerformanceConfig,
        CaseType.Performance1536D500K99P: WeaviatePerformanceConfig,
    },
    DB.ElasticCloud: {
        CaseType.CapacityDim960: ESLoadingConfig,
        CaseType.CapacityDim128: ESLoadingConfig,
        CaseType.Performance768D100M: ESPerformanceConfig,
        CaseType.Performance768D10M: ESPerformanceConfig,
        CaseType.Performance768D1M: ESPerformanceConfig,
        CaseType.Performance768D10M1P: ESPerformanceConfig,
        CaseType.Performance768D1M1P: ESPerformanceConfig,
        CaseType.Performance768D10M99P: ESPerformanceConfig,
        CaseType.Performance768D1M99P: ESPerformanceConfig,
        CaseType.Performance1536D5M: ESPerformanceConfig,
        CaseType.Performance1536D500K: ESPerformanceConfig,
        CaseType.Performance1536D5M1P: ESPerformanceConfig,
        CaseType.Performance1536D500K1P: ESPerformanceConfig,
        CaseType.Performance1536D5M99P: ESPerformanceConfig,
        CaseType.Performance1536D500K99P: ESPerformanceConfig,
    },
    DB.PgVector: {
        CaseType.CapacityDim960: PgVectorLoadingConfig,
        CaseType.CapacityDim128: PgVectorLoadingConfig,
        CaseType.Performance768D100M: PgVectorPerformanceConfig,
        CaseType.Performance768D10M: PgVectorPerformanceConfig,
        CaseType.Performance768D1M: PgVectorPerformanceConfig,
        CaseType.Performance768D10M1P: PgVectorPerformanceConfig,
        CaseType.Performance768D1M1P: PgVectorPerformanceConfig,
        CaseType.Performance768D10M99P: PgVectorPerformanceConfig,
        CaseType.Performance768D1M99P: PgVectorPerformanceConfig,
        CaseType.Performance1536D5M: PgVectorPerformanceConfig,
        CaseType.Performance1536D500K: PgVectorPerformanceConfig,
        CaseType.Performance1536D5M1P: PgVectorPerformanceConfig,
        CaseType.Performance1536D500K1P: PgVectorPerformanceConfig,
        CaseType.Performance1536D5M99P: PgVectorPerformanceConfig,
        CaseType.Performance1536D500K99P: PgVectorPerformanceConfig,
    },
}
