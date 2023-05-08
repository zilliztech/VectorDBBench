from enum import IntEnum
from vector_db_bench.models import DB, CaseType, IndexType, CaseConfigParamType
from pydantic import BaseModel
import typing

# style const
DB_SELECTOR_COLUMNS = 6
DB_CONFIG_SETTING_COLUMNS = 3
CASE_CONFIG_SETTING_COLUMNS = 4
CHECKBOX_INDENT = 30
TASK_LABEL_INPUT_COLUMNS = 2
CHECKBOX_MAX_COLUMNS = 4
DB_CONFIG_INPUT_MAX_COLUMNS = 2
CASE_CONFIG_INPUT_MAX_COLUMNS = 3
DB_CONFIG_INPUT_WIDTH_RADIO = 2
CASE_CONFIG_INPUT_WIDTH_RADIO = 0.98
CASE_INTRO_RATIO = 3
MAX_STREAMLIT_INT = (1 << 53) - 1

COLORS = [
    "#3B69FE",
    "#66C8FF",
    "#35CE73",
    "#FDC513",
    "#FE708D",
    "#8773FB",
]
LEGEND_RECT_WIDTH = 24
LEGEND_RECT_HEIGHT = 16
LEGEND_TEXT_FONT_SIZE = 14

PATTERN_SHAPES = ["", "+", "\\", "x", ".", "|", "/", "-"]


def getPatternShape(i):
    return PATTERN_SHAPES[i % len(PATTERN_SHAPES)]


MAX_AUTO_REFRESH_COUNT = 999999
MAX_AUTO_REFRESH_INTERVAL = 5000  # 2s


DB_LIST = [d for d in DB]

DB_TO_ICON = {
    DB.Milvus: "https://assets.zilliz.com/milvus_c30b0d1994.png",
    DB.ZillizCloud: "https://assets.zilliz.com/zilliz_5f4cc9b050.png",
    DB.ElasticCloud: "https://assets.zilliz.com/elasticsearch_beffeadc29.png",
    DB.Pinecone: "https://assets.zilliz.com/pinecone_94d8154979.png",
    DB.QdrantCloud: "https://assets.zilliz.com/qdrant_b691674fcd.png",
    DB.WeaviateCloud: "https://assets.zilliz.com/weaviate_4f6f171ebe.png",
}

COLOR_MAP = {db.value: COLORS[i] for i, db in enumerate(DB_LIST)}

CASE_LIST = [
    {
        "name": CaseType.LoadSDim,
        "intro": """This case tests the vector database's loading capacity by repeatedly inserting small-dimension vectors (SIFT 100K vectors, <b>128 dimensions</b>) until it is fully loaded.  
Number of inserted vectors will be reported.""",
    },
    {
        "name": CaseType.LoadLDim,
        "divider": True,
        "intro": """This case tests the vector database's loading capacity by repeatedly inserting large-dimension vectors (GIST 100K vectors, <b>960 dimensions</b>) until it is fully loaded.  
Number of inserted vectors will be reported.
""",
    },
    {
        "name": CaseType.PerformanceSZero,
        "intro": """This case tests the search performance of a vector database with a small dataset (<b>Cohere 100K vectors</b>, 768 dimensions) at varying parallel levels.
Results will show index building time, recall, and maximum QPS.""",
    },
    {
        "name": CaseType.PerformanceMZero,
        "intro": """This case tests the search performance of a vector database with a medium dataset (<b>Cohere 1M vectors</b>, 768 dimensions) at varying parallel levels.
Results will show index building time, recall, and maximum QPS.""",
    },
    {
        "name": CaseType.PerformanceLZero,
        "intro": """This case tests the search performance of a vector database with a large dataset (<b>Cohere 10M vectors</b>, 768 dimensions) at varying parallel levels.
Results will show index building time, recall, and maximum QPS.""",
    },
    {
        "name": CaseType.Performance100M,
        "divider": True,
        "intro": """This case tests the search performance of a vector database with a large 100M dataset (<b>LAION 100M vectors</b>, 768 dimensions), at varying parallel levels.
Results will show index building time, recall, and maximum QPS.""",
    },
    {
        "name": CaseType.PerformanceSLow,
        "intro": """This case tests the search performance of a vector database with a small dataset (<b>Cohere 100K vectors</b>, 768 dimensions) under a low filtering rate (<b>1% vectors</b>), at varying parallel levels.
Results will show index building time, recall, and maximum QPS.""",
    },
    {
        "name": CaseType.PerformanceMLow,
        "intro": """This case tests the search performance of a vector database with a medium dataset (<b>Cohere 1M vectors</b>, 768 dimensions) under a low filtering rate (<b>1% vectors</b>), at varying parallel levels.
Results will show index building time, recall, and maximum QPS.""",
    },
    {
        "name": CaseType.PerformanceLLow,
        "intro": """This case tests the search performance of a vector database with a large dataset (<b>Cohere 10M vectors</b>, 768 dimensions) under a low filtering rate (<b>1% vectors</b>), at varying parallel levels.
Results will show index building time, recall, and maximum QPS.""",
    },
    {
        "name": CaseType.PerformanceSHigh,
        "intro": """This case tests the search performance of a vector database with a small dataset (<b>Cohere 100K vectors</b>, 768 dimensions) under a high filtering rate (<b>99% vectors</b>), at varying parallel levels.
Results will show index building time, recall, and maximum QPS.""",
    },
    {
        "name": CaseType.PerformanceMHigh,
        "intro": """This case tests the search performance of a vector database with a medium dataset (<b>Cohere 1M vectors</b>, 768 dimensions) under a high filtering rate (<b>99% vectors</b>), at varying parallel levels.
Results will show index building time, recall, and maximum QPS.""",
    },
    {
        "name": CaseType.PerformanceLHigh,
        "intro": """This case tests the search performance of a vector database with a large dataset (<b>Cohere 10M vectors</b>, 768 dimensions) under a high filtering rate (<b>99% vectors</b>), at varying parallel levels.
Results will show index building time, recall, and maximum QPS.""",
    },
]


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

CASE_CONFIG_MAP = {
    DB.Milvus: {
        CaseType.LoadLDim: MilvusLoadConfig,
        CaseType.LoadSDim: MilvusLoadConfig,
        CaseType.PerformanceLZero: MilvusPerformanceConfig,
        CaseType.PerformanceMZero: MilvusPerformanceConfig,
        CaseType.PerformanceSZero: MilvusPerformanceConfig,
        CaseType.PerformanceLLow: MilvusPerformanceConfig,
        CaseType.PerformanceMLow: MilvusPerformanceConfig,
        CaseType.PerformanceSLow: MilvusPerformanceConfig,
        CaseType.PerformanceLHigh: MilvusPerformanceConfig,
        CaseType.PerformanceMHigh: MilvusPerformanceConfig,
        CaseType.PerformanceSHigh: MilvusPerformanceConfig,
        CaseType.Performance100M: MilvusPerformanceConfig,
    },
    DB.WeaviateCloud: {
        CaseType.LoadLDim: WeaviateLoadConfig,
        CaseType.LoadSDim: WeaviateLoadConfig,
        CaseType.PerformanceLZero: WeaviatePerformanceConfig,
        CaseType.PerformanceMZero: WeaviatePerformanceConfig,
        CaseType.PerformanceSZero: WeaviatePerformanceConfig,
        CaseType.PerformanceLLow: WeaviatePerformanceConfig,
        CaseType.PerformanceMLow: WeaviatePerformanceConfig,
        CaseType.PerformanceSLow: WeaviatePerformanceConfig,
        CaseType.PerformanceLHigh: WeaviatePerformanceConfig,
        CaseType.PerformanceMHigh: WeaviatePerformanceConfig,
        CaseType.PerformanceSHigh: WeaviatePerformanceConfig,
        CaseType.Performance100M: WeaviatePerformanceConfig,
    },
    DB.ElasticCloud: {
        CaseType.LoadLDim: ESLoadingConfig,
        CaseType.LoadSDim: ESLoadingConfig,
        CaseType.PerformanceLZero: ESPerformanceConfig,
        CaseType.PerformanceMZero: ESPerformanceConfig,
        CaseType.PerformanceSZero: ESPerformanceConfig,
        CaseType.PerformanceLLow: ESPerformanceConfig,
        CaseType.PerformanceMLow: ESPerformanceConfig,
        CaseType.PerformanceSLow: ESPerformanceConfig,
        CaseType.PerformanceLHigh: ESPerformanceConfig,
        CaseType.PerformanceMHigh: ESPerformanceConfig,
        CaseType.PerformanceSHigh: ESPerformanceConfig,
        CaseType.Performance100M: ESPerformanceConfig,
    },
}

DB_DBLABEL_TO_PRICE = {
    DB.Milvus.value: {},
    DB.ZillizCloud.value: {
        "1cu-perf": 0.159,
        "8cu-perf": 1.272,
        "1cu-cap": 0.159,
        "2cu-cap": 0.318,
    },
    DB.WeaviateCloud.value: {
        # "sandox": 0, # emmmm
        "standard": 10.10,
        "bus_crit": 32.60,
    },
    DB.ElasticCloud.value: {
        "free-5c8g": 0.260,
        "upTo2.5c8g": 0.4793,
    },
    DB.QdrantCloud.value: {
        "0.5c4g-1node": 0.052,
        "2c8g-1node": 0.166,
        "4c16g-5node": 1.426,
    },
    DB.Pinecone.value: {
        "s1.x1": 0.0973,
        "s1.x2": 0.194,
        "p1.x1": 0.0973,
        "p1.x5-2pod": 0.973,
        "p2.x1": 0.146,
        "p2.x5-2pod": 1.46,
    },
}
