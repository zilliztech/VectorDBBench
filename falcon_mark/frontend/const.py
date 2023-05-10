from enum import Enum, unique

# style const
CHECKBOX_MAX_COLUMNS = 4
INPUT_MAX_COLUMNS = 3
INPUT_WIDTH_RADIO = 1.4


@unique
class DB(Enum):
    Milvus = "Milvus"
    Zilliz = "Zilliz Cloud"


@unique
class Case(Enum):
    Loading = "Loading"
    Performance = "Performance"


@unique
class InputType(Enum):
    String = "String"
    Int = "Int"
    Option = "Option"


@unique
class IndexType(Enum):
    HNSW = "HNSW"
    DiskAnn = "DiskAnn"
    Ivfflat = "IVFFlat"
    Flat = "Flat"


@unique
class CaseConfig(Enum):
    IndexType = "IndexType"
    M = "M"
    EFConstruction = "efConstruction"
    EF = "ef"
    SearchList = "search_list"
    Nlist = "nlist"
    Nprobe = "nprobe"


@unique
class DBConfig(Enum):
    URI = "URI"
    User = "User"
    Password = "Password"


DB_LIST = [DB.Milvus, DB.Zilliz]

DB_CONFIG_MAP = {
    DB.Milvus: [DBConfig.URI],
    DB.Zilliz: [DBConfig.URI, DBConfig.User, DBConfig.Password],
}


CASE_LIST = [
    {
        "name": Case.Loading,
        "intro": "intro_todo",
    },
    {
        "name": Case.Performance,
        "intro": "intro_todo",
    },
]

DEFAULT_CONFIG_CHECKED = lambda x: True

CASE_CONFIG_MAP = {
    DB.Milvus: {
        Case.Loading: [
            {
                "name": CaseConfig.IndexType,
                "inputType": InputType.Option,
                "options": [
                    IndexType.HNSW.value,
                    IndexType.Ivfflat.value,
                    IndexType.DiskAnn.value,
                    IndexType.Flat.value,
                ],
            },
            {
                "name": CaseConfig.M,
                "checked": lambda config: config[CaseConfig.IndexType]
                == IndexType.HNSW.value,
                "inputType": InputType.Int,
                "min": 4,
                "max": 64,
            },
            {
                "name": CaseConfig.EFConstruction,
                "checked": lambda config: config[CaseConfig.IndexType]
                == IndexType.HNSW.value,
                "inputType": InputType.Int,
                "min": 8,
                "max": 512,
            },
            {
                "name": CaseConfig.Nlist,
                "checked": lambda config: config[CaseConfig.IndexType]
                == IndexType.Ivfflat.value,
                "inputType": InputType.Int,
                "min": 1,
                "max": 65536,
            },
        ],
        Case.Performance: [
            {
                "name": CaseConfig.IndexType,
                "inputType": InputType.Option,
                "options": [
                    IndexType.HNSW.value,
                    IndexType.Ivfflat.value,
                    IndexType.DiskAnn.value,
                    IndexType.Flat.value,
                ],
            },
            {
                "name": CaseConfig.M,
                "checked": lambda config: config[CaseConfig.IndexType]
                == IndexType.HNSW.value,
                "inputType": InputType.Int,
                "min": 4,
                "max": 64,
            },
            {
                "name": CaseConfig.EFConstruction,
                "checked": lambda config: config[CaseConfig.IndexType]
                == IndexType.HNSW.value,
                "inputType": InputType.Int,
                "min": 8,
                "max": 512,
            },
            {
                "name": CaseConfig.EF,
                "checked": lambda config: config[CaseConfig.IndexType]
                == IndexType.HNSW.value,
                "inputType": InputType.Int,
                "min": 100,
                "max": (1 << 53) - 1,
            },
            {
                "name": CaseConfig.SearchList,
                "checked": lambda config: config[CaseConfig.IndexType]
                == IndexType.DiskAnn.value,
                "inputType": InputType.Int,
                "min": 100,
                "max": (1 << 53) - 1,
            },
            {
                "name": CaseConfig.Nlist,
                "checked": lambda config: config[CaseConfig.IndexType]
                == IndexType.Ivfflat.value,
                "inputType": InputType.Int,
                "min": 1,
                "max": 65536,
            },
            {
                "name": CaseConfig.Nprobe,
                "checked": lambda config: config[CaseConfig.IndexType]
                == IndexType.Ivfflat.value,
                "inputType": InputType.Int,
                "min": 1,
                "max": 65536,
            },
        ],
    }
}
