from enum import Enum, auto
from pydantic import BaseModel
from pydantic.dataclasses import dataclass

@dataclass
class GIST:
    name: str = "GIST"
    dim: int = 960
    metric_type: str = "L2"

@dataclass
class Cohere:
    name: str = "Cohere"
    dim: int = 768
    metric_type: str = "Consine"

@dataclass
class Glove:
    name: str = "Glove"
    dim: int = 200
    metric_type: str = "Consine"

@dataclass
class SIFT:
    name: str = "SIFT"
    dim: int = 128
    metric_type: str = "L2"

@dataclass
class GIST_S(GIST):
    name: str  = "GIST_S_100K"
    label: str = "SMALL"
    size: int  = 100_000

@dataclass
class GIST_M(GIST):
    name: str  = "GIST_M_1M"
    label: str = "MEDIUM"
    size: int  = 1_000_000

@dataclass
class Cohere_S(Cohere):
    name: str  = "Cohere_S_100K"
    label: str = "SMALL"
    size: int  = 100_000

@dataclass
class Cohere_M(Cohere):
    name: str = "Cohere_M_1M"
    label: str = "MEDIUM"
    size: int = 1_000_000

@dataclass
class Cohere_L(Cohere):
    name  : str = "Cohere_L_10M"
    label : str = "LARGE"
    size  : int = 10_000_000

@dataclass
class Glove_S(Glove):
    name : str = "Glove_S_100K"
    label: str = "SMALL"
    size : int = 100_000

@dataclass
class Glove_M(Glove):
    name : str = "Glove_M_1M"
    label: str = "MEDIUM"
    size : int = 1_000_000

@dataclass
class SIFT_S(SIFT):
    name : str = "SIFT_S_500K"
    label: str = "SMALL"
    size : int = 500_000

@dataclass
class SIFT_M(SIFT):
    name : str = "SIFT_M_5M"
    label: str = "MEDIUM"
    size : int = 5_000_000

@dataclass
class SIFT_L(SIFT):
    name : str = "SIFT_L_50M"
    label: str = "LARGE"
    size : int = 50_000_000


class DataSetManager(BaseModel):
    data:   GIST | Cohere | Glove | SIFT
    data_path: str

    def prepare(self) -> bool:
        """Download the dataset from the default url"""
        pass

    def batch(self):
        # yield
        pass

    def ground_truth(self):
        # yield
        pass

class DataSet(Enum):
    GIST = auto()
    Cohere = auto()
    Glove = auto()
    SIFT = auto()

class Label(Enum):
    SMALL = auto()
    MEDIUM = auto()
    LARGE = auto()

_global_ds_mapping = {
    DataSet.GIST: {
        Label.SMALL: GIST_S(),
        Label.MEDIUM: GIST_M(),
    },
    DataSet.Cohere: {
        Label.SMALL: Cohere_S(),
        Label.MEDIUM: Cohere_M(),
        Label.LARGE: Cohere_L(),
    },
    DataSet.Glove:{
        Label.SMALL: Glove_S(),
        Label.MEDIUM: Glove_M(),
    },
    DataSet.SIFT: {
        Label.SMALL: SIFT_S(),
        Label.MEDIUM: SIFT_M(),
        Label.LARGE: SIFT_L(),
    },
}

def get_data_set(ds: DataSet, label: Label):
    return _global_ds_mapping.get(ds, {}).get(label)
