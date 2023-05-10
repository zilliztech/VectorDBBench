"""
Usage:
    >>> from xxx import dataset as ds
    >>> gist_s = ds.get(ds.Name.GIST, ds.Label.SMALL)
    >>> gist_s.model_dump()
    dataset: {'data': {'name': 'GIST', 'dim': 128, 'metric_type': 'L2', 'label': 'SMALL', 'size': 50000000}, 'data_dir': 'xxx'}
"""

import os
from enum import Enum, auto
from typing import Optional, Type
from decimal import Decimal
from pydantic import BaseModel, validator, computed_field, ConfigDict
from pydantic.dataclasses import dataclass
from .. import DATASET_LOCAL_DIR


def numerize(n) -> str:
    """nuimerize display of positive number

    Examples:
        >>> numerize(1_000)
        '1K'
    """
    def round_num(n):
        n=Decimal(n)
        return n.to_integral() if n == n.to_integral() else round(n.normalize(), 2)

    sufixes = [ "", "K", "M", "B", "END"] 
    sci_expr = [1e0, 1e3, 1e6, 1e9 ]
    for x in range(len(sci_expr)):
        try:
            if n >= sci_expr[x] and n < sci_expr[x+1]:
                sufix = sufixes[x]
                if n >= 1e3:
                    num = str(round_num(n/sci_expr[x]))
                else:
                    num = str(n)
                return f"{num}{sufix}"
        except IndexError:
            print("You've reached the end")


@dataclass
class GIST:
    name: str = "GIST"
    dim: int = 960
    metric_type: str = "L2"

    @computed_field
    @property
    def dir_name(self) -> str:
        return f"{self.name}_{self.label}_{numerize(self.size)}".lower()

@dataclass
class Cohere:
    name: str = "Cohere"
    dim: int = 768
    metric_type: str = "Consine"

    @computed_field
    @property
    def dir_name(self) -> str:
        return f"{self.name}_{self.label}_{numerize(self.size)}".lower()

@dataclass
class Glove:
    name: str = "Glove"
    dim: int = 200
    metric_type: str = "Consine"
    
    @computed_field
    @property
    def dir_name(self) -> str:
        return f"{self.name}_{self.label}_{numerize(self.size)}".lower()

@dataclass
class SIFT:
    name: str = "SIFT"
    dim: int = 128
    metric_type: str = "L2"

    @computed_field
    @property
    def dir_name(self) -> str:
        return f"{self.name}_{self.label}_{numerize(self.size)}".lower()

@dataclass
class GIST_S(GIST):
    label: str = "SMALL"
    size: int  = 100_000

@dataclass
class GIST_M(GIST):
    label: str = "MEDIUM"
    size: int  = 1_000_000

@dataclass
class Cohere_S(Cohere):
    label: str = "SMALL"
    size: int  = 100_000

@dataclass
class Cohere_M(Cohere):
    label: str = "MEDIUM"
    size: int = 1_000_000

@dataclass
class Cohere_L(Cohere):
    label : str = "LARGE"
    size  : int = 10_000_000

@dataclass
class Glove_S(Glove):
    label: str = "SMALL"
    size : int = 100_000

@dataclass
class Glove_M(Glove):
    label: str = "MEDIUM"
    size : int = 1_000_000

@dataclass
class SIFT_S(SIFT):
    label: str = "SMALL"
    size : int = 500_000

@dataclass
class SIFT_M(SIFT):
    label: str = "MEDIUM"
    size : int = 5_000_000

@dataclass
class SIFT_L(SIFT):
    label: str = "LARGE"
    size : int = 50_000_000


class DataSet(BaseModel):
    """Download dataset if not int the local directory. Provide data for cases.

    data local directory: DATASET_LOCAL_DIR/{dataset_name}/{dataset_dirname}
        {dataset_name} = ds.name in DataSet
        {dataset_dirname} = {ds.name}_{ds.label}_{ds.size}

        `DATASET_LOCAL_DIR/sift/sift_small_500k/`
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    data:   GIST | Cohere | Glove | SIFT

    @computed_field
    @property
    def data_dir(self) -> str:
        relative_path = os.path.join(self.data.name, self.data.dir_name).lower()
        return os.path.join(DATASET_LOCAL_DIR, relative_path)


    def prepare(self) -> bool:
        """Download the dataset from S3"""
        # TODO
        return True

    def batch(self):
        # yield
        pass

    def get_line(self) -> list[float] | bytes:
        pass

    def get_lines(self, size: int) -> list[list[float]]:
        pass

    def ground_truth(self):
        # yield
        pass


class Name(Enum):
    GIST = auto()
    Cohere = auto()
    Glove = auto()
    SIFT = auto()


class Label(Enum):
    SMALL = auto()
    MEDIUM = auto()
    LARGE = auto()

_global_ds_mapping = {
    Name.GIST: {
        Label.SMALL: DataSet(data=GIST_S()),
        Label.MEDIUM: DataSet(data=GIST_M()),
    },
    Name.Cohere: {
        Label.SMALL: DataSet(data=Cohere_S()),
        Label.MEDIUM: DataSet(data=Cohere_M()),
        Label.LARGE: DataSet(data=Cohere_L()),
    },
    Name.Glove:{
        Label.SMALL: DataSet(data=Glove_S()),
        Label.MEDIUM: DataSet(data=Glove_M()),
    },
    Name.SIFT: {
        Label.SMALL: DataSet(data=SIFT_S()),
        Label.MEDIUM: DataSet(data=SIFT_M()),
        Label.LARGE: DataSet(data=SIFT_L()),
    },
}

def get(ds: Name, label: Label):
    return _global_ds_mapping.get(ds, {}).get(label)
