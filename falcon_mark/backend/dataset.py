from pydantic import BaseModel

class DataSet(BaseModel):
    name:      str
    dim:       int
    data_path: str
    size:      int
    metric_type: str

    def prepare(self, url: str) -> bool:
        """Download the dataset"""
        pass

    def batch(self):
        # yield
        pass

    def ground_truth(self):
        # yield
        pass

class GIST(DataSet, BaseModel):
    dim: int = 960
    metric_type: str = "L2"

class Cohere(DataSet, BaseModel):
    dim: int = 768
    metric_type: str = "Consine"

class Glove(DataSet, BaseModel):
    dim: int = 200
    metric_type: str = "Consine"

class SIFT(DataSet, BaseModel):
    dim: int = 128
    metric_type: str = "L2"

class GIST_S(GIST, BaseModel):
    name: str = "GIST_S_100K"
    size: 100_000

class GIST_M(GIST, BaseModel):
    name: str = "GIST_M_1M"
    size: 1_000_000

class Cohere_S(Cohere, BaseModel):
    name: str = "Cohere_S_100K"
    size: 100_000

class Cohere_M(Cohere, BaseModel):
    name: str = "Cohere_M_1M"
    size: 1_000_000

class Cohere_L(Cohere, BaseModel):
    name: str = "Cohere_L_10M"
    size: 10_000_000

class Glove_S(Glove, BaseModel):
    name: str = "Glove_S_100K"
    size: 100_000

class Glove_M(Glove, BaseModel):
    name: str = "Glove_M_1M"
    size: 1_000_000

class SIFT_S(SIFT, BaseModel):
    name: str = "SIFT_S_500K"
    size: 500_000

class SIFT_M(SIFT, BaseModel):
    name: str = "SIFT_M_5M"
    size: 5_000_000

class SIFT_L(SIFT, BaseModel):
    name: str = "SIFT_L_50M"
    size: 50_000_000

SMALL = [GIST_S, Cohere_S, Glove_S, SIFT_S]
MEDIUM = [GIST_M, Cohere_M, Glove_M, SIFT_M]
LARGE = [Cohere_L, SIFT_L]
