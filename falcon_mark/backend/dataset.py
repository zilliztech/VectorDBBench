from pydantic import BaseModel

class DataSet(BaseModel):
    name:      str
    data_path: str
    size:      int

    def prepare(self, url: str) -> bool:
        """Download the dataset"""
        pass

    def batch(self):
        # yield
        pass

    def ground_truth(self):
        # yield
        pass
