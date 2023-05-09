class DataSet:
    _name:      str = ""
    _data_path: str = ""
    _size:      int = 0

    def __init__(self):
        pass

    def prepare(self, url: str) -> bool:
        """Download the dataset"""
        pass

    def batch(self):
        # yield
        pass

    def ground_truth(self):
        # yield
        pass

    @property
    def size(self) -> int:
        """The data counts of this dataset"""
        return self._size
    
    @property
    def name(self) -> int:
        """The unique name of the dataset"""
        return self._name
