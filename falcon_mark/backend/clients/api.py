from abc import ABC, abstractmethod


class Client(ABC):
    @abstractmethod
    def load(self):
        """loading multiple fields"""
        pass

    @abstractmethod
    def index(self):
        """building indexes"""
        pass

    @abstractmethod
    def search(self, filters):
        """search with filters"""
        pass
