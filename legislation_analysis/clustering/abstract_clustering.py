from abc import ABC, abstractmethod


class AbstractClustering(ABC):
    """
    Abstract class for the clustering methods.
    """

    @abstractmethod
    def _execute(self):
        pass

    @abstractmethod
    def execute(self):
        pass
