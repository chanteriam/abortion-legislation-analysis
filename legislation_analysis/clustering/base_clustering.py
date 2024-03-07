from abc import ABC, abstractmethod


class BaseClustering(ABC):
    """
    Abstract class for the clustering methods.
    """

    @abstractmethod
    def cluster_parts_of_speech(self) -> None:
        pass

    @abstractmethod
    def visualize(self) -> None:
        pass
