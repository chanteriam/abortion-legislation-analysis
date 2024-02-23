from abc import ABC, abstractmethod


class BaseTopicModeling(ABC):
    """
    Abstract class for topic modeling.
    """

    @abstractmethod
    def get_topics(self) -> None:
        pass

    @abstractmethod
    def lda(self, num_topics: int = 10) -> None:
        pass
