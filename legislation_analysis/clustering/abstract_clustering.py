import logging
from abc import ABC, abstractmethod

import numpy as np
import sklearn


class AbstractClustering(ABC):
    """
    Abstract class for the clustering methods.
    """

    @abstractmethod
    def cluster_parts_of_speech(self) -> None:
        pass

    @abstractmethod
    def visualize(self, tag: str) -> None:
        pass

    @staticmethod
    def cluster_scoring(df_column: str, labels: np.ndarray) -> None:
        logging.info("Cluster scoring:")
        logging.info(
            f"\tHomogeneity: "
            f"{sklearn.metrics.homogeneity_score(df_column, labels):0.3f}"
        )
        logging.info(
            f"\tCompleteness: "
            f"{sklearn.metrics.completeness_score(df_column, labels):0.3f}"
        )
        logging.info(
            f"\tV-measure: "
            f"{sklearn.metrics.v_measure_score(df_column, labels):0.3f}"
        )
        logging.info(
            "\tAdjusted Rand Score: "
            f"{sklearn.metrics.adjusted_rand_score(df_column, labels):0.3f}"
        )
