from abc import ABC, abstractmethod

import numpy as np
import sklearn


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

    @staticmethod
    def cluster_scoring(df_column: str, labels: np.ndarray) -> None:
        # Leaving this as print, since it's for the notebooks
        print("Cluster scoring:")
        print(
            f"\tHomogeneity: "
            f"{sklearn.metrics.homogeneity_score(df_column, labels):0.3f}"
        )
        print(
            f"\tCompleteness: "
            f"{sklearn.metrics.completeness_score(df_column, labels):0.3f}"
        )
        print(
            f"\tV-measure: "
            f"{sklearn.metrics.v_measure_score(df_column, labels):0.3f}"
        )
        print(
            "\tAdjusted Rand Score: "
            f"{sklearn.metrics.adjusted_rand_score(df_column, labels):0.3f}"
        )
