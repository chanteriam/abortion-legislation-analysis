import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn

from legislation_analysis.clustering.abstract_clustering import (
    AbstractClustering,
)


class HierarchyComplete(AbstractClustering):
    """
    Class for implementing hierarchy complete clustering.
    """

    def __init__(
        self, df: pd.DataFrame, df_column: str = "cleaned_text", n_clusters=32
    ):
        self._df = df
        self._n_clusters = n_clusters
        # This vectorizer is configured so that a word cannot show up in more
        # than half the documents, must show up at least 3x, and the model can
        # only have a maximum of 1000 features.
        self._vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(
            max_df=0.5,
            max_features=1000,
            min_df=3,
            stop_words="english",
            norm="l2",
        )
        self._vectors = self._vectorizer.fit_transform(self._df[df_column])

        self._vectors.todense()
        vector_matrix = self._vectors * self._vectors.T
        vector_matrix.setdiag(0)

        self._linkage_matrix = scipy.cluster.hierarchy.complete(
            vector_matrix.toarray()
        )
        self._cluster_algo = scipy.cluster.hierarchy.fcluster(
            self._linkage_matrix, self._n_clusters, "maxclust"
        )

    def get_labels(self) -> np.ndarray:
        return self._cluster_algo

    def visualize(self) -> None:
        plt.title("Hierarchical Complete Clustering Dendrogram")
        plt.xlabel("Cluster Size")
        scipy.cluster.hierarchy.dendrogram(
            self._linkage_matrix, p=5, truncate_mode="level"
        )
        plt.show()
