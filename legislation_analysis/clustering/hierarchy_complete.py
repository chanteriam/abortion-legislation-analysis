import logging
import os

import matplotlib.pyplot as plt
import scipy
import sklearn

from legislation_analysis.clustering.abstract_clustering import BaseClustering
from legislation_analysis.utils.constants import (
    CLUSTERED_DATA_PATH,
    OPTIMAL_CONGRESS_CLUSTERS,
    OPTIMAL_SCOTUS_CLUSTERS,
)
from legislation_analysis.utils.functions import (
    load_file_to_df,
    save_df_to_file,
)


class HierarchyComplete(BaseClustering):
    """
    Class for implementing hierarchy complete clustering.
    """

    def __init__(
        self,
        file_path: str,
        file_name: str,
    ):
        self._df = load_file_to_df(file_path)
        if "congress" in file_name:
            self._n_clusters = OPTIMAL_CONGRESS_CLUSTERS
            self._title_suffix = "Congressional Legislation"
        else:
            self._n_clusters = OPTIMAL_SCOTUS_CLUSTERS
            self._title_suffix = "SCOTUS Decisions"
        self._save_path = os.path.join(CLUSTERED_DATA_PATH, file_name)
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

    def cluster_parts_of_speech(self) -> None:
        logging.debug("Starting Hierarchy Complete clustering..")
        vectors = self._vectorizer.fit_transform(
            self._df["joined_text_pos_tags_of_interest"]
        )

        vectors.todense()
        vector_matrix = vectors * vectors.T
        vector_matrix.setdiag(0)

        linkage_matrix = scipy.cluster.hierarchy.complete(
            vector_matrix.toarray()
        )
        cluster_algo = scipy.cluster.hierarchy.fcluster(
            linkage_matrix, self._n_clusters, "maxclust"
        )

        self._df["hc_clusters"] = cluster_algo
        logging.debug("Finished Hierarchy Complete clustering.")
        logging.debug("Saving Hierarchy Complete assignments.")
        save_df_to_file(self._df, self._save_path)

    def visualize(self) -> None:
        plt.title(
            "Hierarchical Complete Clustering Dendrogram "
            f"of {self._title_suffix}"
        )
        plt.xlabel("Cluster Size")
        vectors = self._vectorizer.fit_transform(
            self._df["joined_text_pos_tags_of_interest"]
        )

        vectors.todense()
        vector_matrix = vectors * vectors.T
        vector_matrix.setdiag(0)

        linkage_matrix = scipy.cluster.hierarchy.complete(
            vector_matrix.toarray()
        )
        scipy.cluster.hierarchy.dendrogram(
            linkage_matrix, p=5, truncate_mode="level"
        )
        plt.show()
