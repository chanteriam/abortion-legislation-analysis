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
    PLOTTED_DATA_PATH,
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
        self.__df = load_file_to_df(file_path)
        if "congress" in file_name:
            self.__n_clusters = OPTIMAL_CONGRESS_CLUSTERS
            self.__title_suffix = "Congressional Legislation"
        else:
            self.__n_clusters = OPTIMAL_SCOTUS_CLUSTERS
            self.__title_suffix = "SCOTUS Decisions"
        self.__save_path = os.path.join(CLUSTERED_DATA_PATH, file_name)
        # This vectorizer is configured so that a word cannot show up in more
        # than half the documents, must show up at least 3x, and the model can
        # only have a maximum of 1000 features.
        self.__vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(
            max_df=0.5,
            max_features=1000,
            min_df=3,
            stop_words="english",
            norm="l2",
        )

    def cluster_parts_of_speech(self) -> None:
        logging.debug("Starting Hierarchy Complete clustering...")
        vectors = self.__vectorizer.fit_transform(
            self.__df["text_pos_tags_of_interest"]
        )

        vectors.todense()
        vector_matrix = vectors * vectors.T
        vector_matrix.setdiag(0)

        linkage_matrix = scipy.cluster.hierarchy.complete(
            vector_matrix.toarray()
        )
        cluster_algo = scipy.cluster.hierarchy.fcluster(
            linkage_matrix, self.__n_clusters, "maxclust"
        )

        self.__df["hc_clusters"] = cluster_algo
        logging.debug("Finished Hierarchy Complete clustering...")
        logging.debug("Saving Hierarchy Complete assignments...")
        save_df_to_file(self.__df, self.__save_path)

    def visualize(self) -> None:
        plt.title(
            "Hierarchical Complete Clustering Dendrogram "
            f"of {self.__title_suffix}"
        )
        plt.xlabel("Cluster Size")
        vectors = self.__vectorizer.fit_transform(
            self.__df["text_pos_tags_of_interest"]
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

        plt.savefig(
            os.path.join(
                PLOTTED_DATA_PATH,
                f"hierarchy_complete_{self.__title_suffix}.png",
            )
        )
        plt.show()
