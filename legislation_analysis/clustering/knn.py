import logging
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.cluster

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


class KNN(BaseClustering):
    """
    Class for implementing K-Nearest Neighbor clustering.
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
        logging.debug("Starting K-Nearest Neighbor clustering.")
        vectors = self._vectorizer.fit_transform(
            self._df["joined_text_pos_tags_of_interest"]
        )
        cluster_algo = sklearn.cluster.KMeans(
            n_clusters=self._n_clusters, init="k-means++"
        )
        cluster_algo.fit(vectors.toarray())
        self._df["knn_clusters"] = cluster_algo.labels_
        logging.debug("Finished K-Nearest Neighbor clustering.")
        logging.debug("Saving K-Nearest Neighbor assignments.")
        save_df_to_file(self._df, self._save_path)

    def visualize(self) -> None:
        vector_array = self._vectorizer.fit_transform(
            self._df["knn_clusters"]
        ).toarray()
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(15, 5))
        ax1.set_xlim([-0.1, 1])
        ax1.set_ylim([0, len(vector_array) + (self._n_clusters + 1) * 10])

        cluster_algo = sklearn.cluster.KMeans(
            n_clusters=self._n_clusters, init="k-means++"
        )
        cluster_labels = cluster_algo.fit_predict(vector_array)

        silhouette_avg = sklearn.metrics.silhouette_score(
            vector_array, cluster_labels
        )

        # Compute the silhouette scores for each sample
        sample_silhouette_values = sklearn.metrics.silhouette_samples(
            vector_array, cluster_labels
        )

        y_lower = 10

        pca = sklearn.decomposition.PCA(n_components=self._n_clusters).fit(
            vector_array
        )
        reduced_data = pca.transform(vector_array)

        for i in range(self._n_clusters):
            ith_cluster_silhouette_values = sample_silhouette_values[
                cluster_labels == i
            ]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            cmap = matplotlib.cm.get_cmap("nipy_spectral")
            color = cmap(float(i) / self._n_clusters)
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            y_lower = y_upper + 10

        ax1.set_title(f"Silhouette Plot for {self._title_suffix}")
        ax1.set_xlabel("Silhouette Coefficient Values")
        ax1.set_ylabel("Cluster labels")

        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        cmap = matplotlib.cm.get_cmap("nipy_spectral")
        colors = cmap(float(i) / self._n_clusters)
        ax2.scatter(
            reduced_data[:, 0],
            reduced_data[:, 1],
            marker=".",
            s=30,
            lw=0,
            alpha=0.7,
            c=colors,
        )

        # Labeling the clusters
        centers = cluster_algo.cluster_centers_
        projected_centers = pca.transform(centers)
        # Draw white circles at cluster centers
        ax2.scatter(
            projected_centers[:, 0],
            projected_centers[:, 1],
            marker="o",
            c="white",
            alpha=1,
            s=200,
        )

        for i, c in enumerate(projected_centers):
            ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50)

        ax2.set_title(f"PCA of {self._title_suffix} Clusters")
        ax2.set_xlabel("Principle Component 1")
        ax2.set_ylabel("Principle Component 2")

        plt.show()

        print(
            f"For n_clusters = {self._n_clusters}, The average "
            f"silhouette_score is : {silhouette_avg:.3f}"
        )
