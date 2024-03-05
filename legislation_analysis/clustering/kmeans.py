import logging
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.cluster

from legislation_analysis.clustering.abstract_clustering import BaseClustering
from legislation_analysis.utils.constants import (
    PLOTTED_DATA_PATH,
    CLUSTERED_DATA_PATH,
    OPTIMAL_CONGRESS_CLUSTERS,
    OPTIMAL_SCOTUS_CLUSTERS,
)
from legislation_analysis.utils.functions import (
    load_file_to_df,
    save_df_to_file,
)


class KMeansClustering(BaseClustering):
    """
    Class for implementing K-Means clustering.
    """

    def __init__(self, file_path: str, file_name: str):
        self.__df = load_file_to_df(file_path)
        self.__n_clusters = (
            OPTIMAL_CONGRESS_CLUSTERS
            if "congress" in file_name
            else OPTIMAL_SCOTUS_CLUSTERS
        )
        self.__title_suffix = (
            "Congressional Legislation"
            if "congress" in file_name
            else "SCOTUS Decisions"
        )
        self._save_path = os.path.join(CLUSTERED_DATA_PATH, file_name)
        self.__vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(
            max_df=0.5,
            max_features=1000,
            min_df=3,
            stop_words="english",
            norm="l2",
        )

    def cluster_parts_of_speech(self) -> None:
        logging.debug("Starting K-Means clustering...")
        vectors = self.__vectorizer.fit_transform(
            self.__df["text_pos_tags_of_interest"]
        )
        cluster_algo = sklearn.cluster.KMeans(
            n_clusters=self.__n_clusters, init="k-means++"
        )
        cluster_algo.fit(vectors)
        self.__df["kmeans_clusters"] = cluster_algo.labels_
        logging.debug("Finished K-Means clustering...")
        logging.debug("Saving K-Means assignments...")
        save_df_to_file(self.__df, self._save_path)

    def visualize(self) -> None:
        # First, we need to fit_transform the text data again (or you could store the vectorized data from `cluster_parts_of_speech` to reuse here)
        vectors = self.__vectorizer.fit_transform(
            self.__df["text_pos_tags_of_interest"]
        ).toarray()
        cluster_labels = self.__df["kmeans_clusters"]

        # Compute the silhouette score to evaluate the quality of clusters
        silhouette_avg = sklearn.metrics.silhouette_score(vectors, cluster_labels)
        print(
            f"For n_clusters = {self.__n_clusters}, The average silhouette_score is : {silhouette_avg:.3f}"
        )

        # Compute the silhouette scores for each sample
        sample_silhouette_values = sklearn.metrics.silhouette_samples(
            vectors, cluster_labels
        )

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        ax1.set_xlim([-0.1, 1])
        ax1.set_ylim([0, len(vectors) + (self.__n_clusters + 1) * 10])
        y_lower = 10

        for i in range(self.__n_clusters):
            # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[
                cluster_labels == i
            ]
            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = plt.cm.nipy_spectral(float(i) / self.__n_clusters)
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the y-axis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        pca = sklearn.decomposition.PCA(n_components=2)
        reduced_data = pca.fit_transform(vectors)
        colors = plt.cm.nipy_spectral(cluster_labels.astype(float) / self.__n_clusters)
        ax2.scatter(
            reduced_data[:, 0],
            reduced_data[:, 1],
            marker=".",
            s=30,
            lw=0,
            alpha=0.7,
            c=colors,
            edgecolor="k",
        )

        # Labeling the clusters
        # Before performing PCA transformation, ensure that the aggregation 
        # results in strings
        agg_texts = (
            self.__df.groupby("kmeans_clusters")["text_pos_tags_of_interest"]
            .agg(lambda x: " ".join(map(str, x)))
            .tolist()
        )
        vectorized_centers = self.__vectorizer.transform(agg_texts).toarray()
        centers = pca.transform(vectorized_centers)
        ax2.scatter(
            centers[:, 0],
            centers[:, 1],
            marker="o",
            c="white",
            alpha=1,
            s=200,
            edgecolor="k",
        )

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(
            (
                "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
                % self.__n_clusters
            ),
            fontsize=14,
            fontweight="bold",
        )

        plt.savefig(
            os.path.join(PLOTTED_DATA_PATH, f"kmeans_{self.__title_suffix}.png")
        )
        plt.show()
