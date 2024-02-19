import pandas as pd
import sklearn

from legislation_analysis.clustering.abstract_clustering import (
    AbstractClustering,
)


class HierarchyComplete(AbstractClustering):
    """
    Class for implementing hierarchy ward clustering.
    """

    def __init__(self, df: pd.DataFrame, n_clusters=32):
        self.df = df
        self.n_cluster = n_clusters
        # This vectorizer is configured so that a word cannot show up in more
        # than half the documents, must show up at least 3x, and the model can
        # only have a maximum of 1000 features.
        self.tfidf_vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(
            max_df=0.5,
            max_features=1000,
            min_df=3,
            stop_words="english",
            norm="l2",
        )

    def execute(self):
        pass

    def visualize(self):
        pass
