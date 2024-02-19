import pandas as pd
import sklearn

from legislation_analysis.clustering.abstract_clustering import (
    AbstractClustering,
)


class HierarchyComplete(AbstractClustering):
    """
    Class for implementing hierarchy ward clustering.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        df_column: str = "cleaned_text",
        n_clusters: int = 32,
    ):
        self._df = df
        self.n_cluster = n_clusters
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

    def get_labels(self):
        pass

    def visualize(self):
        pass
