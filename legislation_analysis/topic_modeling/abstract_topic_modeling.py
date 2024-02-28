"""
Implements the parent topic modeling class.
"""

from abc import ABC, abstractmethod

import numpy as np
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from sklearn.feature_extraction.text import TfidfVectorizer

from legislation_analysis.utils.constants import TFIDF_FILTER_THRESHOLD
from legislation_analysis.utils.functions import load_file_to_df


class BaseTopicModeling(ABC):
    """
    Abstract class for topic modeling.
    """

    def __init__(
        self,
        file_path: str,
        save_name: str,
        column: str = "text_pos_tags_of_interest",
        max_df: float = 0.8,
        min_df: int = 5,
        topic_ranges: tuple = (2, 30),
        model_fp: str = None,
    ):
        self.df = load_file_to_df(file_path)
        self.save_name = save_name
        self.column = column

        # corpus building
        self.tfidf_vectorizer = TfidfVectorizer(
            max_df=max_df, min_df=min_df, stop_words="english"
        )
        self.dictionary = None
        self.corpus = None

        # model building
        self.lda_model = LdaModel.load(model_fp) if model_fp else None
        self.topic_ranges = topic_ranges

    def prepare_corpus(self):
        """
        Prepares the corpus and dictionary from the DataFrame column containing
        tokenized texts.
        """
        # convert list of pos tokens to strings for TF-IDF vectorization
        documents = self.df[self.column].apply(" ".join)
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)

        # get feature names to map back to tokens
        feature_names = np.array(self.tfidf_vectorizer.get_feature_names_out())

        # determine threshold for filtering
        high_tfidf_threshold = np.quantile(
            tfidf_matrix.toarray(), TFIDF_FILTER_THRESHOLD
        )

        # filter tokens based on TF-IDF scores
        filtered_tokens = []
        for doc_idx, _doc in enumerate(documents):
            # find indices of tokens with high TF-IDF score
            high_tfidf_indices = np.where(
                tfidf_matrix[doc_idx].toarray().flatten() > high_tfidf_threshold
            )[0]

            # keep tokens with high TF-IDF scores
            high_tfidf_tokens = feature_names[high_tfidf_indices]

            # filter original tokens based on high TF-IDF tokens
            filtered_doc_tokens = [
                token
                for token in self.df[self.column][doc_idx]
                if token in high_tfidf_tokens
            ]
            filtered_tokens.append(filtered_doc_tokens)

        # update the DataFrame with filtered tokens
        self.df["filtered_tokens"] = filtered_tokens

        # prepare the dictionary and corpus for LDA
        self.dictionary = Dictionary(self.df["filtered_tokens"])
        self.corpus = [
            self.dictionary.doc2bow(text) for text in self.df["filtered_tokens"]
        ]

    @abstractmethod
    def get_topics(self, num_words: int) -> None:
        pass

    @abstractmethod
    def compute_coherence(self, model) -> float:
        pass

    @abstractmethod
    def random_search(self, iterations: int) -> None:
        pass

    @abstractmethod
    def gen_topic_model(self) -> None:
        pass
