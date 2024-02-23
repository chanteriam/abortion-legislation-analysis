import logging
import os

import matplotlib.pyplot as plt
import scipy
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim
from kneed import KneeLocator
import numpy as np
import pandas as pd

from legislation_analysis.topic_modeling.abstract_topic_modeling import (
    BaseTopicModeling,
)
from legislation_analysis.utils.constants import (
    TOPIC_MODELING_PATH,
    MIN_NUM_TOPICS,
    MAX_NUM_TOPICS,
)
from legislation_analysis.utils.functions import (
    load_file_to_df,
)

from legislation_analysis.utils.classes.visualizer import Visualizer


class TopicModeling:
    """
    Applies standard topic modeling techniques to textual data.

    parameters:
        file_path (str): path to the file to apply topic modeling to.
        file_name (str): name of the file to save the topic modeling data to.
    """

    def __init__(self, file_path: str, file_name: str, column: str = "cleaned_text"):
        self.df = load_file_to_df(file_path)
        self.file_name = file_name
        self.serial_path = os.path.join(
            TOPIC_MODELING_PATH, self.file_name.split(".")[0] + "_corpus.mm"
        )
        self.save_path = os.path.join(TOPIC_MODELING_PATH, self.file_name)
        self.tfidf_vectorizer = TfidfVectorizer(
            max_df=0.9, min_df=len(self.df) // 4, stop_words="english"
        )
        self.visualizer = Visualizer()
        self.column = column

        self.corpusm = None
        self.tfidf = None
        self.dictionary = None
        self.corpus = None
        self.lda_model = None
        self.lda_output = None
        self.optimal_topics = None

    def get_corpus(self) -> None:
        """
        Generates and serializes the corpus.
        """
        # fit and transform with tfidf_vectorizer if not already done
        self.tfidf = (
            self.tfidf_vectorizer.fit_transform(self.df[self.column])
            if self.tfidf is None
            else self.tfidf
        )

        # check and reduce the column if not already done
        column_reduced = f"{self.column}_reduced"
        if column_reduced not in self.df.columns:
            self.df[column_reduced] = self.df[self.column].apply(
                lambda x: [
                    word
                    for word in (x.split() if isinstance(x, str) else x)
                    if word in self.tfidf_vectorizer.vocabulary_
                ]
            )

        # create the dictionary if not already done
        self.dictionary = (
            gensim.corpora.Dictionary(self.df[column_reduced])
            if not self.dictionary
            else self.dictionary
        )

        # create corpus if not already done and serialize if the file does not exist
        if not self.corpus:
            self.corpus = [
                self.dictionary.doc2bow(text) for text in self.df[column_reduced]
            ]
            if not os.path.exists(self.serial_path):
                gensim.corpora.MmCorpus.serialize(self.serial_path, self.corpus)
            self.corpusmm = (
                gensim.corpora.MmCorpus(self.serial_path)
                if not self.corpusmm
                else self.corpusmm
            )

    def lda(self, num_topics: int = 10) -> None:
        """
        Applies the LDA model to the corpus.

        parameters:
            num_topics (int): Number of topics to model.
        """
        return gensim.models.LdaModel(
            self.corpusmm,
            num_topics=num_topics,
            id2word=self.dictionary,
            passes=15,
            random_state=100,
        )

    def coherence(self, model: gensim.models.LdaModel) -> float:
        """
        Computes the coherence of the LDA model.

        parameters:
            model (gensim.models.LdaModel): LDA model to compute coherence for.
        """
        if not f"{self.column}_reduced" in self.df.columns:
            raise ValueError(
                """Column not Found: The column must be reduced before 
                computing coherence. Use get_corpus."""
            )
        coherence_model_lda = gensim.models.CoherenceModel(
            model=model,
            texts=self.df[f"{self.column}_reduced"],
            dictionary=self.dictionary,
            coherence="c_v",
        )
        return coherence_model_lda.get_coherence()

    def find_optimal_num_topics(self, visualize: bool = False) -> None:
        """
        Finds the optimal number of topics for the LDA model by computing LDA
        models with different numbers of topics and comparing coherence values.

        parameters:
            visualize (bool): Whether to visualize the coherence scores.
        """
        # optimal topics already found
        if self.optimal_topics:
            if not self.lda_model:
                self.lda_model = self.lda(self.optimal_topics)
            return

        # scores for different numbers of topics
        coherence_scores = []
        model_list = []
        for num_topics in range(MIN_NUM_TOPICS, MAX_NUM_TOPICS + 1):
            print(f"\tComputing LDA model with {num_topics} topics...")
            model = self.lda(num_topics)
            model_list.append(model)
            coherence_scores.append(self.coherence(model))

        # When the optimal number of topics is reached, there will be an
        # "elbow" in the coherence values. Kneed is used to find the elbow.
        print("\tCalculating elbow...")
        kneedle = KneeLocator(
            [i for i in range(MIN_NUM_TOPICS, MAX_NUM_TOPICS + 1)],
            coherence_scores,
            curve="convex",
            direction="increasing",
        )

        # visualize coherence scores
        print("\tVisualizing coherence scores...")
        if visualize:
            x = [i for i in range(MIN_NUM_TOPICS, MAX_NUM_TOPICS + 1)]
            self.visualizer.visualize(
                "line",
                x,
                coherence_scores,
                "Coherence Score by Number of Topics",
                "Number of Topics",
                "Coherence Score",
            )

        # set optimal number of topics
        print("\tSetting optimal number of topics...")
        self.optimal_topics = kneedle.elbow
        self.lda_model = self.lda(self.optimal_topics)

    def get_topics(self, num_words: int = 10) -> list:
        """
        Returns the topics generated by the LDA model.

        parameters:
            num_words (int): Number of words to return for each topic.

        returns:
            (list) Topics generated by the LDA model.
        """
        return self.lda_model.show_topics(
            num_topics=self.optimal_topics, num_words=num_words, formatted=False
        )

    def gen_topic_model(self, visualize: bool = False) -> None:
        """
        Generates the LDA model for the corpus.

        parameters:
            visualize (bool): Whether to visualize the coherence scores.
        """
        # process column
        if isinstance(
            self.df[self.column][0], (list, pd.core.series.Series, np.ndarray)
        ):
            self.df[self.column] = self.df[self.column].apply(
                lambda x: " ".join(list(x))
            )

        print("Getting corpus...")
        self.get_corpus()

        print("Finding optimal number of topics...")
        self.find_optimal_num_topics(visualize)

        print("Generating LDA model...")
        self.lda_output = self.lda_model.show_topics(
            num_topics=self.optimal_topics, num_words=10, formatted=False
        )
