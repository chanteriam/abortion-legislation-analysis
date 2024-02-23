"""
Implements the TopicModeling class, which applies standard topic modeling
"""

import itertools
import os

import gensim
import numpy as np
import pandas as pd
from kneed import KneeLocator
from sklearn.feature_extraction.text import TfidfVectorizer

from legislation_analysis.utils.classes.visualizer import Visualizer
from legislation_analysis.utils.constants import (
    MAX_NUM_TOPICS,
    MIN_NUM_TOPICS,
    MODELED_DATA_PATH,
)
from legislation_analysis.utils.functions import load_file_to_df


class TopicModeling:
    """
    Applies standard topic modeling techniques to textual data.

    parameters:
        file_path (str): path to the file to apply topic modeling to.
        save_name (str): name of the file to save the topic modeling data to.
        testing (bool): whether to run in testing mode.
        column (str): column to apply topic modeling to.
        max_df (float): maximum document frequency for the tfidf vectorizer.
        min_df (int): minimum document frequency for the tfidf vectorizer.
        stop_words (str): stop words to use for the tfidf vectorizer.
    """

    def __init__(
        self,
        file_path: str,
        save_name: str,
        testing: bool = False,
        column: str = "cleaned_text",
        max_df: float = 0.8,
        min_df: int = 5,
        stop_words: str = "english",
    ):
        # file loading and saving
        self.df = load_file_to_df(file_path)
        self.save_name = save_name
        self.serial_path = os.path.join(
            MODELED_DATA_PATH, self.file_name.split(".")[0] + "_corpus.mm"
        )
        self.save_path = os.path.join(MODELED_DATA_PATH, self.save_name)

        # corpus generation
        self.tfidf_vectorizer = TfidfVectorizer(
            max_df=max_df, min_df=min_df, stop_words=stop_words
        )
        self.corpusmm = None
        self.tfidf = None
        self.dictionary = None
        self.corpus = None

        # visualization
        self.visualizer = Visualizer()

        # model building
        self.column = column
        self.testing = testing
        self.lda_model = None
        self.lda_output = None
        self.lda_topics_df = None
        self.coherence_scores = None

        # model parameters
        self.optimal_alpha = (
            self.optimal_eta
        ) = self.optimal_passes = self.optimal_topics = None

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

        # create corpus if not already done and serialize if the file does not
        # exist
        self.corpus = [
            self.dictionary.doc2bow(text) for text in self.df[column_reduced]
        ]
        gensim.corpora.MmCorpus.serialize(self.serial_path, self.corpus)
        self.corpusmm = (
            gensim.corpora.MmCorpus(self.serial_path)
            if not self.corpusmm
            else self.corpusmm
        )

    def lda(self, num_topics: int = 10, passes: int = 15) -> None:
        """
        Applies the LDA model to the corpus.

        parameters:
            num_topics (int): Number of topics to model.
            passes (int): Number of passes to model.
        """
        return gensim.models.LdaModel(
            self.corpusmm,
            num_topics=num_topics,
            id2word=self.dictionary,
            passes=passes,
            random_state=100,
        )

    def coherence(self, model: gensim.models.LdaModel) -> float:
        """
        Computes the coherence of the LDA model.

        parameters:
            model (gensim.models.LdaModel): LDA model to compute coherence for.
        """
        if f"{self.column}_reduced" not in self.df.columns:
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

    def grid_search(
        self,
        alpha_list,
        eta_list,
        passes_list,
        topic_range,
        visualize: bool = False,
    ) -> tuple:
        """
        Finds the optimal parameters for the LDA model by using grid search.

        parameters:
            alpha_list (list): List of alpha values to search.
            eta_list (list): List of eta values to search.
            passes_list (list): List of passes values to search.
            topic_range (range): Range of topics to search.
            visualize (bool): Whether to visualize the coherence scores.

        returns:
            (tuple) Optimal parameters for the LDA model.
        """
        results = []
        grid = list(
            itertools.product(alpha_list, eta_list, passes_list, topic_range)
        )

        for alpha, eta, passes, num_topics in grid:
            print(
                f"""Training LDA model with alpha={alpha},
                  eta={eta}, passes={passes}, num_topics={num_topics}"""
            )

            model = gensim.models.LdaModel(
                corpus=self.corpusmm,
                id2word=self.dictionary,
                num_topics=num_topics,
                alpha=alpha,
                eta=eta,
                passes=passes,
                random_state=100,
            )

            coherence = self.coherence(model)
            results.append((alpha, eta, passes, num_topics, coherence))

            if visualize:
                pass

        # Find the combination with the highest coherence score
        optimal_params = max(results, key=lambda x: x[-1])
        print("Optimal parameters:", optimal_params)

        # Optionally, set your class attributes to use the optimal parameters
        (
            self.optimal_alpha,
            self.optimal_eta,
            self.optimal_passes,
            self.optimal_topics,
        ) = optimal_params[:-1]
        self.lda_model = self.lda(
            self.optimal_topics,
            alpha=self.optimal_alpha,
            eta=self.optimal_eta,
            passes=self.optimal_passes,
        )

        return optimal_params

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
            list(range(MIN_NUM_TOPICS, MAX_NUM_TOPICS + 1)),
            coherence_scores,
            curve="convex",
            direction="increasing",
        )

        # visualize coherence scores
        print("\tVisualizing coherence scores...")
        if visualize:
            x = list(range(MIN_NUM_TOPICS, MAX_NUM_TOPICS + 1))
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
        self.coherence_scores = coherence_scores
        self.optimal_topics = kneedle.elbow
        self.lda_model = self.lda(self.optimal_topics)

    def find_optimal_parameters(self, visualize: bool = False):
        """
        Finds the optimal parameters for the LDA model by using either grid
        search or finding only the optimal number of topics with default
        parameters.

        parameters:
            visualize (bool): Whether to visualize the coherence scores.
        """
        # fine optimal parameters using grid search
        if not self.testing:
            self.grid_search(
                alpha_list=[0.01, 0.1, "symmetric", "asymmetric"],
                eta_list=[0.01, 0.1, "symmetric"],
                passes_list=[10, 20],
                topic_range=range(MIN_NUM_TOPICS, MAX_NUM_TOPICS + 1),
                visualize=visualize,
            )

        # find optimal number of topics with default parameters
        else:
            self.find_optimal_num_topics(visualize=visualize)

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

    def get_topic_df(self) -> None:
        """
        Generates a dataframe of topics and their words.
        """
        topics = self.get_topics()
        df_cols = [f"topic_{i}" for i in range(1, self.optimal_topics + 1)]
        df_dict = {col: [] for col in df_cols}
        for i, col in enumerate(df_cols):
            for word, _ in topics[i][1]:
                df_dict[col].append(word)

        self.lda_topics_df = pd.DataFrame(df_dict)

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
        self.find_optimal_parameters(visualize)

        print("Generating LDA model...")
        self.lda_output = self.lda_model.show_topics(
            num_topics=self.optimal_topics, num_words=10, formatted=False
        )

        self.get_topic_df()
