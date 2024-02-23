"""
Implements the TopicModeling class, which applies standard topic modeling
"""

import os

import gensim
import numpy as np
import pandas as pd
from gensim.models.coherencemodel import CoherenceModel
from scipy.stats import loguniform, randint
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold

from legislation_analysis.utils.classes.visualizer import Visualizer
from legislation_analysis.utils.constants import (
    MAX_NUM_TOPICS_CONGRESS,
    MIN_NUM_TOPICS_CONGRESS,
    MODELED_DATA_PATH,
    TOPIC_MODEL_TRAINING_ITERATIONS,
)
from legislation_analysis.utils.functions import load_file_to_df


class TopicModeling:
    """
    Applies standard topic modeling techniques to textual data.

    parameters:
        file_path (str): path to the file to apply topic modeling to.
        save_name (str): name of the file to save the topic modeling data to.
        column (str): column to apply topic modeling to.
        max_df (float): maximum document frequency for the tfidf vectorizer.
        min_df (int): minimum document frequency for the tfidf vectorizer.
        stop_words (str): stop words to use for the tfidf vectorizer.
        topic_ranges (tuple): range of topics to test.
        k_folds (int): number of folds to use for cross-validation.
    """

    def __init__(
        self,
        file_path: str,
        save_name: str,
        column: str = "text_pos_tags_of_interest",
        max_df: float = 0.8,
        min_df: int = 5,
        topic_ranges: tuple = (
            MIN_NUM_TOPICS_CONGRESS,
            MAX_NUM_TOPICS_CONGRESS,
        ),
        k_folds: int = 5,
    ):
        # file loading and saving
        self.df = load_file_to_df(file_path)
        self.save_name = save_name
        self.serial_path = os.path.join(
            MODELED_DATA_PATH, self.save_name.split(".")[0] + "_corpus.mm"
        )
        self.save_path = os.path.join(MODELED_DATA_PATH, self.save_name)

        # corpus generation
        self.tfidf_vectorizer = TfidfVectorizer(
            max_df=max_df, min_df=min_df, stop_words="english"
        )
        self.corpusmm = None
        self.tfidf = None
        self.dictionary = None
        self.corpus = None

        # visualization
        self.visualizer = Visualizer()

        # model building
        self.column = column
        self.lda_model = None
        self.lda_output = None
        self.lda_topics_df = None
        self.coherence_scores = None
        self.kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

        # model parameters
        self.optimal_alpha = (
            self.optimal_eta
        ) = self.optimal_passes = self.optimal_topics = None
        self.topic_ranges = topic_ranges

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

    def cross_validated_coherence(self, params, texts, dictionary):
        """
        Computes cross-validated coherence score for given LDA parameters.

        parameters:
            params (dict): LDA parameters to test.
            texts (list): List of texts to evaluate.
            dictionary (gensim.corpora.Dictionary): Dictionary of the corpus.

        returns:
            (float) Average coherence score across all folds.
        """
        scores = []

        for train_idx, test_idx in self.kf.split(texts):
            # Split texts into training and validation sets
            train_texts = [texts[i] for i in train_idx]
            test_texts = [texts[i] for i in test_idx]

            # Convert texts to corpus format
            train_corpus = [dictionary.doc2bow(text) for text in train_texts]

            # Train LDA model on training set
            model = gensim.models.LdaModel(
                corpus=train_corpus, id2word=dictionary, **params
            )

            # Evaluate model coherence on validation set
            coherence_model_lda = CoherenceModel(
                model=model,
                texts=test_texts,
                dictionary=dictionary,
                coherence="c_v",
            )
            scores.append(coherence_model_lda.get_coherence())

        # Return the average coherence score across all folds
        return np.mean(scores)

    def random_search(
        self,
        dictionary,
        corpus,
        texts,
        iterations=TOPIC_MODEL_TRAINING_ITERATIONS,
    ):
        """
        Performs random search with cross-validation to find optimal LDA
        parameters.

        parameters:
            dictionary (gensim.corpora.Dictionary): Dictionary of the corpus.
            corpus (list): Corpus of the data.
            texts (list): Texts of the data.
            iterations (int): Number of iterations to perform.
        """
        best_score = float("-inf")
        best_params = None

        for _iter in range(iterations):
            # Sample parameters
            params = {
                "num_topics": randint(
                    self.topic_ranges[0], self.topic_ranges[1]
                ).rvs(),
                "alpha": loguniform(0.01, 1).rvs(),
                "eta": loguniform(0.01, 1).rvs(),
                "passes": randint(10, 50).rvs(),
            }

            print(
                f"""
                \t(Iteration {_iter + 1} of {iterations})
                Testing parameters: {params}..."""
            )

            # Compute cross-validated coherence
            score = self.cross_validated_coherence(params, texts, dictionary)

            if score > best_score:
                best_score = score
                best_params = params

        # After finding optimal parameters, retrain model on the full corpus
        self.lda_model = gensim.models.LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=best_params["num_topics"],
            alpha=best_params["alpha"],
            eta=best_params["eta"],
            passes=best_params["passes"],
        )

        print(
            f"Optimal Parameters: {best_params}, Coherence Score: {best_score}"
        )

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

    def gen_topic_model(self) -> None:
        """
        Generates the LDA model for the corpus.
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
        self.random_search()

        print("Getting topics...")
        self.get_topic_df()
