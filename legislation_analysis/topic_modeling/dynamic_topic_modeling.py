"""
Implements the TopicModeling class, which applies dynamic topic modeling to
congressional legislations.
"""

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
from gensim.models.ldaseqmodel import LdaSeqModel
from scipy.stats import randint, uniform

from legislation_analysis.topic_modeling.abstract_topic_modeling import (
    BaseTopicModeling,
)
from legislation_analysis.utils.constants import (
    MAX_NUM_TOPICS_CONGRESS,
    MIN_NUM_TOPICS_CONGRESS,
    MODELED_DATA_PATH,
    TOPIC_MODEL_TRAINING_ITERATIONS,
)


NUM_BILL_PERIODS = 6  # 18 congresses total; 6 periods of 3 congresses each
PERIOD_GAP = 3


class DynamicTopicModeling(BaseTopicModeling):
    """
    DynamicTopicModeling class for applying LDA dynamic topic modeling
    techniques to pre-tokenized congressional textual data.

    parameters:
        file_path (str): The file path to the pre-tokenized data.
        save_name (str): The name to save the model.
        topic_ranges (tuple[int, int]): The range of topics to consider.
        min_df (float): The minimum document frequency for the TfidfVectorizer.
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
    ) -> None:
        super().__init__(
            file_path=file_path,
            save_name=save_name,
            column=column,
            max_df=max_df,
            min_df=min_df,
            topic_ranges=topic_ranges,
        )

        # getting time series attributes
        self.bills_per_congress = None
        self.min_congress = self.df["congress_num"].min()
        self.max_congress = self.df["congress_num"].max()
        self.congressional_periods = list(
            range(self.min_congress, self.max_congress, PERIOD_GAP)
        )
        self.bills_per_congressional_period = [0] * NUM_BILL_PERIODS
        self.topics_by_period = {i: [] for i in range(NUM_BILL_PERIODS)}

        # model building
        self.lda_model = None
        self.optimal_params = {"num_topics": None, "chain_variance": None}

    def append_bill_to_period(self, congress_num: int) -> None:
        """
        Appends a bill to the appropriate period based on the congress number.
        """
        for i, period in enumerate(self.congressional_periods):
            if congress_num < self.congressional_periods[0]:
                break
            if congress_num <= period + PERIOD_GAP - 1:
                self.bills_per_congressional_period[i] += 1
                break

    def get_bills_per_congress_period(self) -> None:
        """
        Groups congressional legislation based on congress number.
        Optionall visualizes the number of legislations over time.
        """
        self.df["congress_num"].apply(self.append_bill_to_period)
        self.df.sort_values(by=["congress_num"], inplace=True).reset_index(
            drop=True
        )

    def get_bills_per_congress(self, visualize=False) -> None:
        """
        Groups the number of bills per congress.
        """
        self.bills_per_congress = (
            self.df.groupby("congress_num", as_index=False)
            .agg({"legislation_number": "count"})
            .sort_values(by=["congress_num"])
            .rename(columns={"legislation_number": "num_bills"})
        )

        if visualize:
            plt.title("Number of Bills per Congress")
            plt.xlabel("Congress Number")
            plt.ylabel("Number of Bills")
            plt.xticks(np.arange(self.min_congress, self.max_congress, 2))
            plt.plot(
                self.bills_per_congress["congress_num"],
                self.bills_per_congress["num_bills"],
            )

    def random_search(self, iterations=TOPIC_MODEL_TRAINING_ITERATIONS) -> None:
        """
        Performs random search for the optimal number of topics.
        """
        best_score = float("-inf")
        for _iter in range(iterations):
            params = {
                "num_topics": randint(
                    self.topic_ranges[0], self.topic_ranges[1]
                ).rvs(),
                "chain_variance": uniform(0.005, 0.05).rvs(),
            }

            logging.debug(
                f"""\t(Iteration {_iter+1} of {iterations})
                Trying parameters: {params}"""
            )

            model = LdaSeqModel(
                corpus=self.corpus,
                id2word=self.dictionary,
                time_slice=self.bills_per_congressional_period,
                **params,
            )
            score = self.compute_coherence(model)

            logging.debug(f"\t\tCoherence Score: {score:.2f}")

            if score > best_score:
                best_score = score
                self.optimal_params = params
                self.lda_model = model

        logging.debug(f"Best Score: {best_score}")
        logging.debug(f"Best Params: {self.optimal_params}")

        # save model
        self.lda_model.save(os.path.join(MODELED_DATA_PATH, self.save_name))

    def get_topics(self, num_words: int = 10) -> None:
        """
        Gets the topics for each period, specifically the top words and their
        probabilities.

        parameters:
            num_words (int): The number of words to display for each topic.
        """
        for i in range(NUM_BILL_PERIODS):
            self.topics_by_period[i] = sorted(
                self.lda_model.print_topics(num_words=num_words, time=i)[1],
                key=lambda x: x[0],
            )

    def gen_topic_model(self) -> None:
        """ """
        self.prepare_corpus()
        self.random_search()