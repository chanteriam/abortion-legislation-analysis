"""
Implements the TopicModeling class, which applies dynamic topic modeling to
congressional legislations.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldaseqmodel import LdaSeqModel
from scipy.stats import randint, uniform

from legislation_analysis.topic_modeling.abstract_topic_modeling import (
    BaseTopicModeling,
)
from legislation_analysis.utils.constants import (
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
            2,
            15,
        ),
        model=None,
    ) -> None:
        super().__init__(
            file_path=file_path,
            save_name=save_name,
            column=column,
            max_df=max_df,
            min_df=min_df,
            topic_ranges=topic_ranges,
            model=model,
        )

        # getting time series attributes
        self.bills_per_congress = None
        self.min_congress = self.df["congress_num"].min()
        self.max_congress = self.df["congress_num"].max()
        self.congressional_periods = list(
            range(self.min_congress, self.max_congress, PERIOD_GAP)
        )
        self.bills_per_congressional_period = [0] * NUM_BILL_PERIODS
        self.topics_by_period = {i: {} for i in range(NUM_BILL_PERIODS)}

        # model building
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
        self.df.sort_values(by=["congress_num"], inplace=True)
        self.df = self.df.reset_index(drop=True)

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

    def compute_ind_coherence(self, model, time, num_words=10) -> float:
        """
        Computes the coherence score for and individual time period.

        parameters:
            model (LdaSeqModel): The LDA model.
            time (int): The time period to consider.

        returns:
            (float) The coherence score.
        """
        topics = []
        for topic_idx in range(model.num_topics):
            topic_terms = sorted(
                model.print_topics(time=time)[topic_idx],
                key=lambda x: x[1],
                reverse=True,
            )
            top_terms = [term for term, _ in topic_terms[:num_words]]
            topics.append(top_terms)

        # compute coherence using extracted topics
        if time == 0:
            texts = self.df["filtered_tokens"][
                : self.bills_per_congressional_period[0]
            ]
        else:
            texts = self.df["filtered_tokens"][
                sum(self.bills_per_congressional_period[:time]) : sum(
                    self.bills_per_congressional_period[: time + 1]
                )
            ]
        coherence_model = CoherenceModel(
            topics=topics,
            texts=texts,
            dictionary=self.dictionary,
            coherence="u_mass",
        )
        return coherence_model.get_coherence()

    def compute_coherence(self, model) -> float:
        """
        Computes the average coherence score across time periods.
        """
        return np.mean(
            [
                self.compute_ind_coherence(model, time=i)
                for i in range(NUM_BILL_PERIODS)
            ]
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

            print(
                f"""\t(Iteration {_iter+1} of {iterations})
                Trying parameters: {params}"""
            )

            model = LdaSeqModel(
                corpus=self.corpus,
                id2word=self.dictionary,
                time_slice=self.bills_per_congressional_period,
                **params,
            )

            print("\t\tModel Trained")

            score = self.compute_coherence(model)

            print(f"\t\tCoherence Score: {score:.2f}")

            if score > best_score:
                best_score = score
                self.optimal_params = params
                self.lda_model = model

        print(f"Best Score: {best_score}")
        print(f"Best Params: {self.optimal_params}")

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
            for j in range(self.optimal_params["num_topics"]):
                self.topics_by_period[i][j] = sorted(
                    self.lda_model.print_topics(time=i)[j],
                    key=lambda x: x[0],
                )

    def gen_topic_model(self) -> None:
        """
        Generates the dynamic topic model.
        """
        self.get_bills_per_congress_period()
        self.prepare_corpus()

        if not self.lda_model:
            self.random_search()
        else:
            self.optimal_params = {
                "num_topics": self.lda_model.num_topics,
            }
