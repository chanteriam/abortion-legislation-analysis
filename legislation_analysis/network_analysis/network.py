"""
Implements a class for generating network analysis between congressional data
and SCOTUS opinions.
"""

import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
import nltk
import numpy as np
import pandas as pd
import random
import scipy
import seaborn as sns
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from legislation_analysis.utils.constants import (
    CONGRESS_DATA_FILE_NER_NAME,
    SCOTUS_DATA_FILE_NER_NAME,
    PROCESSED_DATA_PATH,
    PLOTTED_DATA_PATH,
)
from legislation_analysis.utils.functions import load_file_to_df


class NetworkAnalysis:
    """
    Class to generate network analysis between congressional data and SCOTUS.

    parameters:
        congress_file (str): file path to congress data
        scotus_file (str): file path to scotus data
    """

    def __init__(
        self,
        congress_file: str = CONGRESS_DATA_FILE_NER_NAME,
        scotus_file: str = SCOTUS_DATA_FILE_NER_NAME,
    ):
        self.congress_df = load_file_to_df(congress_file)
        self.scotus_df = load_file_to_df(scotus_file)

        # storing congress to congress references
        self.congress_refs = np.empty((len(self.congress_df), len(self.congress_df)))

        # storing congress to scotus references
        self.scotus_refs = np.empty((len(self.congress_df), len(self.scotus_df)))

        # for embedding
        self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.nlp = spacy.load("en_core_web_sm")

    def extract_laws(self, agg_ner: list) -> list:
        """
        Extracts law references from text.

        parameters:
            agg_ner (list): list of named entity recognition results

        returns:
            list: list of law references
        """
        laws = []
        for entity, _type, count in agg_ner:
            if _type == "LAW":
                laws.append((entity, count))
        return laws

    def pop_ref_matrices(self, congress_law_vecs, scotus_law_vecs) -> None:
        """ """
        for i, row in self.congress_df.iterrows():
            law_entities = row["law_references"]

            for entity, count in law_entities:
                # convert ref to vector
                entity_vector = self.sentence_model.encode([entity])

                # calc cosine similarities
                cong_similarities = cosine_similarity(entity_vector, congress_law_vecs)[
                    0
                ]
                scotus_similarities = cosine_similarity(entity_vector, scotus_law_vecs)[
                    0
                ]

                # update for congress similarities
                for j, similarity in enumerate(cong_similarities):
                    if j == i:  # don't include self references
                        continue
                    if similarity > 0.96:
                        self.congress_refs.at[i, j] += int(count)

                # update for scotus similarities
                for j, similarity in enumerate(scotus_similarities):
                    if similarity > 0.96:
                        self.scotus_refs.at[i, j] += int(count)

    def create_network(self) -> None:
        """
        TODO: create network graph
        """
        pass

    def plot_network(self) -> None:
        """
        TODO: plot network graph
        """
        pass

    def heatmap(self) -> None:
        """
        TODO: plot heatmaps
        """
        pass

    def process(self):
        """ """
        # extract LAW entities from congress data
        self.congress_df["law_references"] = self.congress_df[
            "cleaned_text_ner_agg"
        ].apply(self.extract_laws)

        # convert congressional law names to vectors
        congress_law_names = self.congress_df["title"].tolist()
        congress_law_vecs = self.sentence_model.encode(congress_law_names)

        # convert scootus law names to vectors
        scotus_law_names = self.scotus_df["title"].tolist()
        scotus_law_vecs = self.sentence_model.encode(scotus_law_names)

        # populate reference matrices
        self.pop_ref_matrices(congress_law_vecs, scotus_law_vecs)
