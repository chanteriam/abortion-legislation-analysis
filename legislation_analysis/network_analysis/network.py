"""
Implements a class for generating network analysis between congressional data
and SCOTUS opinions.
"""

import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from legislation_analysis.utils.constants import (
    CONGRESS_DATA_FILE_NER_NAME,
    PLOTTED_DATA_PATH,
    PROCESSED_DATA_PATH,
    PROJECT_PATH,
    SCOTUS_ABBREVS,
    SCOTUS_DATA_FILE_NER_NAME,
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
        congress_file: str = os.path.join(
            PROCESSED_DATA_PATH, CONGRESS_DATA_FILE_NER_NAME
        ),
        scotus_file: str = os.path.join(
            PROCESSED_DATA_PATH, SCOTUS_DATA_FILE_NER_NAME
        ),
    ):
        self.congress_df = load_file_to_df(congress_file)
        self.scotus_df = load_file_to_df(scotus_file)
        self.law_network = nx.Graph()
        self.congress_network = nx.Graph()
        self.scotus_network = nx.Graph()

        # storing congress to congress references
        self.congress_refs = np.zeros(
            (len(self.congress_df), len(self.congress_df))
        )

        # storing congress to scotus references
        self.congress_scotus_refs = np.zeros(
            (len(self.congress_df), len(self.scotus_df))
        )

        # storing scotus to scotus references
        self.scotus_refs = np.zeros((len(self.scotus_df), len(self.scotus_df)))

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
            if _type in ["LAW", "CASE"]:
                laws.append((entity, count))
        return laws

    def pop_congr_regs(
        self,
        congress_law_vecs,
        scotus_law_vecs,
    ) -> None:
        """
        Populates congressional references to other congressional legislation
        and SCOTUS opinions.

        parameters:
            congress_law_vecs (np.array): vector representation of congress law
            names
            scotus_law_vecs (np.array): vector representation of scotus law
            names
        """

        # get both congress-to-congress and congress-to-scotus references
        for i, row in self.congress_df.iterrows():
            law_entities = row["law_references"]

            for entity, count in law_entities:
                # convert ref to vector
                entity_vector = self.sentence_model.encode([entity])

                # calc cosine similarities
                cong_similarities = cosine_similarity(
                    entity_vector, congress_law_vecs
                )[0]
                scotus_similarities = cosine_similarity(
                    entity_vector, scotus_law_vecs
                )[0]

                # update for congress similarities
                for j, similarity in enumerate(cong_similarities):
                    if j == i:  # don't include self references
                        continue
                    if similarity > 0.96:
                        self.congress_refs[i, j] += int(count)

                # update for scotus similarities
                for j, similarity in enumerate(scotus_similarities):
                    if similarity > 0.96:
                        self.congress_scotus_refs[i, j] += int(count)

    def pop_scotus_refs(self, scotus_law_vecs) -> None:
        """
        Populates SCOTUS references to other SCOTUS opinions.

        parameters:
            scotus_law_vecs (np.array): vector representation of scotus law
            names
        """
        # get scotus-to-scotus references
        for i, row in self.scotus_df.iterrows():
            law_entities = row["law_references"]

            for entity, count in law_entities:
                # convert ref to vector
                entity_vector = self.sentence_model.encode([entity])

                # calc cosine similarities
                scotus_similarities = cosine_similarity(
                    entity_vector, scotus_law_vecs
                )[0]

                # update for scotus similarities
                for j, similarity in enumerate(scotus_similarities):
                    if j == i:
                        continue
                    if similarity > 0.96:
                        self.scotus_refs[i, j] += int(count)

    def pop_ref_matrices(self, congress_law_vecs, scotus_law_vecs) -> None:
        """
        Populates reference matrices for congress and scotus data.

        parameters:
            congress_law_vecs (np.array): vector representation of congress law
            names
            scotus_law_vecs (np.array): vector representation of scotus law
            names
        """
        congr_law_names = self.congress_df["title"].tolist()
        scotus_law_names = self.scotus_df["title"].tolist()

        # populate congress references
        self.pop_congr_regs(
            congress_law_vecs,
            scotus_law_vecs,
            congr_law_names,
            scotus_law_names,
        )

        # populate scotus references
        self.pop_scotus_refs(scotus_law_vecs, scotus_law_names)

    def create_congress_network(self) -> None:
        """
        Creates a network of cross-references within congressional data.
        """
        # mapping index to congress legislation names
        index_to_name = dict(enumerate(self.congress_df["legislation_number"]))

        # adding nodes and edges for congress-to-congress references
        for i, row in enumerate(self.congress_refs):
            # add the reffering congress legislation as a node
            if any(reference_count > 1 for reference_count in row):
                source_name = index_to_name[i]
                self.congress_network.add_node(source_name, type="congress")

                # add the referenced congress legislation as a node
                for j, count in enumerate(row):
                    if count > 1:
                        target_name = index_to_name[j]
                        if target_name != source_name:  # avoid self-loops
                            self.congress_network.add_edge(
                                source_name, target_name, weight=count
                            )

    def create_scotus_network(self) -> None:
        """
        Creates a network of cross-references within SCOTUS data.
        """
        # mapping index to scotus legislation names
        index_to_name = dict(enumerate(self.scotus_df["title"]))

        # adding nodes and edges for scotus-to-scotus references
        for i, row in enumerate(self.scotus_refs):
            # add the reffering scotus opinion as a node
            source_name = index_to_name[i]
            self.scotus_network.add_node(source_name, type="scotus")

            # add the referenced scotus opinion as a node
            for j, count in enumerate(row):
                if count > 0:
                    target_name = index_to_name[j]
                    if target_name != source_name:
                        self.scotus_network.add_edge(
                            source_name, target_name, weight=count
                        )

    def create_congress_scotus_network(self) -> None:
        """
        Creates a network of congressional legislation references to SCOTUS
        opinions.
        """
        congr_index_to_name = dict(
            enumerate(self.congress_df["legislation_number"])
        )
        scotus_index_to_name = dict(enumerate(self.scotus_df["title"]))

        # each row is a congress legislation and each column is a
        # scotus opinion reference
        for i, row in enumerate(self.congress_scotus_refs):
            # add the reffering congress legislation as a node
            if any(reference_count > 0 for reference_count in row):
                source_name = congr_index_to_name[i]
                self.law_network.add_node(source_name, type="scotus")

                # add the referenced scotus opinion as a node
                for j, count in enumerate(row):
                    if count > 0:
                        target_name = scotus_index_to_name[j]
                        self.law_network.add_edge(
                            source_name, target_name, weight=count
                        )

    def plot_network(self) -> None:
        """
        Plots a network graph of congressional legislation references to SCOTUS
        opinions.
        """
        pos = nx.spring_layout(self.law_network, k=0.3, iterations=25)

        plt.figure(figsize=(18, 18))

        in_degrees = dict(self.law_network.in_degree())
        sizes = [
            (in_degrees[node] * 100) + 100 for node in self.law_network.nodes()
        ]

        nx.draw(
            self.law_network,
            pos,
            node_color="orange",
            node_size=sizes,
            font_size=20,
            edge_color="gray",
            alpha=0.6,
        )

        nx.draw_networkx_labels(self.law_network, pos)

        plt.title(
            """Network of Abortion-Related Congressional Legislation
            References to SCOTUS Opinions""",
            fontsize=20,
        )
        plt.savefig(os.path.join(PLOTTED_DATA_PATH, "congr_scotus_network.png"))
        plt.show()

    def heatmap_congress(self) -> None:
        """
        Plots a heatmap of congressional legislation cross-references.
        """
        adjacency_matrix_list = []

        # filter nodes with edges
        nodes_with_edges = [
            node
            for node in self.congress_network.nodes()
            if list(self.congress_network.edges(node))
        ]

        # iterate over each node (law) to build the matrix
        for source_node in nodes_with_edges:
            row = []
            for target_node in nodes_with_edges:
                if source_node == target_node:
                    row.append(0)  # no self-connections
                elif self.congress_network.has_edge(source_node, target_node):
                    # if there's an edge, append its weight
                    row.append(
                        self.congress_network.edges[source_node, target_node][
                            "weight"
                        ]
                    )
                else:
                    row.append(0)  # no edge exists
            adjacency_matrix_list.append(row)

        # convert the list to a NumPy array
        matrix = np.array(adjacency_matrix_list)

        # create a DataFrame for the heatmap
        heatmap_df = pd.DataFrame(
            matrix, columns=nodes_with_edges, index=nodes_with_edges
        )

        # plotting
        plt.figure(figsize=(14, 14))
        sns.heatmap(heatmap_df, cmap="Reds")
        plt.title("Heatmap of Abortion-Related Congressional Cross-References")
        plt.savefig(os.path.join(PLOTTED_DATA_PATH, "congress_heatmap.png"))
        plt.show()

    def heatmap_scotus(self) -> None:
        """
        Plots a heatmap of SCOTUS opinion cross-references.
        """
        PLOTTED_DATA_PATH = os.path.join(PROJECT_PATH, "data", "plots")
        adjacency_matrix_list = []
        nodes = self.scotus_network.nodes()

        # iterate over each node (law) to build the matrix
        for source_node in nodes:
            row = []
            for target_node in nodes:
                if source_node == target_node:
                    row.append(0)  # No self-connections
                elif self.scotus_network.has_edge(source_node, target_node):
                    # If there's an edge, append its weight
                    row.append(
                        self.scotus_network.edges[source_node, target_node][
                            "weight"
                        ]
                    )
                else:
                    row.append(0)  # No edge exists
            adjacency_matrix_list.append(row)

        # convert the list to a NumPy array
        matrix = np.array(adjacency_matrix_list)

        # create a DataFrame for the heatmap
        cols = [SCOTUS_ABBREVS[node] for node in list(nodes)]
        heatmap_df = pd.DataFrame(matrix, columns=cols, index=cols)

        # plotting
        plt.figure(figsize=(16, 12))
        sns.heatmap(heatmap_df, cmap="Reds")
        plt.title("Heatmap of Abortion-Related SCOTUS Opinion Cross-References")
        plt.savefig(os.path.join(PLOTTED_DATA_PATH, "scotus_heatmap.png"))
        plt.show()

    def process(self) -> None:
        """
        Processes the data to create networks and plots.
        """
        # extract LAW entities from congress data
        self.congress_df["law_references"] = self.congress_df[
            "cleaned_text_ner_agg"
        ].apply(self.extract_laws)

        self.scotus_df["law_references"] = self.scotus_df[
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

        # create networks
        self.create_congress_network()
        self.create_congress_scotus_network()
        self.create_scotus_network()

        # plot network
        self.plot_network()
        self.heatmap_scotus()
        self.heatmap_congress()
