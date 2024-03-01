"""
Implements a class for generating network analysis between congressional data
and SCOTUS opinions.
"""


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
        congress_file: str = CONGRESS_DATA_FILE_NER_NAME,
        scotus_file: str = SCOTUS_DATA_FILE_NER_NAME,
    ):
        self.congress_df = load_file_to_df(congress_file)
        self.scotus_df = load_file_to_df(scotus_file)
        self.law_network = nx.Graph()
        self.congress_network = nx.Graph()

        # storing congress to congress references
        self.congress_refs = np.zeros(
            (len(self.congress_df), len(self.congress_df))
        )

        # storing congress to scotus references
        self.scotus_refs = np.zeros(
            (len(self.scotus_df), len(self.congress_df))
        )

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
                        print(
                            f"""
                            Entity: {entity}
                            Reference: {congr_law_names[j]}
                            Similarity: {similarity}
                              """
                        )
                        self.congress_refs[i, j] += int(count)

                # update for scotus similarities
                for j, similarity in enumerate(scotus_similarities):
                    if similarity > 0.96:
                        print(
                            f"""
                            Entity: {entity}
                            Reference: {scotus_law_names[j]}
                            Similarity: {similarity}
                              """
                        )
                        self.scotus_refs[i, j] += int(count)

    def create_congress_network(self):
        """
        Creates a network of cross-references within congressional data.
        """
        # mapping index to congress legislation names
        index_to_name = dict(enumerate(self.congress_df["legislation_number"]))

        # adding nodes and edges for congress-to-congress references
        for i, row in enumerate(self.congress_refs):
            source_name = index_to_name[i]
            self.congress_network.add_node(source_name, type="congress")

            for j, count in enumerate(row):
                if count > 0:
                    target_name = index_to_name[j]
                    if target_name != source_name:  # avoid self-loops
                        self.congress_network.add_edge(
                            source_name, target_name, weight=count
                        )

    def create_total_network(self):
        """
        Creates a network including both congressional data and SCOTUS opinions.
        """
        # mapping index to congress legislation names
        index_to_name = dict(enumerate(self.congress_df["legislation_number"]))

        # extend index mapping to include SCOTUS opinions
        offset = len(self.congress_df)
        index_to_name.update(
            {
                i + offset: title
                for i, title in enumerate(self.scotus_df["title"])
            }
        )

        # combine reference matrices for a total view
        total_refs = np.concatenate((self.congress_refs, self.scotus_refs))

        for i, name in index_to_name.items():
            self.law_network.add_node(
                name, type="congress" if i < offset else "scotus"
            )

            for j, count in enumerate(total_refs[i]):
                if count > 0:
                    target_name = index_to_name[j]
                    self.law_network.add_edge(name, target_name, weight=count)

    def plot_network(self) -> None:
        """
        Plots a network graph of congressional legislation references to other
        legislation and SCOTUS opinions.
        """
        # Assuming law_network is your graph
        pos = nx.spring_layout(
            self.law_network, k=0.3, iterations=25
        )  # Adjust k for spacing, iterations for layout accuracy

        plt.figure(figsize=(18, 18))  # Increase figure size

        nx.draw(
            self.law_network,
            pos,
            node_color="orange",
            node_size=1600,
            font_size=9,
            edge_color="gray",
            alpha=0.6,
        )

        nx.draw_networkx_labels(self.law_network, pos)

        plt.title(
            """Abortion-Related Congressional Legislation
            & SCOTUS Opinion Cross-References""",
            fontsize=20,
        )
        plt.show()

    def heatmap(self) -> None:
        """
        Plots a heatmap of congressional legislation cross-references.
        """
        adjacency_matrix_list = []

        # Step 1: Filter nodes with edges
        nodes_with_edges = [
            node
            for node in self.congress_network.nodes()
            if list(self.congress_network.edges(node))
        ]

        # Iterate over each node (law) to build the matrix
        for source_node in nodes_with_edges:
            row = []
            for target_node in nodes_with_edges:
                if source_node == target_node:
                    row.append(0)  # No self-connections
                elif self.congress_network.has_edge(source_node, target_node):
                    # If there's an edge, append its weight
                    row.append(
                        self.congress_network.edges[source_node, target_node][
                            "weight"
                        ]
                    )
                else:
                    row.append(0)  # No edge exists
            adjacency_matrix_list.append(row)

        # Convert the list to a NumPy array
        matrix = np.array(adjacency_matrix_list)

        # Create a DataFrame for the heatmap
        heatmap_df = pd.DataFrame(
            matrix, columns=nodes_with_edges, index=nodes_with_edges
        )

        # Plotting
        plt.figure(figsize=(14, 14))
        sns.heatmap(heatmap_df, cmap="Reds")
        plt.title("Heatmap of Abortion-Related Congressional Cross-References")
        plt.show()

    def process(self) -> None:
        """
        Processes the data to create networks and plots.
        """
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

        # create networks
        self.create_congress_network()
        self.create_total_network()

        # plot network
        self.plot_network()
        self.heatmap()
