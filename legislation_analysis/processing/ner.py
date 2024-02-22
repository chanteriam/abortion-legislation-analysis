"""
Apply Named Entity Recognition (NER) to the text of the legislation.
"""

import logging
import os
from collections import defaultdict

import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from legislation_analysis.utils.constants import (
    NLP_MAX_CHAR_LENGTH,
    PROCESSED_DATA_PATH,
)
from legislation_analysis.utils.functions import load_file_to_df


model = SentenceTransformer("all-MiniLM-L6-v2")
nlp = spacy.load("en_core_web_sm")


class NER:
    """
    Class to apply Named Entity Recognition (NER) to the text of the
    legislation.

    parameters:
        file_path (str): path to the file to apply NER to.
        file_name (str): name of the file to save the NER data to.
    """

    def __init__(
        self,
        file_path: str,
        file_name: str,
    ):
        self.df = load_file_to_df(file_path)
        self.file_name = file_name
        self.ner_df = None
        self.save_path = os.path.join(PROCESSED_DATA_PATH, self.file_name)

    @staticmethod
    def apply_ner(text: str, ner_lst: list) -> list:
        """
        Applies Named Entity Recognition (NER) to a chunk of text.

        Parameters:
            text (str): Chunk of text to apply NER to.
            ner_dict (dict): Dictionary of Named Entity Recognition (NER) data.

        Returns:
            ner_lst (list): Updated NER data with entities from the chunk.
        """
        doc = nlp(text)
        for ent in doc.ents:
            ner_lst.append((ent.text, ent.label_))
        return ner_lst

    @staticmethod
    def __post_process(ner: list) -> list:
        """
        Post-processes Named Entity Recognition (NER) data to edit entity labels
        and remove unimportant entities.

        Parameters:
            ner (list): Named Entity Recognition (NER) data.

        Returns:
            (list): Post-processed Named Entity Recognition (NER) data.
        """
        new_ner = []

        # edit entity labels
        for _, (name, label) in enumerate(ner):
            if "ammendment" in name.lower() or "act" in name.lower():
                new_ner.append((name, "LAW"))
            elif "case" in name.lower() or "v." in name.lower():
                new_ner.append((name, "CASE"))
            else:
                new_ner.append((name, label))

        # remove unimportant entities
        important_labels = [
            "PERSON",
            "ORG",
            "GPE",
            "LAW",
            "DATE",
            "EVENT",
            "CASE",
        ]

        return [ent for ent in new_ner if ent[1] in important_labels]

    @classmethod
    def ner(cls, text: str) -> list:
        """
        Applies Named Entity Recognition (NER) to the text of the legislation,
        considering maximum character length.

        Parameters:
            text (str): Text to apply NER to.

        Returns:
            (list): Named Entity Recognition (NER) data.
        """
        ner = []

        if not text or str(text).lower() == "nan":
            return ner

        # initialize starting point for chunking
        start = 0
        text_length = len(text)

        while start < text_length:
            # if remaining text is shorter than max_chunk_size,
            # adjust end to length of text
            end = min(start + NLP_MAX_CHAR_LENGTH, text_length)

            # move the end to the nearest whitespace to avoid splitting words
            if end < text_length:
                while end > start and not text[end].isspace():
                    end -= 1

            # ensure we don't create an infinite loop
            if end == start:
                break

            chunk = text[start:end]
            ner = cls.apply_ner(chunk, ner)

            # move to the next chunk
            start = end

        return cls.__post_process(ner)

    @staticmethod
    def group_ent_by_label(entities: list) -> dict:
        """
        Groups entities by their label.

        Parameters:
            entities (list): List of Named Entity Recognition (NER) data.

        Returns:
            entities_by_label (dict): Dictionary of entities grouped by label.
        """
        entities_by_label = defaultdict(list)
        for name, label in entities:
            entities_by_label[label].append(name)
        return entities_by_label

    @classmethod
    def aggregate_ner(cls, entities: list) -> dict:
        """
        Aggregates Named Entity Recognition (NER) data.
        Conductions aggregation by embedding NER into sentence vectors,
        computing the cosine similarity between the vectors, and aggregating
        similar NERs using the longest entity name.

        Parameters:
            ner_lst (list): List of Named Entity Recognition (NER) data.

        Returns:
            ner_dict (dict): Aggregated Named Entity Recognition (NER) data.
        """
        # group entities by their label
        entities_by_label = cls.group_ent_by_label(entities)

        # generate Embeddings for Each Group
        embeddings_by_label = {}
        for label, names in entities_by_label.items():
            embeddings_by_label[label] = model.encode(names)

        # compute similarity and aggregate within groups
        threshold = 0.70
        canonical_entities = defaultdict(int)
        for label, embeddings in embeddings_by_label.items():
            similarity_matrix = cosine_similarity(embeddings)

            # flag to mark entities that have been aggregated to avoid double
            # counting
            aggregated = [False] * len(embeddings)

            for i in range(len(embeddings)):
                # skip already aggregated entities
                if aggregated[i]:
                    continue

                aggregated[i] = True

                # initialize aggregation with the current entity
                similar_entities = [entities_by_label[label][i]]

                for j in range(i + 1, len(embeddings)):
                    if similarity_matrix[i, j] > threshold:
                        similar_entities.append(entities_by_label[label][j])
                        aggregated[j] = True

                # choose the canonical name as the longest name among similar
                # entities
                canonical_name = max(similar_entities, key=len)
                canonical_entities[(canonical_name, label)] += len(
                    similar_entities
                )

        # convert to list of tuples
        canonical_entities_tuples = []
        for (name, label), count in canonical_entities.items():
            canonical_entities_tuples.append((name, label, str(count)))

        return canonical_entities_tuples

    def process(self, cols_to_ner: list = None) -> None:
        """
        Process the text of the legislation to apply Named Entity Recognition
        (NER).

        parameters:
            cols_to_ner (list): columns to apply NER to.
        """
        if cols_to_ner is None:
            cols_to_ner = [("cleaned_text", "cleaned_text_ner")]
        self.ner_df = self.df.copy()

        # apply NER to the text of the legislation
        for col in cols_to_ner:
            logging.debug(f"\tApplying NER to {col[0]}...")
            self.ner_df[col[1]] = self.ner_df[col[0]].apply(self.ner)

        # aggregate NER data
        for col in cols_to_ner:
            logging.debug(f"\tAggregating NER data for {col[0]}...")
            self.ner_df[f"{col[1]}_agg"] = self.ner_df[col[1]].apply(
                self.aggregate_ner
            )
