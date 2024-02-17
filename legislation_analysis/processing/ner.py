"""
Apply Named Entity Recognition (NER) to the text of the legislation.
"""

import logging
import os

import spacy

from legislation_analysis.utils.constants import PROCESSED_DATA_PATH
from legislation_analysis.utils.functions import load_file_to_df, save


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
        file_path,
        file_name,
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
            ner_dict (dict): Updated NER data with entities from the chunk.
        """
        doc = nlp(text)
        for ent in doc.ents:
            ner_lst.append((ent.text, ent.label_))
        return ner_lst

    @classmethod
    def ner(cls, text: str) -> dict:
        """
        Applies Named Entity Recognition (NER) to the text of the legislation,
        considering maximum character length.

        Parameters:
            text (str): Text to apply NER to.

        Returns:
            ner (dict): Named Entity Recognition (NER) data.
        """
        max_chunk_size = (
            999980  # Set slightly below spaCy max to account for whitespace
        )
        ner = []

        if not text or str(text).lower() == "nan":
            return ner

        # Initialize starting point for chunking
        start = 0
        text_length = len(text)

        while start < text_length:
            # If remaining text is shorter than max_chunk_size,
            # adjust end to length of text
            end = min(start + max_chunk_size, text_length)

            # Move the end to the nearest whitespace to avoid splitting words
            if end < text_length:
                while end > start and not text[end].isspace():
                    end -= 1

            # Ensure we don't create an infinite loop
            if end == start:
                break  # This should not happen, but it's a safety measure

            chunk = text[start:end]
            ner = cls.apply_ner(chunk, ner)

            start = end  # Move to the next chunk

        return ner

    def process(self, cols_to_ner=None) -> None:
        """
        Process the text of the legislation to apply Named Entity Recognition
        (NER).

        parameters:
            cols_to_ner (list): columns to apply NER to.
        """
        if cols_to_ner is None:
            cols_to_ner = [("cleaned_text", "cleaned_text_ner")]
        self.ner_df = self.df.copy()

        for col in cols_to_ner:
            logging.debug(f"\tApplying NER to {col[0]}...")
            self.ner_df[col[1]] = self.ner_df[col[0]].apply(self.ner)


def main() -> None:
    """
    Main function to apply Named Entity Recognition (NER) to the text of the
    legislation.
    """
    logging.info(
        """
        Applying Named Entity Recognition (NER) to the
        text of the legislation..."""
    )

    # Apply NER to congressional legislation
    logging.debug("Applying NER to congressional legislation...")
    congress_ner = NER(
        file_path=os.path.join(
            PROCESSED_DATA_PATH, "congress_legislation_tokenized.fea"
        ),
        file_name="congress_legislation_ner.fea",
    )
    congress_ner.process(
        cols_to_ner=[
            ("cleaned_text", "cleaned_text_ner"),
            ("cleaned_summary", "cleaned_summary_ner"),
        ]
    )
    save(congress_ner.ner_df, congress_ner.save_path)

    # Apply NER to SCOTUS opinions
    logging.debug("Applying NER to SCOTUS opinions...")
    scotus_ner = NER(
        file_path=os.path.join(
            PROCESSED_DATA_PATH, "scotus_cases_tokenized.fea"
        ),
        file_name="scotus_cases_ner.fea",
    )
    scotus_ner.process()
    save(scotus_ner.ner_df, scotus_ner.save_path)
