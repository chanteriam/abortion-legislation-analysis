"""
Apply Named Entity Recognition (NER) to the text of the legislation.
"""

import logging
import os

import spacy

from legislation_analysis.utils.constants import (
    PROCESSED_DATA_PATH,
    TOKENIZED_DATA_PATH,
)
from legislation_analysis.utils.functions import load_file_to_df, save


nlp = spacy.load("en_core_web_sm")


class NER:
    """
    Class to apply Named Entity Recognition (NER) to the text of the legislation.

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
        self.save_path = os.path.join(TOKENIZED_DATA_PATH, self.file_name)

    @staticmethod
    def apply_ner(text: str) -> dict:
        """
        Applies Named Entity Recognition (NER) to the text of the legislation.

        parameters:
            text (str): text to apply NER to.

        returns:
            ner (dict): Named Entity Recognition (NER) data.
        """
        ner = {"ents": []}

        # Use spaCy for NER
        doc = nlp(text)
        for ent in doc.ents:
            ner["ents"].append(
                {
                    "text": ent.text,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "label": ent.label_,
                }
            )

        return ner

    def process(self, cols_to_ner: list = ["cleaned_text"]):
        """
        Process the text of the legislation to apply Named Entity Recognition (NER).

        parameters:
            cols_to_ner (list): columns to apply NER to.
        """
        self.ner_df = self.df.copy()

        for col in cols_to_ner:
            logging.debug(f"\tApplying NER to {col}...")
            new_col = f"{col}_ner"
            self.ner_df[new_col] = self.ner_df[col].apply(self.apply_ner)


def main():
    """
    Main function to apply Named Entity Recognition (NER) to the text of the legislation.
    """
    logging.info(
        "Applying Named Entity Recognition (NER) to the text of the legislation..."
    )

    # Apply NER to congressional legislation
    congress_ner = NER(
        file_path=os.path.join(
            PROCESSED_DATA_PATH, "congress_legislation_tokenized.fea"
        ),
        file_name="congress_legislation_ner.fea",
    )

    # Apply NER to congressional legislation
    logging.debug("Applying NER to congressional legislation...")
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
        file_path=os.path.join(PROCESSED_DATA_PATH, "scotus_cases_tokenized.fea"),
        file_name="scotus_cases_ner.fea",
    )
    scotus_ner.process()
    save(scotus_ner.ner_df, scotus_ner.save_path)
