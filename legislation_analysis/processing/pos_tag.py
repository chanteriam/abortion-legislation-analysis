"""
Applies Part-of-Speech (POS) tagging to the text of the legislation.
"""

import logging
import os

import spacy

from legislation_analysis.utils.constants import (
    PROCESSED_DATA_PATH,
    NLP_MAX_CHAR_LENGTH,
)
from legislation_analysis.utils.functions import load_file_to_df, save


nlp = spacy.load("en_core_web_sm")


class POSTagger:
    """
    A class for applying Part-of-Speech (POS) tagging to the text of the
    legislation.

    parameters:
        file_path (str): path to the file to tokenize.
        file_name (str): name of the file to save the tokenized data to.
    """

    def __init__(self, file_path: str, file_name: str):
        self.df = load_file_to_df(file_path)
        self.pos_df = None
        self.file_path = file_path
        self.file_name = file_name
        self.save_path = os.path.join(PROCESSED_DATA_PATH, self.file_name)

    @staticmethod
    def tag_text(text: str) -> [str]:
        """
        Tags the text of the legislation.

        parameters:
            text (str): text to tag.
        """
        doc = nlp(text)
        return [(token.text, token.pos_) for token in doc]

    @classmethod
    def pos_tag(cls, text: str) -> [str]:
        """
        Applies Part-of-Speech (POS) tagging to the text of the legislation.

        parameters:
            text (str): text to apply POS tagging to.

        returns:
            tagged (list): text with POS tagging applied.
        """
        tagged = []

        if not text or str(text).lower() == "nan":
            return None

        # Check if text exceeds the max length and needs to be chunked
        if len(text) > NLP_MAX_CHAR_LENGTH:
            start = 0
            while start < len(text):
                # Determine the end index of the current chunk, trying not to
                # split words
                end = start + NLP_MAX_CHAR_LENGTH
                if end < len(text) and not text[end].isspace():
                    # Try to move the end index to the next space to avoid
                    # splitting a word
                    while end < len(text) and not text[end].isspace():
                        end += 1
                # Tag the current chunk
                chunk = text[start:end]
                tagged.extend(cls.tag_text(chunk))
                start = end
        else:
            # If text is within the limit, tag it directly
            tagged = cls.tag_text(text)

        return tagged

    @staticmethod
    def extract_tags_of_interest(tags: [str], tags_of_interest: [str]) -> [str]:
        """
        Extracts parts of speech of interest from the POS tagging.

        parameters:
            tags (list): POS tagging to extract parts of speech from.
            tags_of_interest (list): parts of speech to extract.

        returns:
            tags_of_interest (list): parts of speech of interest.
        """
        if not tags:
            return None

        # extract the tags
        interested_tags = [tag for tag in tags if tag[1] in tags_of_interest]

        # keep only the text
        return [tag[0] for tag in interested_tags]

    def process(
        self,
        cols_to_apply_pos: list = None,
        tags_of_interest: list = None,
    ) -> None:
        """
        Applies POS tagging to the text of the legislation and saves the results
        to a file.
        """
        if cols_to_apply_pos is None:
            cols_to_apply_pos = [("cleaned_text", "text_pos")]

        self.pos_df = self.df.copy()

        # extract all part of speach elements
        for col in cols_to_apply_pos:
            logging.debug(f"\tApplying POS tagging to {col[0]}...")
            self.pos_df[col[1]] = self.pos_df[col[0]].apply(self.pos_tag)

        # isolate parts of speech of interest
        if tags_of_interest:
            logging.debug("\tExtracting tags of interest...")

            for col in cols_to_apply_pos:
                logging.debug(f"\tExtracting tags of interest from {col[1]}...")
                col_name = f"{col[1]}_tags_of_interest"

                self.pos_df[col_name] = self.pos_df[col[1]].apply(
                    lambda x: self.extract_tags_of_interest(x, tags_of_interest)
                )


def main() -> None:
    """
    Applies POS tagging to the text of the legislation and saves the results to
    a file.
    """
    tags_of_interest = ["NOUN", "ADJ", "VERB", "ADV"]
    congress_pos = POSTagger(
        file_path=os.path.join(
            PROCESSED_DATA_PATH, "congress_legislation_tokenized.fea"
        ),
        file_name="congress_legislation_pos.fea",
    )
    scotus_pos = POSTagger(
        file_path=os.path.join(PROCESSED_DATA_PATH, "scotus_cases_tokenized.fea"),
        file_name="scotus_cases_pos.fea",
    )

    # Apply POS tagging to congressional legislation
    logging.debug("Applying POS tagging to congressional text...")
    congress_pos.process(
        cols_to_apply_pos=[
            ("cleaned_text", "text_pos"),
            ("cleaned_summary", "summary_pos"),
        ],
        tags_of_interest=tags_of_interest,
    )
    save(congress_pos.pos_df, congress_pos.save_path)

    # Apply POS tagging to SCOTUS opinions
    logging.debug("Applying POS tagging to SCOTUS text...")
    scotus_pos.process(tags_of_interest=tags_of_interest)
    save(scotus_pos.pos_df, scotus_pos.save_path)
