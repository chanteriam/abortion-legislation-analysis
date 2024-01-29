"""
Processes legislation text by stemming/lemmatizing, removing stop words, and
removing punctuation. Also sentences the text.
"""

import os

import pandas as pd

from legislation_analysis.utils.constants import (
    CONGRESS_DATA_FILE_CLEANED,
    SCOTUS_DATA_FILE_CLEANED,
    TOKENIZED_DATA_PATH,
)
from legislation_analysis.utils.functions import save


class Tokenizer:
    """
    Tokenizes legislation text.
    """

    def __init__(
        self,
        file_path: str = CONGRESS_DATA_FILE_CLEANED,
        file_name: str = "congress_legislation_tokenized.csv",
    ):
        self.df = pd.read_csv(file_path)
        self.file_name = file_name
        self.tokenized_df = None
        self.save_path = os.path.join(TOKENIZED_DATA_PATH, self.file_name)

    def process(self):
        """
        Processes the legislation text for tokenization.
        """
        pass


def main(verbose: bool = True) -> None:
    """
    Runs data tokenizer.
    """
    congress_tokenizer = Tokenizer(
        CONGRESS_DATA_FILE_CLEANED, "congress_legislation_tokenized.csv"
    )
    scotus_tokenizer = Tokenizer(
        SCOTUS_DATA_FILE_CLEANED, "scotus_cases_tokenized.csv"
    )

    congress_tokenizer.process(verbose)
    scotus_tokenizer.process(verbose)

    save(congress_tokenizer.cleaned_df, congress_tokenizer.save_path)
    save(scotus_tokenizer.cleaned_df, scotus_tokenizer.save_path)
