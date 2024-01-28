"""
Processes legislation text by stemming/lemmatizing, removing stop words, and
removing punctuation. Also sentences the text.
"""

import os

import pandas as pd

from legislation_analysis.utils.constants import (
    CONGRESS_CLEANED_DATA_FILE,
    SCOTUS_CLEANED_DATA_FILE,
    TOKENIZED_DATA_PATH,
)
from legislation_analysis.utils.functions import save


class Tokenizer:
    """
    Tokenizes legislation text

    parameters:
        filepath (str): path to the legislation text csv.
        filename (str): name of file to save.
    """

    def __init__(
        self,
        file_path: str = CONGRESS_CLEANED_DATA_FILE,
        file_name: str = "congress_legislation_tokenized.csv",
    ):
        self.df = pd.read_csv(file_path)
        self.file_name = file_name
        self.tokenized_df = None
        self.save_path = os.path.join(TOKENIZED_DATA_PATH, self.file_name)

    def process(self):
        """
        Processes the legislation text for tokenization.

        parameters:
            verbose (bool): whether to print status updates.
        """
        pass


def main(verbose: bool = True) -> None:
    """
    Runs data tokenizer.

    parameters:
        verbose (bool): whether to print status updates.

    returns:
        True (bool): whether the data cleaner ran successfully.
    """
    congress_tokenizer = Tokenizer(
        CONGRESS_CLEANED_DATA_FILE, "congress_legislation_tokenized.csv"
    )
    scotus_tokenizer = Tokenizer(
        SCOTUS_CLEANED_DATA_FILE, "scotus_cases_tokenized.csv"
    )

    congress_tokenizer.process(verbose)
    scotus_tokenizer.process(verbose)

    save(congress_tokenizer.cleaned_df, congress_tokenizer.save_path)
    save(scotus_tokenizer.cleaned_df, scotus_tokenizer.save_path)
