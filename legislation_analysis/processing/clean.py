"""
Develops a data cleaner class for the abortion legislation content analysis
project.
"""

import os
import re

import pandas as pd

from legislation_analysis.utils.constants import (
    CLEANED_DATA_PATH,
    CONGRESS_DATA_FILE,
    SCOTUS_DATA_FILE,
)
from legislation_analysis.utils.functions import save


class Cleaner:
    """
    Cleans legislation text.

    parameters:
        filepath (str): path to the legislation text csv.
        filename (str): name of file to save.
    """

    def __init__(
        self,
        file_path=CONGRESS_DATA_FILE,
        file_name="congress_legislation_cleaned.csv",
    ):
        self.df = pd.read_csv(file_path)
        self.file_name = file_name
        self.cleaned_df = None
        self.save_path = os.path.join(CLEANED_DATA_PATH, self.file_name)

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Cleans text.

        parameters:
            text (str): text to clean.

        returns:
            text (str): cleaned text.
        """

        if not isinstance(text, str):
            return "None"

        cleaned_text = re.sub(
            r"\n", " ", text
        )  # replace any new line characters with spaces

        # remove html tags
        cleaned_text = re.sub(r"\<.*?\>", "", cleaned_text)

        cleaned_text = re.sub(
            r"\([0-9]+\)", "", cleaned_text
        )  # parentheses with numbers

        cleaned_text = re.sub(
            r"[^A-Za-z0-9 \,\.\?\!\;\:\(\)\'\"-]+", "", cleaned_text
        ).strip()  # remove special characters

        cleaned_text = re.sub(
            r"[\-\_]+", " ", cleaned_text
        )  # remove underscores and hyphens

        # remove extra whitespace
        cleaned_text = re.sub(r"\s+", " ", cleaned_text)

        return cleaned_text

    def process(self, verbose: bool = True) -> None:
        """
        Processes the legislation text.

        parameters:
            verbose (bool): whether to print status updates.
        """

        if "raw_text" not in self.df.columns and "text" in self.df.columns:
            self.df.rename(columns={"text": "raw_text"}, inplace=True)

        # remove rows with null text values
        df = self.df.dropna(subset=["raw_text"]).copy()
        df["cleaned_text"] = df["raw_text"].copy()
        df["cleaned_summary"] = df["latest summary"]

        # clean summarys
        df["cleaned_summary"] = df["cleaned_summary"].apply(
            lambda x: self.clean_text(x)
        )

        # clean text
        df["cleaned_text"] = df["cleaned_text"].apply(
            lambda x: self.clean_text(x)
        )

        self.cleaned_df = df


def main(verbose: bool = True):
    """
    Runs data cleaner.

    parameters:
        verbose (bool): whether to print status updates.

    returns:
        True (bool): whether the data cleaner ran successfully.
    """
    congress_cleaner = Cleaner(
        CONGRESS_DATA_FILE, "congress_legislation_cleaned.csv"
    )
    scotus_cleaner = Cleaner(SCOTUS_DATA_FILE, "scotus_cases_cleaned.csv")

    congress_cleaner.process(verbose)
    scotus_cleaner.process(verbose)

    save(congress_cleaner.cleaned_df, congress_cleaner.save_path)
    save(scotus_cleaner.cleaned_df, scotus_cleaner.save_path)
