"""
Processes legislation text by stemming/lemmatizing, removing stop words, and
removing punctuation. Also sentences the text.
"""

import os

import pandas as pd
import spacy

from legislation_analysis.utils.constants import (
    CONGRESS_DATA_FILE_CLEANED,
    SCOTUS_DATA_FILE_CLEANED,
    TOKENIZED_DATA_PATH,
)
from legislation_analysis.utils.functions import save


nlp = spacy.load("en_core_web_sm")


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

    @staticmethod
    def tokenize_and_normalize(text: str, extra_stop: list = None) -> dict:
        """
        Tokenizes and normalizes text into sentences and words.

        parameters:
            text (str): text to tokenize and normalize.
            extra_stop (list): additional stop words to remove.

        returns:
            processed (dict): tokenized and normalized text.
        """
        if extra_stop is None:
            extra_stop = []
        processed = {"sents": [], "words": [], "words_norm": []}

        # Use spaCy for sentence segmentation and initial tokenization
        doc = nlp(text.lower())
        sentences = list(doc.sents)
        processed["sents"] = [sent.text.strip() for sent in sentences]

        for token in doc:
            # Check if token meets criteria
            if (
                not token.is_stop
                and not token.is_punct
                and not token.like_num
                and token.text.strip()
            ):
                processed["words"].append(token.text)
                processed["words_norm"].append(token.lemma_)

        return processed

    def process(self, verbose=True, cols_to_tokenize=None):
        """
        Processes the legislation text.

        parameters:
            verbose (bool): whether to print updates.
            cols_to_tokenize (list): columns to tokenize.
        """
        if not cols_to_tokenize:
            cols_to_tokenize = [("cleaned_text", "tokenized_text")]

        for col, new_col in cols_to_tokenize:
            if verbose:
                print(f"\tTokenizing and normalizing {col}...")
            self.df[new_col] = (
                self.df[col]
                .dropna()
                .apply(lambda x: self.tokenize_and_normalize(x))
            )

            # Unpack processed text into separate columns
            self.df[f"{new_col}_sents"] = self.df[new_col].apply(
                lambda x: x["sents"]
            )
            self.df[f"{new_col}_words"] = self.df[new_col].apply(
                lambda x: x["words"]
            )
            self.df[f"{new_col}_words_norm"] = self.df[new_col].apply(
                lambda x: x["words_norm"]
            )

            # Drop the intermediate column
            self.df.drop(columns=[new_col], inplace=True)


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

    if verbose:
        print("Tokenizing Congress data...")
    congress_tokenizer.process(verbose)

    if verbose:
        print("Tokenizing SCOTUS data...")
    scotus_tokenizer.process(verbose)

    save(congress_tokenizer.cleaned_df, congress_tokenizer.save_path)
    save(scotus_tokenizer.cleaned_df, scotus_tokenizer.save_path)
