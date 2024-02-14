"""
Processes legislation text by stemming/lemmatizing, removing stop words, and
removing punctuation. Also sentences the text.
"""

import os

import pandas as pd
import spacy
from nltk.text import Text
from nltk.tokenize import sent_tokenize, word_tokenize

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

    # Adapted from UChicago's SOCI40133 - Homework 2, Exercise 2
    @staticmethod
    def normalize_tokens(tokens: list, extra_stop: list = []) -> list:
        # We can use a generator here as we just need to iterate over it
        normalized = []
        if type(tokens) == list and len(tokens) == 1:
            tokens = tokens[0]

        if type(tokens) == list:
            tokens = " ".join([str(elem) for elem in tokens])

        doc = nlp(tokens.lower())

        # add the property of stop word to words considered as stop words
        if len(extra_stop) > 0:
            for stopword in extra_stop:
                lexeme = nlp.vocab[stopword]
                lexeme.is_stop = True

        for w in doc:
            # if it's not a stop word or punctuation mark, add it to our article
            if (
                w.text != "\n"
                and not w.is_stop
                and not w.is_punct
                and not w.like_num
                and len(w.text.strip()) > 0
            ):
                # we add the lematized version of the word
                normalized.append(str(w.lemma_))

        return normalized

    def process(self, verbose=True, cols_to_tokenize=None):
        """
        Processes the legislation text for tokenization.
        """
        if not cols_to_tokenize:
            cols_to_tokenize = [("cleaned_text", "tokenized_text")]

        for col in cols_to_tokenize:
            if verbose:
                print(f"\tCleaning {col[0]}...")
            col, new_col = col
            df = self.df.dropna(subset=[col]).copy()

            # sentence tokenizer
            df[new_col + "_sents"] = df[col].copy()
            df[new_col + "_sents"] = df[new_col + "_sents"].apply(
                lambda x: sent_tokenize(x)
            )

            # word tokenizer
            df[new_col + "_words"] = df[col].copy()
            df[new_col + "_words"] = df[new_col + "_words"].apply(
                lambda x: Text(word_tokenize(x))
            )

            # normalize tokens
            df[new_col + "_words_norm"] = df[new_col + "_words"].apply(
                lambda x: self.normalize_tokens(x)
            )

def main(verbose: bool = True) -> None:
    """
    Runs data tokenizer.
    """
    congress_tokenizer = Tokenizer(
        CONGRESS_DATA_FILE_CLEANED, "congress_legislation_tokenized.csv"
    )
    scotus_tokenizer = Tokenizer(SCOTUS_DATA_FILE_CLEANED, "scotus_cases_tokenized.csv")

    congress_tokenizer.process(verbose)
    scotus_tokenizer.process(verbose)

    save(congress_tokenizer.cleaned_df, congress_tokenizer.save_path)
    save(scotus_tokenizer.cleaned_df, scotus_tokenizer.save_path)
