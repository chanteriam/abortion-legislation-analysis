"""
Develops a data cleaner class for the abortion legislation content analysis
project.
"""

import os
import re
from typing import Tuple

import nltk
import pandas as pd
from nltk.corpus import wordnet, words

from legislation_analysis.utils.constants import (
    CLEANED_DATA_PATH,
    CONGRESS_DATA_FILE,
    SCOTUS_DATA_FILE,
)
from legislation_analysis.utils.functions import save


nltk.download("wordnet")
nltk.download("words")


class Cleaner:
    """
    Abstract class for a cleaner object.

    parameters:
        filepath (str): path to the legislation text csv.
        filename (str): name of file to save.
    """

    DICTIONARY = set(words.words())
    ITER_LIMIT = 4

    def __init__(
        self,
        file_path=CONGRESS_DATA_FILE,
        file_name="congress_legislation_cleaned.csv",
    ):
        self.df = pd.read_csv(file_path)
        self.file_name = file_name
        self.cleaned_df = None
        self.save_path = os.path.join(CLEANED_DATA_PATH, self.file_name)

    @classmethod
    def is_word_or_in_dictionary(cls, test_word: str) -> bool:
        return (
            wordnet.synsets(test_word.lower())
            or test_word.lower() in cls.DICTIONARY
        )

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Cleans the text.

        parameters:
            text (str): text to clean.

        returns:
            text (str): cleaned text.
        """
        split_text = text.split(" ")
        new_split_text = []

        for word in split_text:
            new_words = []
            # check if the word is separated by new line and/or hyphen
            if "-" in word:
                if "\n" in word:
                    word = word.replace("\n", "")

                if not any(char.isdigit() for char in word):
                    # don't remove hyphens for words like "30-year-old"
                    word = word.replace("-", "")

            # check if two words are combined with a new line character
            if "\n" in word:
                new_words = word.split("\n")
            else:
                new_words.append(word)

            # check if words are combined with a period
            for new_word in new_words:
                if "." in new_word:
                    words = new_word.split(".")
                    # if words are combined with a period, retain all but the
                    # last period
                    combined_words = [
                        w for w in words if len(w.strip()) and len(w.strip("_"))
                    ]
                    for i, w in enumerate(combined_words):
                        if i == len(words) - 1:
                            new_split_text.append(w.strip().strip("_"))
                        else:
                            new_split_text.append(w.strip().strip("_") + ".")
                else:
                    new_split_text.append(new_word)

        # combined to form cleaned text
        cleaned_text = " ".join(new_split_text)

        # remove special characters
        cleaned_text = re.sub(
            r"[^a-zA-Z0-9\s\,\.\?\;\:\)\(\[\]\"\'\-]", "", cleaned_text
        )

        # remove excess whitespace
        cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()

        return Cleaner.spell_check(cleaned_text)

    @classmethod
    def is_valid_word(cls, word: str) -> bool:
        """
        Check if the word is valid based on wordnet or a custom dictionary.

        parameters:
            word (str): word to check.

        returns:
            bool: whether the word is valid.
        """
        return bool(wordnet.synsets(word)) or word in cls.DICTIONARY

    @classmethod
    def combine_with_surrounding(
        cls, word_list: [str], current_index: int
    ) -> Tuple:
        """
        Attempt to combine the current word with surrounding words within
        ITER_LIMIT.

        parameters:
            words (list): list of words.
            current_index (int): index of the current word.

        returns:
            combined_word (str): combined word.
            idx (int): index of the combined word.
        """
        for direction in [1, -1]:  # Forward and backward
            for j in range(1, cls.ITER_LIMIT + 1):
                idx = current_index + j * direction
                if 0 <= idx < len(word_list):
                    combined_word = (
                        word_list[current_index] + word_list[idx]
                        if direction == 1
                        else word_list[idx] + word_list[current_index]
                    )
                    if cls.is_valid_word(combined_word):
                        return combined_word, idx
        return None, None

    @classmethod
    def find_internal_split(cls, word: str) -> Tuple:
        """
        Find a valid internal split of the word, if any.

        parameters:
            word (str): word to split.

        returns:
            split_word1 (str): first split word.
            split_word2 (str): second split word.
        """
        for j in range(1, len(word)):
            if cls.is_valid_word(word[:j]) and cls.is_valid_word(word[j:]):
                return word[:j], word[j:]
        return None, None

    @classmethod
    def spell_check(cls, text: str) -> str:
        """
        Uses NLTK library and custom logic to fix spelling errors.

        parameters:
            text (str): text to spell check.

        returns:
            str: spell checked text.
        """
        words = text.split()
        new_words = []
        i = 0
        while i < len(words):
            word = words[i]
            if cls.is_valid_word(word) or any(char.isdigit() for char in word):
                new_words.append(word)
            else:
                combined_word, skip_idx = cls.combine_with_surrounding(words, i)
                if combined_word:
                    new_words.append(combined_word)
                    i = skip_idx  # Skip to the index of the word combined with
                else:
                    split_word1, split_word2 = cls.find_internal_split(word)
                    if split_word1:
                        new_words.extend([split_word1, split_word2])
                    else:
                        new_words.append(word)
            i += 1  # Move to the next word
        return " ".join(new_words)

    def process(
        self,
        verbose: bool = True,
        cols_to_clean=None,
    ) -> None:
        """
        Processes the legislation text.

        parameters:
            verbose (bool): whether to print status updates.
        """
        df = pd.DataFrame()

        if cols_to_clean is None:
            cols_to_clean = [("raw_text", "cleaned_text")]
        for col in cols_to_clean:
            if verbose:
                print(f"\tCleaning {col[0]}...")
            col, new_col = col
            df = self.df.dropna(subset=[col]).copy()
            df[new_col] = df[col].copy()

            # clean text
            df[new_col] = df[new_col].apply(lambda x: self.clean_text(x))

        self.cleaned_df = df


def main(verbose: bool = True) -> None:
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

    if verbose:
        print("Cleaning Congress Data...")
    congress_cleaner.process(
        verbose,
        cols_to_clean=[
            ("raw_text", "cleaned_text"),
            ("latest summary", "cleaned_summary"),
        ],
    )

    if verbose:
        print("Cleaning SCOTUS Data...")
    scotus_cleaner.process(verbose)

    save(congress_cleaner.cleaned_df, congress_cleaner.save_path)
    save(scotus_cleaner.cleaned_df, scotus_cleaner.save_path)
