"""
Develops a data cleaner class for the abortion legislation content analysis
project.
"""

import os
import re

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
ITER_LIMIT = 4


class Cleaner:
    """
    Abstract class for a cleaner object.

    parameters:
        filepath (str): path to the legislation text csv.
        filename (str): name of file to save.
    """

    DICTIONARY = set(words.words())

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
                        w
                        for w in words
                        if len(w.strip(" ")) and len(w.strip("_"))
                    ]
                    for i, w in enumerate(combined_words):
                        if i == len(words) - 1:
                            new_split_text.append(w.strip(" ").strip("_"))
                        else:
                            new_split_text.append(w.strip(" ").strip("_") + ".")
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
    def spell_check(cls, text: str) -> str:
        """
        Uses the NLTK library to spell check the text and fix spelling errors.

        parameters:
            text (str): text to spell check.

        returns:
            text (str): spell checked text.
        """
        words = text.split(" ")
        new_words = []
        ignore = []

        # checks if a word exists in the dictionary and tries to fix it
        for idx, word in enumerate(words):
            if idx in ignore:
                continue

            added = False
            if (
                not wordnet.synsets(word.lower())
                and word.lower() not in cls.DICTIONARY
            ):
                new_word = word

                # check if a word is numeric
                if any(char.isdigit() for char in word):
                    new_words.append(word)
                    continue

                # a word has been split by a space - forward
                for j in range(idx + 1, len(words)):
                    # if the word is too far away, stop
                    if abs(idx - j) > ITER_LIMIT:
                        break
                    new_word += words[j]
                    if (
                        wordnet.synsets(new_word.lower())
                        or new_word.lower() in cls.DICTIONARY
                    ):
                        new_words.append(new_word)
                        ignore.extend(list(range(idx + 1, j + 1)))
                        added = True
                        break

                # a word has been split by a space - backward
                if not added:
                    new_word = word
                    for j in range(idx - 1, -1, -1):
                        # if the word is too far away, stop
                        if abs(idx - j) > ITER_LIMIT:
                            break
                        new_word = words[j] + new_word
                        if (
                            wordnet.synsets(new_word.lower())
                            or new_word.lower() in cls.DICTIONARY
                        ):
                            # remove the previous word from the list
                            new_words = new_words[:j]
                            new_words.append(new_word)
                            added = True
                            break

                # two words have been combined - both are words
                if not added:
                    for j in range(len(word)):
                        if (
                            wordnet.synsets(word[:j].lower())
                            or word[:j].lower() in cls.DICTIONARY
                        ) and (
                            wordnet.synsets(word[j:].lower())
                            or word[j:].lower() in cls.DICTIONARY
                        ):
                            new_words.append(word[:j])
                            new_words.append(word[j:])
                            added = True
                            break

                # a word was just misspelled
                if not added:
                    new_words.append(word)
            else:
                new_words.append(word)

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
