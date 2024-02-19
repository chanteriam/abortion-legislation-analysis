"""
Develops a data cleaner class for the abortion legislation content analysis
project.
"""

import logging
import os
import re

import nltk
import pandas as pd
from nltk.corpus import wordnet, words

from legislation_analysis.utils.constants import (
    CLEANED_DATA_PATH,
    CONGRESS_DATA_FILE,
)
from legislation_analysis.utils.functions import load_file_to_df


nltk.download("words")
nltk.download("wordnet")


class Cleaner:
    """
    Cleaner object class.

    parameters:
        file_path (str): path to the legislation text csv.
        file_name (str): name of file to save.
    """

    DICTIONARY = set(words.words())
    ITER_LIMIT = 4

    def __init__(
        self,
        file_path=CONGRESS_DATA_FILE,
        file_name="congress_legislation_cleaned.fea",
    ):
        self.df = load_file_to_df(file_path)
        self.cleaned_df = None
        self.save_path = os.path.join(CLEANED_DATA_PATH, file_name)

    @staticmethod
    def process_words(split_text: list) -> list:
        """
        Processes words.

        parameters:
            split_text (list): list of words to process.

        returns:
            words (list): processed words.
        """
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

        return new_split_text

    @classmethod
    def clean_text(cls, text: str) -> str:
        """
        Cleans the text.

        parameters:
            text (str): text to clean.

        returns:
            text (str): cleaned text.
        """
        if not text or pd.isna(text):
            return text

        split_text = text.split(" ")
        new_split_text = cls.process_words(split_text)

        # remove html tags
        new_split_text = [
            re.sub(r"<[^>]*>", "", word)
            for word in new_split_text
            if len(re.sub(r"<[^>]*>", "", word)) > 0
        ]

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
        Determines if a word is valid.

        parameters:
            word (str): word to check.

        returns:
            bool: whether the word is valid.
        """
        punctuation = [
            ",",
            ".",
            "!",
            "?",
            ";",
            ":",
            ")",
            "(",
            "[",
            "]",
            "{",
            "}",
            "\\",
            "/",
        ]

        for punc in punctuation:
            word = word.replace(punc, "")

        return bool(wordnet.synsets(word)) or (word.lower() in cls.DICTIONARY)

    @classmethod
    def combine_with_surrounding(cls, words: list, current_index: int) -> tuple:
        """
        Attempt to combine the current word with surrounding words within
        ITER_LIMIT.

        parameters:
            words (list): list of words.
            current_index (int): index of the current word.

        returns:
            combined_word (str): combined word.
            indexes (list): indices of the combined words.
            add_type (str): type of addition.
        """
        for direction in [1, -1]:  # Forward and backward
            combined_word = words[current_index]
            indexes = []
            add_type = None  # skip, pop

            # check surroundings
            for j in range(1, cls.ITER_LIMIT + 1):
                idx = current_index + j * direction
                indexes.append(idx)

                # if the word is too far away, stop
                if 0 <= idx < len(words):
                    combined_word = (
                        combined_word + words[idx]
                        if direction == 1
                        else words[idx] + combined_word
                    )

                    # check if the combined word is valid
                    if cls.is_valid_word(combined_word):
                        if direction == 1:
                            add_type = "skip"
                        else:
                            add_type = "pop"
                        return combined_word, indexes, add_type
        return None, None, None

    @classmethod
    def find_internal_split(cls, word: str) -> tuple:
        """
        Find a valid internal split of the word, if any.

        parameters:
            word (str): word to split.

        returns:
            word1 (str): first word.
            word2 (str): second word.
        """
        for j in range(1, len(word)):
            if cls.is_valid_word(word[:j]) and cls.is_valid_word(word[j:]):
                return word[:j], word[j:]
        return None, None

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
            # check if we should ignore the word
            if idx in ignore:
                continue

            # check if the word is a valid word or contains a number
            if cls.is_valid_word(word.lower()) or any(
                char.isdigit() for char in word
            ):
                new_words.append(word)
                continue

            # check if one word was split by a space
            combined_word, idxs, add_type = cls.combine_with_surrounding(
                words, idx
            )
            if combined_word:
                # for forward, we skip the next words
                if add_type == "skip":
                    ignore.extend(idxs)

                # for backward, we remove the previous words
                elif add_type == "pop":
                    new_words = new_words[: idxs[-1] + 1]
                new_words.append(combined_word)
                continue

            # check if two words were combined
            word1, word2 = cls.find_internal_split(word)
            if word1 and word2:
                new_words.append(word1)
                new_words.append(word2)
                continue

            # misspelled word
            new_words.append(word)

        return " ".join(new_words)

    def process(
        self,
        cols_to_clean=None,
    ) -> None:
        """
        Processes the legislation text.

        parameters:
            cols_to_clean (list): columns to clean.
        """
        cleaned_df = self.df.copy()

        for col in cols_to_clean:
            logging.debug(f"\tCleaning {col[0]}.")
            col, new_col = col
            cleaned_df[new_col] = cleaned_df[col]

            # clean text
            cleaned_df[new_col] = cleaned_df[new_col].apply(
                lambda x: self.clean_text(x)
            )

        self.cleaned_df = cleaned_df
