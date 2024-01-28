"""
Processes legislation text by stemming/lemmatizing, removing stop words, and removing punctuation.
Also sentences the text.
"""

import os
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
import spacy

# constants
from abortion_legislation_content_analysis.utils.constants import (
    TOKENIZED_DATA_PATH,
    CONGRESS_CLEANED_DATA_FILE,
    SCOTUS_CLEANED_DATA_FILE,
)

# functions
from abortion_legislation_content_analysis.utils.functions import save


class Tokenizer:
    """
    Tokenizes legislation text

    parameters:
        filepath (str): path to the legislation text csv.
        filename (str): name of file to save.
    """

    def __init__(
        self,
        filepath=CONGRESS_CLEANED_DATA_FILE,
        filename="congress_legislation_tokenized.csv",
    ):
        self.df = pd.read_csv(filepath)
        self.filename = filename
        self.tokenized_df = None
        self.savepath = os.path.join(TOKENIZED_DATA_PATH, self.filename)

    def process(self):
        """
        Processes the legislation text for tokenization.

        parameters:
            verbose (bool): whether to print status updates.
        """
        pass


def main(verbose=True):
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
    scotus_tokenizer = Tokenizer(SCOTUS_CLEANED_DATA_FILE, "scotus_cases_tokenized.csv")

    congress_tokenizer.process(verbose)
    scotus_tokenizer.process(verbose)

    save(congress_tokenizer.cleaned_df, congress_tokenizer.savepath)
    save(scotus_tokenizer.cleaned_df, scotus_tokenizer.savepath)

    return True