import logging
import os

from legislation_analysis.processing.clean import Cleaner
from legislation_analysis.processing.pos_tagger import POSTagger
from legislation_analysis.processing.tokenizer import Tokenizer
from legislation_analysis.utils.constants import (
    CONGRESS_DATA_FILE,
    CONGRESS_DATA_FILE_CLEANED,
    PROCESSED_DATA_PATH,
    SCOTUS_DATA_FILE,
    SCOTUS_DATA_FILE_CLEANED,
)
from legislation_analysis.utils.functions import save


def run_data_cleaner() -> None:
    """
    Runs data cleaner.
    """
    congress_cleaner = Cleaner(
        CONGRESS_DATA_FILE, "congress_legislation_cleaned.fea"
    )
    scotus_cleaner = Cleaner(SCOTUS_DATA_FILE, "scotus_cases_cleaned.fea")

    # Clean congressional legislation
    logging.debug("Cleaning Congress Data...")
    congress_cleaner.process(
        cols_to_clean=[
            ("raw_text", "cleaned_text"),
            ("latest summary", "cleaned_summary"),
        ],
    )
    save(congress_cleaner.cleaned_df, congress_cleaner.save_path)

    # Clean SCOTUS opinions
    logging.debug("Cleaning SCOTUS Data...")
    scotus_cleaner.process()
    save(scotus_cleaner.cleaned_df, scotus_cleaner.save_path)


def run_data_tokenizer() -> None:
    """
    Runs data tokenizer.
    """
    congress_tokenizer = Tokenizer(
        CONGRESS_DATA_FILE_CLEANED, "congress_legislation_tokenized.fea"
    )
    scotus_tokenizer = Tokenizer(
        SCOTUS_DATA_FILE_CLEANED, "scotus_cases_tokenized.fea"
    )

    # Tokenize congressional legislation
    logging.debug("Tokenizing Congress Data...")
    congress_tokenizer.process(
        cols_to_tokenize=[
            ("cleaned_text", "tokenized_text"),
            ("cleaned_summary", "tokenized_summary"),
        ]
    )
    save(congress_tokenizer.tokenized_df, congress_tokenizer.save_path)

    # Tokenize SCOTUS opinions
    logging.debug("Tokenizing SCOTUS Data...")
    scotus_tokenizer.process()
    save(scotus_tokenizer.tokenized_df, scotus_tokenizer.save_path)


def run_pos_tagger() -> None:
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
        file_path=os.path.join(
            PROCESSED_DATA_PATH, "scotus_cases_tokenized.fea"
        ),
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
