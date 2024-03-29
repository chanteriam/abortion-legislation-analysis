import logging

from legislation_analysis.processing.clean import Cleaner
from legislation_analysis.processing.ner import NER
from legislation_analysis.processing.pos_tagger import POSTagger
from legislation_analysis.processing.tokenizer import Tokenizer
from legislation_analysis.utils.constants import (
    CONGRESS_DATA_CLEANED_FILE,
    CONGRESS_DATA_CLEANED_FILE_NAME,
    CONGRESS_DATA_FILE,
    CONGRESS_DATA_NER_FILE_NAME,
    CONGRESS_DATA_TOKENIZED_FILE_NAME,
    SCOTUS_DATA_CLEANED_FILE,
    SCOTUS_DATA_CLEANED_FILE_NAME,
    SCOTUS_DATA_FILE,
    SCOTUS_DATA_FILE_POS_TAGGED_NAME,
    SCOTUS_DATA_NER_FILE_NAME,
    SCOTUS_DATA_TOKENIZED_FILE_NAME,
)
from legislation_analysis.utils.functions import save_df_to_file


def run_data_cleaner() -> None:
    """
    Runs data cleaner.
    """
    # clean congressional legislation
    logging.info("Cleaning Congress Data...")
    congress_cleaner = Cleaner(
        CONGRESS_DATA_FILE, CONGRESS_DATA_CLEANED_FILE_NAME
    )
    scotus_cleaner = Cleaner(SCOTUS_DATA_FILE, SCOTUS_DATA_CLEANED_FILE_NAME)

    # clean congressional legislation
    logging.info("Cleaning Congress Data...")
    congress_cleaner.process(
        cols_to_clean=[
            ("raw_text", "cleaned_text"),
            ("raw_summary", "cleaned_summary"),
        ],
    )
    save_df_to_file(congress_cleaner.cleaned_df, congress_cleaner.save_path)

    # clean SCOTUS opinions
    logging.info("Cleaning SCOTUS Data...")
    scotus_cleaner = Cleaner(SCOTUS_DATA_FILE, SCOTUS_DATA_CLEANED_FILE_NAME)
    scotus_cleaner.process()
    save_df_to_file(scotus_cleaner.cleaned_df, scotus_cleaner.save_path)


def run_data_tokenizer() -> None:
    """
    Runs data tokenizer.
    """
    # tokenize congressional legislation
    logging.info("Tokenizing Congress Data...")
    congress_tokenizer = Tokenizer(
        CONGRESS_DATA_CLEANED_FILE, CONGRESS_DATA_TOKENIZED_FILE_NAME
    )

    # tokenize congressional legislation
    logging.info("Tokenizing Congress Data...")
    congress_tokenizer.process(
        cols_to_tokenize=[
            ("cleaned_text", "tokenized_text"),
            ("cleaned_summary", "tokenized_summary"),
        ]
    )
    save_df_to_file(
        congress_tokenizer.tokenized_df, congress_tokenizer.save_path
    )

    # tokenize SCOTUS opinions
    logging.info("Tokenizing SCOTUS Data...")
    scotus_tokenizer = Tokenizer(
        SCOTUS_DATA_CLEANED_FILE, SCOTUS_DATA_TOKENIZED_FILE_NAME
    )
    scotus_tokenizer.process()
    save_df_to_file(scotus_tokenizer.tokenized_df, scotus_tokenizer.save_path)


def run_pos_tagger() -> None:
    """
    Applies POS tagging to the text of the legislation and saves the results to
    a file.
    """
    tags_of_interest = ["NOUN", "ADJ", "VERB", "ADV"]

    # apply POS tagging to congressional legislation
    logging.info("Applying POS tagging to congressional text...")
    congress_pos = POSTagger(
        file_path=CONGRESS_DATA_CLEANED_FILE,
        file_name=SCOTUS_DATA_FILE_POS_TAGGED_NAME,
    )
    congress_pos.process(
        cols_to_apply_pos=[
            ("cleaned_text", "text_pos"),
            ("cleaned_summary", "summary_pos"),
        ],
        tags_of_interest=tags_of_interest,
    )
    save_df_to_file(congress_pos.pos_df, congress_pos.save_path)

    # apply POS tagging to SCOTUS opinions
    logging.info("Applying POS tagging to SCOTUS text...")
    scotus_pos = POSTagger(
        file_path=SCOTUS_DATA_CLEANED_FILE,
        file_name=SCOTUS_DATA_FILE_POS_TAGGED_NAME,
    )
    scotus_pos.process(tags_of_interest=tags_of_interest)
    save_df_to_file(scotus_pos.pos_df, scotus_pos.save_path)


def run_ner() -> None:
    """
    Main function to apply Named Entity Recognition (NER) to the text of the
    legislation.
    """
    logging.info(
        """
        Applying Named Entity Recognition (NER) to the
        text of the legislation..."""
    )

    # apply NER to congressional legislation
    logging.debug("Applying NER to congressional legislation...")
    congress_ner = NER(
        file_path=CONGRESS_DATA_CLEANED_FILE,
        file_name=CONGRESS_DATA_NER_FILE_NAME,
    )
    congress_ner.process(
        cols_to_ner=[
            ("cleaned_text", "cleaned_text_ner"),
            ("cleaned_summary", "cleaned_summary_ner"),
        ]
    )
    save_df_to_file(congress_ner.ner_df, congress_ner.save_path)

    # apply NER to SCOTUS opinions
    logging.debug("Applying NER to SCOTUS opinions...")
    scotus_ner = NER(
        file_path=SCOTUS_DATA_CLEANED_FILE,
        file_name=SCOTUS_DATA_NER_FILE_NAME,
    )
    scotus_ner.process()
    save_df_to_file(scotus_ner.ner_df, scotus_ner.save_path)
