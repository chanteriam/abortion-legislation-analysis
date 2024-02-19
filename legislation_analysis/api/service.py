import logging
import os

from legislation_analysis.api.congress import CongressAPI
from legislation_analysis.api.scotus import SCOTUSDataExtractor
from legislation_analysis.utils.constants import API_DATA_PATH, RAW_DATA_PATH
from legislation_analysis.utils.functions import save


def download_congress_data() -> None:
    """
    Iterates through legislation csv search results and extracts the text of
    each bill. Saves the results to a new csv titled file_name_text.csv.
    """
    logging.info("Downloading Congressional abortion legislation.")
    file_path = os.path.join(RAW_DATA_PATH, "congress_abortion_legislation.csv")
    file_name = os.path.basename(file_path).split(".")[0]

    cleaner = CongressAPI(file_path)
    cleaner.process()

    save(
        cleaner.processed_df,
        os.path.join(API_DATA_PATH, f"{file_name}_full-text.fea"),
    )


def download_scotus_data() -> None:
    """
    Processes SCOTUS abortion decisions, pulling text from pdf urls.
    """
    logging.info("Downloading SCOTUS abortion decisions.")
    scotus_api = SCOTUSDataExtractor()
    scotus_api.process()

    # save data
    scotus_api.df.to_csv(
        os.path.join(API_DATA_PATH, "scotus_cases_full-text.csv"), index=False
    )
