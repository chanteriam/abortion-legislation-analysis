"""
Script for pulling legislation text or pdfs from api.congress.gov.
Search results derived from https://www.congress.gov/advanced-search/legislation
using the following search terms:
    - abortion
    - reproduction
    - reproductive health
"""

import logging
import os
import re
import time
import urllib.parse
from typing import Optional

import bs4
import numpy as np
import pandas as pd
import requests

from legislation_analysis.utils.constants import (
    CONGRESS_API_KEY,
    CONGRESS_API_ROOT_URL,
    CONGRESS_COLUMNS_API,
    CONGRESS_ROOT_URL,
)
from legislation_analysis.utils.functions import (
    extract_pdf_text,
    load_file_to_df,
)


class CongressAPI:
    """
    Pulls legislation text from api.congress.gov.

    parameters:
        file_path (str): file_path to csv of legislation search results.
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.file_name = os.path.basename(file_path).split(".")[0]
        self.processed_df = None
        self.df = None

    def get_df(self) -> None:
        """
        Sets self.df by extracting data from the file path and reformatting
        columns.
        """
        # load data
        df = load_file_to_df(self.file_path)

        # get header row
        first_col = df.iloc[:, 0]
        header_row = list(first_col).index("Legislation Number")
        col_row = list(df.iloc[header_row, 0:]).index(np.nan)

        # reformat columns
        df = df.iloc[:, :col_row].copy()
        df.columns = list(df.iloc[header_row, :col_row])
        self.df = df.iloc[header_row + 1 :, :].reset_index(drop=True).copy()
        self.df.columns = [c.lower() for c in list(self.df.columns)]

    @staticmethod
    def extract_legislation_details(df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts the congress number, bill type, and bill number for each piece
        of legislation.

        parameters:
            df (pd.DataFrame): dataframe of legislation search results.

        returns:
            df (pd.DataFrame): dataframe of legislation search results with
            congress number, bill type, and bill number extracted.
        """
        new_df = df.copy()
        new_df.loc[:, "congress_num"] = new_df.loc[:, "congress"].apply(
            lambda x: x.split(" ")[0][:-2]
        )
        new_df = new_df.loc[new_df.loc[:, "congress_num"] != "", :]

        new_df.loc[:, "bill_type"] = new_df.loc[:, "legislation number"].apply(
            lambda x: x.split(" ")[0].lower().replace(".", "")
        )
        new_df.loc[:, "bill_num"] = new_df.loc[:, "legislation number"].apply(
            lambda x: int(x.split(" ")[1])
        )

        return new_df

    @staticmethod
    def get_api_url(congress_num: int, bill_type: str, bill_num: int) -> str:
        """
        Builds the api url for a given piece of legislation.

        parameters:
            congress_num (int): congress number.
            bill_type (str): bill type.
            bill_num (int): bill number.

        returns:
            url (str): api url for legislation.
        """

        url = urllib.parse.urljoin(
            CONGRESS_API_ROOT_URL,
            f"{congress_num}/{bill_type}/{bill_num}/text?"
            f"format=xml&api_key={CONGRESS_API_KEY}",
        )
        return url

    @staticmethod
    def extract_text_url(api_url: str) -> Optional[str]:
        """
        Extracts the url for the text of a given piece of legislation.

        parameters:
            api_url (str): api url for legislation.

        returns:
            text_url (str): url for the text of the legislation.
        """
        assert api_url.startswith(
            CONGRESS_API_ROOT_URL
        ), "URL does not start with legislation root url."

        logging.debug(f"\tExtracting text url from {api_url}...")

        time.sleep(3.6)  # to prevent overloading the api
        request = requests.get(api_url)
        soup = bs4.BeautifulSoup(request.text, features="xml")

        urls = soup.find_all("url")

        if len(urls) > 0:
            text_url = str(urls[0])
            text_url = re.sub(r"\</?url\>", "", text_url).strip()
            return text_url

        return None

    @staticmethod
    def build_text_url(congress_num: int, bill_type: str, bill_num: int) -> str:
        """
        Builds the url for the text of a given piece of legislation.

        parameters:
            congress_num (int): congress number.
            bill_type (str): bill type.
            bill_num (int): bill number.

        returns:
            text_url (str): url for the text of the legislation.
        """
        url = urllib.parse.urljoin(
            CONGRESS_ROOT_URL,
            f"{congress_num}/bills/{bill_type}{bill_num}/BILLS-"
            f"{congress_num}{bill_type}{bill_num}is.xml",
        )
        return url

    @staticmethod
    def extract_html_text(text_url: str) -> str:
        """
        Extracts the text of a given piece of legislation.

        parameters:
            text_url (str): url for the text of the legislation.

        returns:
            text (str): text of the legislation.
        """
        logging.debug(f"\tExtracting text from {text_url}...")

        request = requests.get(text_url)
        soup = bs4.BeautifulSoup(request.text, "html.parser")

        return soup.text

    def extract_text(self) -> None:
        """
        Extracts either html or pdf text for each legislation.
        """
        # extract html text
        self.processed_df.loc[
            (self.processed_df.loc[:, "text_url"].str[-3:] == "htm"),
            "text",
        ] = self.processed_df.loc[
            (self.processed_df.loc[:, "text_url"].str[-3:] == "htm"),
            "text_url",
        ].apply(
            lambda x: self.extract_html_text(x)
        )

        # extract pdf text
        self.processed_df.loc[
            (self.processed_df.loc[:, "text_url"].str[-3:] == "pdf"),
            "text",
        ] = self.processed_df.loc[
            (self.processed_df.loc[:, "text_url"].str[-3:] == "pdf"),
            "text_url",
        ].apply(
            lambda x: extract_pdf_text(x)
        )

    def post_process(self) -> None:
        """
        Post-processing dataframe, including dropping null values and renaming
        columns.
        """
        # remove rows with null text values
        self.processed_df = self.processed_df.loc[
            ~self.processed_df.loc[:, "text"].isna(), :
        ].copy()
        self.processed_df.rename(columns={"text": "raw_text"}, inplace=True)

        # keep relevant columns
        self.processed_df.rename(
            columns={
                "legislation number": "legislation_number",
                "latest summary": "raw_summary",
            },
            inplace=True,
        )
        self.processed_df = self.processed_df.loc[
            :, CONGRESS_COLUMNS_API
        ].copy()

    def process(self) -> None:
        """
        Processing function for extracting text from legislation search results.
        """
        # load data
        logging.debug(f"Loading data from {self.file_path}...")
        self.get_df()

        # extract legislation information
        logging.debug("Extracting legislation information...")
        self.processed_df = self.extract_legislation_details(self.df)

        # extract legislation api url
        logging.debug("Extracting legislation api url...")

        self.processed_df.loc[:, "api_url"] = self.processed_df.apply(
            lambda x: self.get_api_url(
                x["congress_num"], x["bill_type"], x["bill_num"]
            ),
            axis=1,
        )

        # extract legislation text url
        logging.debug("Extracting legislation text url...")
        self.processed_df.loc[:, "text_url"] = self.processed_df.loc[
            :, "api_url"
        ].apply(lambda x: self.extract_text_url(x))

        # extract text
        logging.debug("Extracting legislation text...")
        self.extract_text()

        # post process
        logging.debug("Post-processing self.processed_df...")
        self.post_process()
