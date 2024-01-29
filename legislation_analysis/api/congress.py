"""
Script for pulling legislation text or pdfs from api.congress.gov.
Search results derived from https://www.congress.gov/advanced-search/legislation
using the following search terms:
    - abortion
    - reproduction
    - reproductive health
"""
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
    API_DATA_PATH,
    CONGRESS_API_KEY,
    CONGRESS_API_ROOT_URL,
    CONGRESS_ROOT_URL,
    RAW_DATA_PATH,
)
from legislation_analysis.utils.functions import extract_pdf_text


class CongressAPI:
    """
    Pulls legislation text from api.congress.gov.

    parameters:
        filepath (str): filepath to csv of legislation search results.
    """

    def __init__(self, filepath):
        self.filepath = filepath
        self.filename = os.path.basename(filepath).split(".")[0]
        self.processed_df = None
        self.df = None

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
    def get_api_url(
        congress_num: int, bill_type: str, bill_num: int, verbose: bool = True
    ) -> str:
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
    def extract_text_url(api_url: str, verbose: bool = True) -> Optional[str]:
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

        if verbose:
            print(f"\tExtracting text url from {api_url}...")

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
    def build_text_url(
        congress_num: int, bill_type: str, bill_num: int, verbose=True
    ) -> str:
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
    def extract_text(text_url: str, verbose: bool = True) -> str:
        """
        Extracts the text of a given piece of legislation.

        parameters:
            text_url (str): url for the text of the legislation.

        returns:
            text (str): text of the legislation.
        """
        if verbose:
            print(f"\textracting text from {text_url}...")

        request = requests.get(text_url)
        soup = bs4.BeautifulSoup(request.text, "html.parser")
        text = soup.text

        return text

    def process(
        self, file_path: str, file_name: str, verbose: bool = True
    ) -> None:
        """
        Processing function for extracting text from legislation search results.
        """

        # load data
        if verbose:
            print(f"Loading data from {file_path}...")
        df = pd.read_csv(file_path)

        # get header row
        first_col = df.iloc[:, 0]
        header_row = list(first_col).index("Legislation Number")
        col_row = list(df.iloc[header_row, 0:]).index(np.nan)

        df = df.iloc[:, :col_row].copy()
        df.columns = list(df.iloc[header_row, :col_row])
        self.df = df.iloc[header_row + 1 :, :].reset_index(drop=True).copy()

        self.df.columns = [c.lower() for c in list(self.df.columns)]

        # extract legislation information
        if verbose:
            print("Extracting legislation information...")
        self.processed_df = self.extract_legislation_details(self.df)

        # extract legislation api url
        if verbose:
            print("Extracting legislation api url...")

        self.processed_df.loc[:, "api_url"] = self.processed_df.apply(
            lambda x: self.get_api_url(
                x["congress_num"], x["bill_type"], x["bill_num"], verbose
            ),
            axis=1,
        )

        # intermediary saving
        save_path = os.path.join(
            os.path.dirname(file_path), f"{file_name}_api_url.csv"
        )
        self.processed_df.to_csv(save_path, index=False)

        # extract legislation text url
        if verbose:
            print("Extracting legislation text url...")
        self.processed_df.loc[:, "text_url"] = self.processed_df.loc[
            :, "api_url"
        ].apply(lambda x: self.extract_text_url(x, verbose))

        # extract htm text
        self.processed_df.loc[
            (self.processed_df.loc[:, "text_url"].str[-3:] == "htm"),
            "text",
        ] = self.processed_df.loc[
            (self.processed_df.loc[:, "text_url"].str[-3:] == "htm"),
            "text_url",
        ].apply(
            lambda x: self.extract_text(x, verbose)
        )

        save_path = os.path.join(
            os.path.dirname(file_path), f"{file_name}_htm-text.csv"
        )
        self.processed_df.to_csv(save_path, index=False)

        # extract pdf text
        self.processed_df.loc[
            (self.processed_df.loc[:, "text_url"].str[-3:] == "pdf"),
            "text",
        ] = self.processed_df.loc[
            (self.processed_df.loc[:, "text_url"].str[-3:] == "pdf"),
            "text_url",
        ].apply(
            lambda x: extract_pdf_text(x, verbose)
        )

        # remove rows with null text values
        self.processed_df = self.processed_df.loc[
            ~self.processed_df.loc[:, "text"].isna(), :
        ].copy()
        self.processed_df.rename(columns={"text": "raw_text"}, inplace=True)


def main(verbose: bool = True) -> None:
    """
    Iterates through legislation csv search results and extracts the text of
    each bill. Saves the results to a new csv titled filename_text.csv.
    """
    file_path = os.path.join(RAW_DATA_PATH, "congress_abortion_legislation.csv")
    file_name = os.path.basename(file_path).split(".")[0]

    cleaner = CongressAPI(file_path)

    # process legislation data
    cleaner.process(file_path, file_name, verbose)

    # save to csv
    save_path = os.path.join(API_DATA_PATH, f"{file_name}_full-text.csv")
    cleaner.processed_df.to_csv(save_path, index=False)
