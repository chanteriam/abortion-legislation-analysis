"""
Script for pulling legislation text from abortion-related SCOTS decisions.
"""
import logging
import os
import time
from typing import Optional

import bs4
import pandas as pd
import requests

from legislation_analysis.utils.constants import (
    API_DATA_PATH,
    SCOTUS_DATA_URL,
    SCOTUS_ROOT_URL,
)
from legislation_analysis.utils.functions import extract_pdf_text


class SCOTUSDataExtractor:
    """
    Pulls text from SCOTUS abortion-related decisions using supreme.justia.com.
    """

    def __init__(self, scotus_url: str = SCOTUS_DATA_URL):
        """
        Initializes SCOTUSDataExtractor object.
        """
        self.url = scotus_url
        self.df = None

    def extract_case_data(self, request: requests.models.Response) -> None:
        """
        Extracts the case data from the supreme.justia site.
        """
        soup = bs4.BeautifulSoup(request.text, "html.parser")

        data = []

        # Iterate through each section in the div
        for section in soup.select("div.has-margin-top-50 > strong"):
            # Extract case URL and title
            case_tag = section.find("a")

            if not case_tag:
                break

            case_url = SCOTUS_ROOT_URL + case_tag["href"]
            case_title = case_tag.get_text(strip=True)

            # Extract author URL and name
            author_tag = section.find_next_sibling("p").find("a")
            author_url = SCOTUS_ROOT_URL + author_tag["href"]
            author_name = author_tag.get_text(strip=True)

            # Extract description
            description = (
                section.find_next_sibling("p")
                .find_next_sibling("p")
                .get_text(strip=True)
            )

            # Append to data list
            data.append(
                {
                    "title": case_title,
                    "case_url": case_url,
                    "author": author_name,
                    "author_url": author_url,
                    "description": description,
                }
            )

        self.df = pd.DataFrame(data)

    @staticmethod
    def get_pdf_url(case_url: str) -> Optional[str]:
        """
        Gets the pdf url for a given piece of legislation.
        """
        logging.debug(f"\tgetting pdf url from {case_url}...")

        time.sleep(3.6)
        request = requests.get(case_url)
        soup = bs4.BeautifulSoup(request.text, "html.parser")

        # Find the link to the PDF
        pdf_tag = soup.find("a", string="Download PDF")

        if not pdf_tag:
            return None

        logging.debug(f"\t\tpdf tag:{pdf_tag}")

        pdf_url = pdf_tag["href"]

        return pdf_url

    @staticmethod
    def extract_html_text(case_url: str) -> str:
        """
        Extracts the text of a given piece of legislation.
        """
        logging.debug(f"\textracting text from {case_url}...")

        time.sleep(3.6)
        request = requests.get(case_url)
        soup = bs4.BeautifulSoup(request.text, "html.parser")

        text = soup.find(
            "div", class_="-display-inline-block text-left"
        ).get_text()

        return text

    def process(self) -> None:
        """
        Processes the SCOTUS data, extracting the case data and pdf urls.
        """
        request = requests.get(self.url)

        # get case data
        self.extract_case_data(request)
        self.df.loc[:, "pdf_url"] = self.df.loc[:, "case_url"].apply(
            lambda x: self.get_pdf_url(x)
        )

        # if pdf not available, extract text from html
        self.df.loc[self.df.loc[:, "pdf_url"].isna(), "raw_text"] = self.df.loc[
            self.df.loc[:, "pdf_url"].isna(), "case_url"
        ].apply(lambda x: self.extract_html_text(x))

        # extract text from pdf
        self.df.loc[
            ~(self.df.loc[:, "pdf_url"].isna()), "raw_text"
        ] = self.df.loc[~(self.df.loc[:, "pdf_url"].isna()), "pdf_url"].apply(
            lambda x: extract_pdf_text(x)
        )


def main() -> None:
    """
    Processes SCOTUS abortion legislation, pulling text from pdf urls.
    """
    scotus_api = SCOTUSDataExtractor()
    scotus_api.process()

    # save data
    scotus_api.df.to_csv(
        os.path.join(API_DATA_PATH, "scotus_cases_full-text.csv"), index=False
    )
