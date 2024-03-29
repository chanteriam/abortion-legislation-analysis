"""
Script for pulling legislation text from abortion-related SCOTS decisions.
"""

import logging
import time
from typing import Optional

import bs4
import pandas as pd
import requests

from legislation_analysis.utils.constants import (
    SCOTUS_API_COLUMNS,
    SCOTUS_DATA_URL,
    SCOTUS_ROOT_URL,
)


class SCOTUSDataExtractor:
    """
    Pulls text from SCOTUS abortion-related decisions using supreme.justia.com.

    parameters:
        scotus_url (str): url to the supreme.justia.com abortion decisions.
    """

    def __init__(self, scotus_url: str = SCOTUS_DATA_URL):
        """
        Initializes SCOTUSDataExtractor object.
        """
        self.url = scotus_url
        self.processed_df = None

    def extract_case_data(self, request: requests.models.Response) -> None:
        """
        Extracts the case data from the supreme.justia site.

        parameters:
            request (requests.models.Response): request object from the
                supreme.justia site.
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

        self.processed_df = pd.DataFrame(data)

    @staticmethod
    def get_pdf_url(case_url: str) -> Optional[str]:
        """
        Gets the pdf url for a given piece of legislation.

        parameters:
            case_url (str): url to the case.

        returns:
            pdf_url (str): url to the pdf.
        """
        logging.debug(f"\tGetting pdf url from {case_url}...")

        time.sleep(3.6)
        request = requests.get(case_url)
        soup = bs4.BeautifulSoup(request.text, "html.parser")

        # Find the link to the PDF
        pdf_tag = soup.find("a", string="Download PDF")

        if not pdf_tag:
            return None

        logging.debug(f"\tPDF tag:{pdf_tag}")

        pdf_url = pdf_tag["href"]

        return pdf_url

    @staticmethod
    def extract_html_text(case_url: str) -> str:
        """
        Extracts the text of a given decision.

        parameters:
            case_url (str): url of the pdf to extract text from.

        returns:
            text (str): extracted text.
        """
        logging.debug(f"\tExtracting text from {case_url}...")

        time.sleep(3.6)
        request = requests.get(case_url)
        soup = bs4.BeautifulSoup(request.text, "html.parser")

        return soup.find(
            "div", class_="-display-inline-block text-left"
        ).get_text()

    def process(self) -> None:
        """
        Processes the SCOTUS data, extracting the decision data and pdf urls.
        """
        request = requests.get(self.url)

        # get case data
        self.extract_case_data(request)
        self.processed_df["raw_text"] = self.processed_df["case_url"].apply(
            lambda x: self.extract_html_text(x)
        )
        self.processed_df = self.processed_df.loc[:, SCOTUS_API_COLUMNS]
