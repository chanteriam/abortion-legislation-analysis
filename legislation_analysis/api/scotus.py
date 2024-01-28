"""
Script for pulling legislation text from abortion-related SCOTS decisions.
"""

# imports
import os
import time

import bs4
import pandas as pd
import requests

# constants
from legislation_analysis.utils.constants import (
    API_DATA_PATH,
    SCOTUS_DATA,
    SCOTUS_ROOT,
)

# functions
from legislation_analysis.utils.functions import extract_pdf_text


class SCOTUSDataExtractor:
    """
    Pulls text from SCOTUS abortion-related decisions using supreme.justia.com.
    """

    def __init__(self, verbose=True, scotus_url=SCOTUS_DATA):
        """
        Initializes SCOTUSDataExtractor object.

        parameters:
            verbose (bool): whether to print status updates.
        """
        self.verbose = verbose
        self.url = scotus_url
        self.df = None

    def extact_case_data(self, request):
        """
        Extracts the case data from the supreme.justia site.

        parameters:
            request (requests.models.Response): request object for the url.
        """
        soup = bs4.BeautifulSoup(request.text, "html.parser")

        data = []

        # Iterate through each section in the div
        for section in soup.select("div.has-margin-top-50 > strong"):
            # Extract case URL and title
            case_tag = section.find("a")

            if not (case_tag):
                break

            case_url = SCOTUS_ROOT + case_tag["href"]
            case_title = case_tag.get_text(strip=True)

            # Extract author URL and name
            author_tag = section.find_next_sibling("p").find("a")
            author_url = SCOTUS_ROOT + author_tag["href"]
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

    def get_pdf_url(self, case_url):
        """
        Gets the pdf url for a given piece of legislation.

        parameters:
            case_url (str): url for the given legislation.

        returns:
            pdf_url (str): url for the pdf of the given legislation.
        """
        if self.verbose:
            print(f"\tgetting pdf url from {case_url}...")

        time.sleep(3.6)
        request = requests.get(case_url)
        soup = bs4.BeautifulSoup(request.text, "html.parser")

        # Find the link to the PDF
        pdf_tag = soup.find("a", string="Download PDF")

        if not (pdf_tag):
            return None

        if self.verbose:
            print(f"\t\tpdf tag:{pdf_tag}")

        pdf_url = pdf_tag["href"]

        return pdf_url

    def extract_html_text(self, case_url):
        """
        Extracts the text of a given piece of legislation.

        parameters:
            case_url (str): url for the given legislation.

        returns:
            text (str): text of the legislation.
        """
        if self.verbose:
            print(f"\textracting text from {case_url}...")

        time.sleep(3.6)
        request = requests.get(case_url)
        soup = bs4.BeautifulSoup(request.text, "html.parser")

        text = soup.find("div", class_="-display-inline-block text-left").get_text()

        return text

    def process(self):
        """
        Processes the SCOTUS data, extracting the case data and pdf urls.
        """
        request = requests.get(self.url)

        # get case data
        self.extact_case_data(request)
        self.df.loc[:, "pdf_url"] = self.df.loc[:, "case_url"].apply(
            lambda x: self.get_pdf_url(x)
        )

        # if pdf not available, extract text from html
        self.df.loc[self.df.loc[:, "pdf_url"].isna(), "raw_text"] = self.df.loc[
            self.df.loc[:, "pdf_url"].isna(), "case_url"
        ].apply(lambda x: self.extract_html_text(x))

        # extract text from pdf
        self.df.loc[~(self.df.loc[:, "pdf_url"].isna()), "raw_text"] = self.df.loc[
            ~(self.df.loc[:, "pdf_url"].isna()), "pdf_url"
        ].apply(lambda x: extract_pdf_text(x))


def main(verbose=True):
    """
    Processes SCOTUS abortion legislation, pulling text from pdf urls.

    parameters:
        verbose (bool): whether to print status updates.

    returns:
        True (bool): whether the function ran successfully.
    """
    scotus_api = SCOTUSDataExtractor(verbose=verbose)
    scotus_api.process()

    # save data
    scotus_api.df.to_csv(
        os.path.join(API_DATA_PATH, "scotus_cases_full-text.csv"), index=False
    )

    return True
