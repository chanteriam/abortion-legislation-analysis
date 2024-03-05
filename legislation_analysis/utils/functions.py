import logging
import os
import re
import time
from io import BytesIO

import pandas as pd
import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader

from legislation_analysis.utils.constants import (
    GPO_ABBREVS_FILE,
    LEGAL_DICTIONARY_FILE,
)


def extract_pdf_text(pdf_url: str) -> str:
    """
    Extracts the text of a given piece of legislation.

    parameters:
        pdf_url (str): url of the pdf to extract text from.

    returns:
        text (str): extracted text.
    """
    logging.debug(f"\tExtracting text from {pdf_url}...")

    # Prevent overloading the api
    time.sleep(3.6)

    response = requests.get(pdf_url)
    pdf_file = BytesIO(response.content)
    reader = PdfReader(pdf_file)

    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"

    return text


def get_legal_dictionary() -> set:
    """
    Scrapes dictionary of legal terms from https://www.uscourts.gov/glossary.

    returns:
        legal_terms (set): set of legal terms.
    """
    if os.path.exists(LEGAL_DICTIONARY_FILE):
        with open(LEGAL_DICTIONARY_FILE, "r") as file:
            return set(file.read().split())

    url = "https://www.uscourts.gov/glossary"
    legal_terms = set()

    # get the html content
    html_content = requests.get(url).text
    soup = BeautifulSoup(html_content, "html.parser").find(
        "div", class_="lexicon-list"
    )

    # get the legal terms
    for term in soup.find_all("dt"):
        term = term.text.strip().lower()

        if len(term.split()) > 1:
            terms = term.split()
            for t in terms:
                legal_terms.add(t)
        else:
            legal_terms.add(term)

    # save the terms to a file
    with open(LEGAL_DICTIONARY_FILE, "w") as file:
        for term in legal_terms:
            file.write(term + "\n")

    return legal_terms


def get_gpo_dictionary() -> set:
    """
    Scrapes common state abbreviations from
    https://en.wikipedia.org/wiki/List_of_U.S._state_and_territory_abbreviations

    returns:
        gpo_terms (set): set of gpo/state abbreviation terms.
    """
    if os.path.exists(GPO_ABBREVS_FILE):
        with open(GPO_ABBREVS_FILE, "r") as file:
            return set(file.read().split())

    url = "https://en.wikipedia.org/wiki/List_of_U.S._state_and_territory_abbreviations"

    # get the html content
    html_content = requests.get(url).text
    table = (
        BeautifulSoup(html_content, "html.parser").find("table").find("tbody")
    )

    # get the state abbreviations
    gpos = []
    aps = []
    others = []
    for row in table.find_all("tr")[11:]:
        gpos.append(row.find_all("td")[-3].text)
        aps.append(row.find_all("td")[-2].text)
        others.append(row.find_all("td")[-1].text)

    # process terms
    gpos = [gpo.replace("\xa0", " ").lower() for gpo in gpos if len(gpo) > 0]
    aps = [ap.replace("\xa0", " ").lower() for ap in aps if len(ap) > 0]
    others = [
        other.replace("\xa0", " ").lower().strip("\n")
        for other in others
        if len(other.strip("\n")) > 0
    ]

    others_2 = []
    for other in others:
        other = re.sub(r"\[.\]?", "", other)
        if "," in other or "&" in other:
            split = ", " if "," in other else " & "
            others_2.extend(other.split(split))
            continue
        others_2.append(other)

    # build dictionary
    gpo_terms = set(gpos + aps + others_2)

    # save the terms to a file
    with open(GPO_ABBREVS_FILE, "w") as file:
        for term in gpo_terms:
            file.write(term + "\n")

    return gpo_terms


def load_file_to_df(file_path: str) -> pd.DataFrame:
    """
    Loads a file into a dataframe.

    parameters:
        file_path (str): path to the file to load.
        load_tokenized (bool): whether to load tokenized data.
        tokenized_cols (list): list of columns to load tokenized data for.

    returns:
        df (pd.DataFrame): dataframe of the file.
    """
    ext = file_path.split(".")[-1].lower()

    if ext in ["pickle", "pkl"]:
        df = pd.read_pickle(file_path)
    elif ext in ["csv", "txt"]:
        df = pd.read_csv(file_path)
    elif ext in ["xlsx", "xls"]:
        df = pd.read_excel(file_path)
    elif ext in ["fea", "feather"]:
        df = pd.read_feather(file_path)
    else:
        raise ValueError(f"File type {ext} not supported.")

    return df


def save_df_to_file(df: pd.DataFrame, file_path: str) -> None:
    """
    Saves the given dataframe out to the specified file_path.

    parameters:
        df (pd.DataFrame): dataframe to save.
        file_path (str): path to save the dataframe to.
    """
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    ext = file_path.split(".")[-1].lower()

    if ext in ["pickle", "pkl"]:
        df.to_pickle(file_path)
    elif ext in ["csv", "txt"]:
        df.to_csv(file_path, index=False)
    elif ext in ["xlsx", "xls"]:
        df.to_excel(file_path, index=False)
    elif ext in ["fea", "feather"]:
        df.to_feather(file_path)
    else:
        raise ValueError(f"File type {ext} not supported.")
