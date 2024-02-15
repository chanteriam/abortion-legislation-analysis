import logging
import time
from io import BytesIO

import pandas as pd
import requests
from PyPDF2 import PdfReader


def extract_pdf_text(pdf_url: str) -> str:
    """
    Extracts the text of a given piece of legislation.

    parameters:
        pdf_url (str): url of the pdf to extract text from.

    returns:
        text (str): extracted text.
    """
    logging.debug(f"\textracting text from {pdf_url}...")

    # Prevent overloading the api
    time.sleep(3.6)

    response = requests.get(pdf_url)
    pdf_file = BytesIO(response.content)
    reader = PdfReader(pdf_file)

    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"

    return text


def load_file_to_df(filepath: str) -> pd.DataFrame:
    """
    Loads a file into a dataframe.

    parameters:
        filepath (str): path to the file to load.

    returns:
        df (pd.DataFrame): dataframe of the file.
    """
    ext = filepath.split(".")[-1]
    if ext.lower() in ["pickle", "pkl"]:
        df = pd.read_pickle(filepath)
    elif ext[".csv", ".txt"]:
        df = pd.read_csv(filepath)
    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(filepath)
    else:
        raise ValueError(f"File type {ext} not supported.")

    return df


def save(df: pd.DataFrame, filepath: str) -> None:
    """
    Saves the given dataframe out to the specified filepath.

    parameters:
        df (pd.DataFrame): dataframe to save.
        filepath (str): path to save the dataframe to.
    """
    ext = filepath.split(".")[-1]
    if ext.lower() in ["pickle", "pkl"]:
        df.to_pickle(filepath)
    elif ext[".csv", ".txt"]:
        df.to_csv(filepath, index=False)
    elif ext in [".xlsx", ".xls"]:
        df.to_excel(filepath, index=False)
    else:
        raise ValueError(f"File type {ext} not supported.")
