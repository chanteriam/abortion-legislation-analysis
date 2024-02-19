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
    logging.debug(f"\textracting text from {pdf_url}.")

    # Prevent overloading the api
    time.sleep(3.6)

    response = requests.get(pdf_url)
    pdf_file = BytesIO(response.content)
    reader = PdfReader(pdf_file)

    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"

    return text


def load_file_to_df(file_path: str) -> pd.DataFrame:
    """
    Loads a file into a dataframe.

    parameters:
        file_path (str): path to the file to load.
        load_tokenized (bool): whether or not to load tokenized data.
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


def save(df: pd.DataFrame, file_path: str) -> None:
    """
    Saves the given dataframe out to the specified file_path.

    parameters:
        df (pd.DataFrame): dataframe to save.
        file_path (str): path to save the dataframe to.
    """
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
