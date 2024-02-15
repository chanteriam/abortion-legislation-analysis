import time
from io import BytesIO

import pandas as pd
import requests
from PyPDF2 import PdfReader


def extract_pdf_text(pdf_url: str, verbose: bool = True) -> str:
    """
    Extracts the text of a given piece of legislation.
    """
    if verbose:
        print(f"\textracting text from {pdf_url}...")

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
    """
    ext = filepath.split(".")
    if ext.lower() == "pickle":
        df = pd.read_pickle(filepath)
    elif ext == ".csv":
        df = pd.read_csv(filepath)
    elif ext == ".xlsx":
        df = pd.read_excel(filepath)
    else:
        raise ValueError(f"File type {ext} not supported.")

    return df


def save(df: pd.DataFrame, filepath: str) -> None:
    """
    Saves the given dataframe out to the specified filepath.
    """
    ext = filepath.split(".")
    if ext.lower() == "pickle":
        df.to_pickle(filepath)
    elif ext == ".csv":
        df.to_csv(filepath, index=False)
    elif ext == ".xlsx":
        df.to_excel(filepath, index=False)
    else:
        raise ValueError(f"File type {ext} not supported.")
