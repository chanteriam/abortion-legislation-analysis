import requests
from io import BytesIO
from PyPDF2 import PdfReader
import time


def extract_pdf_text(pdf_url, verbose=True):
    """
    Extracts the text of a given piece of legislation.

    parameters:
        pdf_url (str): url for the pdf of the legislation.

    returns:
        text (str): text of the legislation.
    """
    if verbose:
        print(f"\textracting text from {pdf_url}...")

    time.sleep(3.6)  # to prevent overloading the api

    response = requests.get(pdf_url)
    pdf_file = BytesIO(response.content)
    reader = PdfReader(pdf_file)

    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"

    return text


def save(df, filepath):
    """
    Saves the given dataframe out to the specified filepath.
    """
    df.to_csv(filepath, index=False)