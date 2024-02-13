"""
Develops a data cleaner class for the abortion legislation content analysis
project.
"""

import os
import pandas as pd
import time
from openai import OpenAI

from legislation_analysis.utils.constants import (
    CLEANED_DATA_PATH,
    CONGRESS_DATA_FILE,
    SCOTUS_DATA_FILE,
)
from legislation_analysis.utils.functions import save


class Cleaner:
    """
    Abstract class for a cleaner object.

    parameters:
        filepath (str): path to the legislation text csv.
        filename (str): name of file to save.
    """

    TOTAL_TOKENS_USED = 0
    TOKEN_LIMIT = 3000
    OPENAI_CLIENT = OpenAI()

    def __init__(
        self,
        file_path=CONGRESS_DATA_FILE,
        file_name="congress_legislation_cleaned.csv",
    ):
        self.df = pd.read_csv(file_path)
        self.file_name = file_name
        self.cleaned_df = None
        self.save_path = os.path.join(CLEANED_DATA_PATH, self.file_name)

    @classmethod
    def clean_text(cls, text: str) -> str:
        """
        Spell checks text.

        parameters:
            text (str): text to spell check.

        returns:
            text (str): spell checked text.
        """
        checked_text = ""
        spell_check_prompt = {
            "role": "system",
            "content": """
            You are a legislation-cleaning assistant. You correct legislative text delimited with triple quotes by
            identifying and fixing errors such as misplaced or duplicate spaces, missing spaces,
            removal of new line characters, and identification and concatenation of split up words.
            In your response, please only included the corrected text and no additional information.

            [Example Request]
            Please clean this piece of legislation:
            \"\"\"
            \n\n\nThis is a pi ece of legis-lation that   nee ds to\n\n becleaned..
            \"\"\"'

            [Example Response]
            \"\"\"
            This is a piece of legislation that needs to be cleaned.
            \"\"\"
            """,
        }
        tokens = text.split(" ")
        max_token_ct = len(tokens)
        start_time = time.time()

        for i in range(0, max_token_ct, cls.TOKEN_LIMIT):
            # timer for token limitations
            if cls.TOTAL_TOKENS_USED + len(tokens[i : i + cls.TOKEN_LIMIT]) > 60000:
                end_time = time.time()
                elapsed_time = end_time - start_time
                if elapsed_time > 60:
                    cls.TOTAL_TOKENS_USED = 0
                    start_time = time.time()
                else:
                    while elapsed_time < 60:
                        print("Sleeping...")
                        elapsed_time = end_time - start_time
                    cls.TOTAL_TOKENS_USED = 0
                    start_time = time.time()

            cls.TOTAL_TOKENS_USED += len(tokens[i : i + cls.TOKEN_LIMIT])
            text = " ".join(tokens[i : i + cls.TOKEN_LIMIT])

            message = {
                "role": "user",
                "content": f"""Please clean this piece of legislation:
                \"\"\"{text}\"\"\"
                """,
            }

            response = cls.OPENAI_CLIENT.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                messages=[spell_check_prompt, message],
            )

            resp = response.choices[0].message.content.strip()
            resp = resp.replace("'", "'")
            resp = resp.strip('"').strip("\n").strip(" ")

            checked_text += f"{resp} "

        return checked_text

    def process(
        self,
        verbose: bool = True,
        cols_to_clean=[
            ("raw_text", "cleaned_text"),
        ],
    ) -> None:
        """
        Processes the legislation text.

        parameters:
            verbose (bool): whether to print status updates.
        """
        for col in cols_to_clean:
            if verbose:
                print(f"Cleaning {col[0]}...")
            col, new_col = col
            df = self.df.dropna(subset=[col]).copy()
            df[new_col] = df[col].copy()

            # clean text
            df[new_col] = df[new_col].apply(lambda x: self.clean_text(x))

        self.cleaned_df = df


def main(verbose: bool = True) -> None:
    """
    Runs data cleaner.

    parameters:
        verbose (bool): whether to print status updates.

    returns:
        True (bool): whether the data cleaner ran successfully.
    """
    congress_cleaner = Cleaner(CONGRESS_DATA_FILE, "congress_legislation_cleaned.csv")
    scotus_cleaner = Cleaner(SCOTUS_DATA_FILE, "scotus_cases_cleaned.csv")

    congress_cleaner.process(verbose)
    scotus_cleaner.process(verbose, cols_to_clean=[("raw_text", "cleaned_text")])

    save(congress_cleaner.cleaned_df, congress_cleaner.save_path)
    save(scotus_cleaner.cleaned_df, scotus_cleaner.save_path)
