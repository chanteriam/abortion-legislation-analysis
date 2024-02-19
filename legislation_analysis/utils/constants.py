import os


# Base Path Constants
PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API_DATA_PATH = os.path.join(PROJECT_PATH, "data", "api")
CLEANED_DATA_PATH = os.path.join(PROJECT_PATH, "data", "cleaned")
CLUSTERED_DATA_PATH = os.path.join(PROJECT_PATH, "data", "clustered")
PROCESSED_DATA_PATH = os.path.join(PROJECT_PATH, "data", "processed")
RAW_DATA_PATH = os.path.join(PROJECT_PATH, "data", "raw")

# API Constants
CONGRESS_API_KEY = os.environ.get("CONGRESS_API_KEY")
CONGRESS_API_ROOT_URL = "https://api.congress.gov/v3/bill/"
CONGRESS_ROOT_URL = "https://congress.gov/"

SCOTUS_ROOT_URL = "https://supreme.justia.com"
SCOTUS_DATA_URL = (
    f"{SCOTUS_ROOT_URL}/cases-by-topic/abortion-reproductive-rights/"
)


# Data File Constants
CONGRESS_DATA_FILE = os.path.join(
    API_DATA_PATH, "congress_abortion_legislation_full-text.csv"
)
CONGRESS_DATA_FILE_CLEANED = os.path.join(
    CLEANED_DATA_PATH, "congress_legislation_cleaned.fea"
)
SCOTUS_DATA_FILE = os.path.join(API_DATA_PATH, "scotus_cases_full-text.csv")
SCOTUS_DATA_FILE_CLEANED = os.path.join(
    CLEANED_DATA_PATH, "scotus_cases_cleaned.fea"
)

# General Constants
TAGS_OF_INTEREST = ["NOUN", "ADJ", "VERB", "ADV"]
