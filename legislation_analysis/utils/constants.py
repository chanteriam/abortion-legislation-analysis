import os


PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(PROJECT_PATH, "data", "raw")
API_DATA_PATH = os.path.join(PROJECT_PATH, "data", "api")
CLEANED_DATA_PATH = os.path.join(PROJECT_PATH, "data", "cleaned")
TOKENIZED_DATA_PATH = os.path.join(PROJECT_PATH, "data", "processed")

# api
CONGRESS_API_ROOT = "https://api.congress.gov/v3/bill/"
CONGRESS_ROOT = "https://congress.gov/"
CONGRESS_API_KEY = os.environ.get("CONGRESS_API_KEY")

SCOTUS_DATA = "https://supreme.justia.com/cases-by-topic/abortion-reproductive-rights/"
SCOTUS_ROOT = "https://supreme.justia.com"

# data files
CONGRESS_DATA_FILE = os.path.join(
    API_DATA_PATH, "congress_abortion_legislation_full-text.csv"
)
CONGRESS_CLEANED_DATA_FILE = os.path.join(
    CLEANED_DATA_PATH, "congress_legislation_cleaned"
)

SCOTUS_DATA_FILE = os.path.join(API_DATA_PATH, "scotus_cases_full-text.csv")
SCOTUS_CLEANED_DATA_FILE = os.path.join(CLEANED_DATA_PATH, "scotus_cases_cleaned")
