import os


# base paths
PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API_DATA_PATH = os.path.join(PROJECT_PATH, "data", "api")
CLEANED_DATA_PATH = os.path.join(PROJECT_PATH, "data", "cleaned")
RAW_DATA_PATH = os.path.join(PROJECT_PATH, "data", "raw")
PROCESSED_DATA_PATH = os.path.join(PROJECT_PATH, "data", "processed")

# api
CONGRESS_API_ROOT_URL = "https://api.congress.gov/v3/bill/"
CONGRESS_ROOT_URL = "https://congress.gov/"
CONGRESS_API_KEY = os.environ.get("CONGRESS_API_KEY")

SCOTUS_ROOT_URL = "https://supreme.justia.com"
SCOTUS_DATA_URL = f"{SCOTUS_ROOT_URL}/cases-by-topic/abortion-reproductive-rights/"


# data files
CONGRESS_DATA_FILE = os.path.join(
    API_DATA_PATH, "congress_abortion_legislation_full-text.csv"
)
CONGRESS_DATA_FILE_CLEANED = os.path.join(
    CLEANED_DATA_PATH, "congress_legislation_cleaned.fea"
)
SCOTUS_DATA_FILE = os.path.join(API_DATA_PATH, "scotus_cases_full-text.csv")
SCOTUS_DATA_FILE_CLEANED = os.path.join(CLEANED_DATA_PATH, "scotus_cases_cleaned.fea")
LEGAL_DICTIONARY_FILE = os.path.join(CLEANED_DATA_PATH, "legal_terms.txt")
MISC_DICTIONARY_ENTRIES = set(
    [
        "dobbs",
        "roe",
        "wade",
        "breyer",
        "sotomayor",
        "kagan",
        "hellerstedt",
        "carhart",
        "stenberg",
        "eisenstadt",
        "baird",
        "griswold",
        "doe",
        "hodgson",
        "webster",
        "maher",
    ]
)

# NLP
NLP_MAX_CHAR_LENGTH = 999980
