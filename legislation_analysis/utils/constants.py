import os


# base paths
PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API_DATA_PATH = os.path.join(PROJECT_PATH, "data", "api")
CLEANED_DATA_PATH = os.path.join(PROJECT_PATH, "data", "cleaned")
CLUSTERED_DATA_PATH = os.path.join(PROJECT_PATH, "data", "clustered")
PROCESSED_DATA_PATH = os.path.join(PROJECT_PATH, "data", "processed")
RAW_DATA_PATH = os.path.join(PROJECT_PATH, "data", "raw")
MODELED_DATA_PATH = os.path.join(PROJECT_PATH, "data", "modeled")

# api
CONGRESS_API_KEY = os.environ.get("CONGRESS_API_KEY")
CONGRESS_API_ROOT_URL = "https://api.congress.gov/v3/bill/"
CONGRESS_ROOT_URL = "https://congress.gov/"

SCOTUS_ROOT_URL = "https://supreme.justia.com"
SCOTUS_DATA_URL = (
    f"{SCOTUS_ROOT_URL}/cases-by-topic/abortion-reproductive-rights/"
)


# congress data files
CONGRESS_DATA_CLEANED_FILE_NAME = "congress_legislation_cleaned.fea"
CONGRESS_DATA_TOKENIZED_FILE_NAME = "congress_legislation_tokenized.fea"
SCOTUS_DATA_FILE_POS_TAGGED_NAME = "congress_legislation_pos.fea"
CONGRESS_DATA_CLUSTERED_FILE_NAME = "congress_legislation_clustered.fea"
CONGRESS_DATA_NER_FILE_NAME = "congress_legislation_ner.fea"
CONGRESS_TOPIC_MODEL_FILE_NAME = "congress_tm.model"
CONGRESS_DYN_TOPIC_MODEL_FILE_NAME = "congress_dyn_tm.model"

CONGRESS_DATA_FILE = os.path.join(
    API_DATA_PATH, "congress_abortion_legislation_full-text.csv"
)
CONGRESS_DATA_CLEANED_FILE = os.path.join(
    CLEANED_DATA_PATH, CONGRESS_DATA_CLEANED_FILE_NAME
)
CONGRESS_DATA_TOKENIZED_FILE = os.path.join(
    PROCESSED_DATA_PATH, CONGRESS_DATA_TOKENIZED_FILE_NAME
)
CONGRESS_DATA_POS_TAGGED_FILE = os.path.join(
    PROCESSED_DATA_PATH, SCOTUS_DATA_FILE_POS_TAGGED_NAME
)
CONGRESS_DATA_CLUSTERED_FILE = os.path.join(
    CLUSTERED_DATA_PATH, CONGRESS_DATA_CLUSTERED_FILE_NAME
)
CONGRESS_DATA_NER_FILE = os.path.join(
    PROCESSED_DATA_PATH, CONGRESS_DATA_NER_FILE_NAME
)
CONGRESS_TOPIC_MODEL_FILE = os.path.join(
    MODELED_DATA_PATH, CONGRESS_TOPIC_MODEL_FILE_NAME
)
CONGRESS_DYN_TOPIC_MODEL_FILE = os.path.join(
    MODELED_DATA_PATH, CONGRESS_DYN_TOPIC_MODEL_FILE_NAME
)

CONGRESS_API_COLUMNS = [
    "title",
    "legislation_number",
    "congress",
    "congress_num",
    "bill_type",
    "bill_num",
    "raw_summary",
    "raw_text",
]

# scotus data files
SCOTUS_DATA_CLEANED_FILE_NAME = "scotus_cases_cleaned.fea"
SCOTUS_DATA_TOKENIZED_FILE_NAME = "scotus_cases_tokenized.fea"
CONGRESS_DATA_FILE_POS_TAGGED_NAME = "scotus_cases_pos.fea"
SCOTUS_DATA_FILE_CLUSTERED_NAME = "scotus_cases_clustered.fea"
SCOTUS_DATA_NER_FILE_NAME = "scotus_cases_ner.fea"
SCOTUS_TOPIC_MODEL_FILE_NAME = "scotus_tm.model"

SCOTUS_DATA_FILE = os.path.join(API_DATA_PATH, "scotus_cases_full-text.csv")
SCOTUS_DATA_CLEANED_FILE = os.path.join(
    CLEANED_DATA_PATH, SCOTUS_DATA_CLEANED_FILE_NAME
)
SCOTUS_DATA_TOKENIZED_FILE = os.path.join(
    PROCESSED_DATA_PATH, SCOTUS_DATA_TOKENIZED_FILE_NAME
)
CONGRESS_DATA_FILE_POS_TAGGED = os.path.join(
    PROCESSED_DATA_PATH, CONGRESS_DATA_FILE_POS_TAGGED_NAME
)
SCOTUS_DATA_FILE_CLUSTERED_NAME = os.path.join(
    CLUSTERED_DATA_PATH, SCOTUS_DATA_FILE_CLUSTERED_NAME
)
SCOTUS_DATA_NER_FILE = os.path.join(
    PROCESSED_DATA_PATH, SCOTUS_DATA_NER_FILE_NAME
)
SCOTUS_TOPIC_MODEL_FILE = os.path.join(
    MODELED_DATA_PATH, SCOTUS_TOPIC_MODEL_FILE_NAME
)

SCOTUS_API_COLUMNS = [
    "title",
    "author",
    "author_url",
    "description",
    "raw_text",
]

# general
# cluster numbers determined in exercise 3
LEGAL_DICTIONARY_FILE = os.path.join(CLEANED_DATA_PATH, "legal_terms.txt")
MISC_DICTIONARY_ENTRIES = {
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
    "gonzales",
    "doe",
    "hodgson",
    "webster",
    "maher",
    "arg",
}
GPO_ABBREVS_FILE = os.path.join(CLEANED_DATA_PATH, "gpo_abbrevs.txt")


OPTIMAL_CONGRESS_CLUSTERS = 32
OPTIMAL_SCOTUS_CLUSTERS = 6

# NLP
NLP_MAX_CHAR_LENGTH = 999980

# topic modeling
MIN_NUM_TOPICS_SCOTUS = 4
MAX_NUM_TOPICS_SCOTUS = 10
MIN_NUM_TOPICS_CONGRESS = 5
MAX_NUM_TOPICS_CONGRESS = 35
TOPIC_MODEL_TRAINING_ITERATIONS = 50
SCOTUS_MIN_DF = 2
TFIDF_FILTER_THRESHOLD = 0.75

# misc
NUM_SCOTUS_CASES = 14

# visualization
SCOTUS_ABBREVS = {
    "Dobbs v. Jackson Women's Health Organization": "Dobbs v. Jackson",
    "Gonzales v. Carhart": "Gonzales v. Carhart",
    "Doe v. Bolton": "Doe v. Bolton",
    "Whole Woman's Health v. Hellerstedt": "Health",
    "Roe v. Wade": "Roe v. Wade",
    "Stenberg v. Carhart": "Stenberg v. Carhart",
    "Harris v. McRae": "Harris v. McRae",
    # ruff: noqa: E501
    "Planned Parenthood of Southeastern Pennsylvania v. Casey": "Planned Parenthood v. Casey",
    "Webster v. Reproductive Health Services": "Webster v. Health Services",
    "Maher v. Roe": "Maher v. Roe",
    "Hodgson v. Minnesota": "Hodgson v. Minnesota",
    "Planned Parenthood v. Danforth": "Planned Parenthood v. Danforth",
    "Eisenstadt v. Baird": "Eisenstadt v. Baird",
    "Griswold v. Connecticut": "Griswold v. Connecticut",
}
