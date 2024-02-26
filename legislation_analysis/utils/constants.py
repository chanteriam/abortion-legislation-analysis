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
CONGRESS_DATA_FILE = os.path.join(
    API_DATA_PATH, "congress_abortion_legislation_full-text.csv"
)
CONGRESS_DATA_FILE_CLEANED = os.path.join(
    CLEANED_DATA_PATH, "congress_legislation_cleaned.fea"
)
CONGRESS_DATA_CLUSTERED_FILE_NAME = "congress_legislation_clustered.fea"
CONGRESS_DATA_POS_TAGGED_FILE_NAME = "congress_legislation_pos.fea"
CONGRESS_DATA_FILE_TOPIC_MODELED_NAME = "congress_legislation_topic_modeled.fea"
CONGRESS_DATA_POS_TAGGED_FILE = os.path.join(
    PROCESSED_DATA_PATH, CONGRESS_DATA_POS_TAGGED_FILE_NAME
)

# scotus data files
SCOTUS_DATA_FILE = os.path.join(API_DATA_PATH, "scotus_cases_full-text.csv")
SCOTUS_DATA_FILE_CLEANED = os.path.join(
    CLEANED_DATA_PATH, "scotus_cases_cleaned.fea"
)
SCOTUS_DATA_FILE_CLUSTERED_NAME = "scotus_cases_clustered.fea"
SCOTUS_DATA_FILE_POS_TAGGED_NAME = "scotus_cases_pos.fea"
SCOTUS_DATA_FILE_TOPIC_MODELED_NAME = "scotus_cases_topic_modeled.fea"
SCOTUS_DATA_FILE_POS_TAGGED = os.path.join(
    PROCESSED_DATA_PATH, SCOTUS_DATA_FILE_POS_TAGGED_NAME
)

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
    "doe",
    "hodgson",
    "webster",
    "maher",
}
MISC_STATE_ABBREVIATIONS = {
    "a.s.",
    "ak.",
    "ala.",
    "alaska",
    "ariz.",
    "ark.",
    "cal.",
    "calif.",
    "cnmi",
    "colo.",
    "conn.",
    "d.c.",
    "del.",
    "dis. col.",
    "fla.",
    "ga.",
    "geo.",
    "guam",
    "hawaii",
    "hi.; haw.",
    "ida.",
    "idaho",
    "ill.",
    "ind.",
    "ioa.",
    "iowa",
    "ka.",
    "kan.",
    "kans.",
    "ken.",
    "kent.",
    "ky.",
    "la.",
    "m.p.",
    "maine",
    "mar.",
    "mary.",
    "mass.",
    "md.",
    "mich.",
    "minn.",
    "miss.",
    "mo.",
    "mont.",
    "n. car.",
    "n. dak.",
    "n. jersey",
    "n. mex.",
    "n. york",
    "n.c.",
    "n.d.",
    "n.h.",
    "n.j.",
    "n.m.",
    "n.y.",
    "neb.",
    "nebr.",
    "nev.",
    "new m.",
    "new mex.",
    "o.",
    "oh.",
    "ohio",
    "okla.",
    "ore.",
    "oreg.",
    "p.p.",
    "p.r.",
    "pa.",
    "penn.",
    "penna.",
    "qm",
    "qw",
    "r.i.",
    "s. car.",
    "s. dak.",
    "s.c.",
    "s.d.",
    "sodak",
    "tenn.",
    "terr.",
    "tex.",
    "texas",
    "u.s.",
    "u.s.a.",
    "u.s.v.i.",
    "ut.",
    "utah",
    "v.i.",
    "va.",
    "verm.",
    "virg.",
    "vt.",
    "w. va.",
    "w. virg.",
    "w.v.",
    "w.va.",
    "wash.",
    "wis.",
    "wisc.",
    "wn.",
    "wyo.",
    "xb",
    "xh",
    "xl",
    "xm",
    "xq",
    "xu",
    "xv",
}
OPTIMAL_CONGRESS_CLUSTERS = 32
OPTIMAL_SCOTUS_CLUSTERS = 6

# NLP
NLP_MAX_CHAR_LENGTH = 999980

# topic modeling
MIN_NUM_TOPICS_SCOTUS = 2
MAX_NUM_TOPICS_SCOTUS = 20
MIN_NUM_TOPICS_CONGRESS = 5
MAX_NUM_TOPICS_CONGRESS = 35
TOPIC_MODEL_TRAINING_ITERATIONS = 50

# misc
NUM_SCOTUS_CASES = 13
