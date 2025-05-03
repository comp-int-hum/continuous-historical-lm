import os
from custom import *

# Local paths (TO CHANGE)
DATA_ROOT = os.path.expanduser("~/corpora") 
GUTENBERG_PATH = os.path.join(DATA_ROOT, "gutenberg/")

WORK_DIR = f"0_data_preparation/work/{PROJECT_NAME}"
ORIGINAL_WORK_DIR = WORK_DIR

# Author and work filtering
P1_THRESH = 90
P2_THRESH = 101
BD_THRESH = 5
OMIT_AUTHORS = []
MAX_WORKS = 20

# Model and prompt settings
USE_INFERENCE = True
WORK_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
USE_DATES_FILE = True
DATES_FILE ="0_data_preparation/data/gb_authors_dates_1950.jsonl"

#dataset settings
USE_CA = True
CA_ROOT = "0_data_preparation/data/CA"

# Custom data preparation settings
STARTING_YEAR = 1750
ENDING_YEAR = 1940
TIME_SLICES = []
CATEGORY_MODE = "TOTAL" # can be 'TOTAL' or 'INTERVAL'
SHUFFLE = 0 # 0 for no shuffle, 1 for shuffle
SORT = 1 # 0 for no sorting, 1 for sorting
TOTAL_SPLITS = 20
SPLIT_INTERVAL = 20 # Interval for splitting the data, in years


# Data split settings
TRAIN_PORTION = 0.7
DEV_PORTION = 0.1
TEST_PORTION = 0.2
SPLIT_STYLE = "percent" #percent or count
SPLIT_LEVEL = "paragraph" # can be sentence, paragraph, or chapter.
