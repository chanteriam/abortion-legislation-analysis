"""
Services for topic modeling.
"""

import logging

from legislation_analysis.topic_modeling.topic_modeling import TopicModeling
from legislation_analysis.utils.constants import (
    CONGRESS_DATA_FILE_TOPIC_MODELED_NAME,
    CONGRESS_DATA_POS_TAGGED_FILE,
    NUM_SCOTUS_CASES,
    SCOTUS_DATA_FILE_POS_TAGGED,
    SCOTUS_DATA_FILE_TOPIC_MODELED_NAME,
)


def run_topic_modeling() -> None:
    """
    Runs topic modeling for both Congressional legislation and SCOTUS decisions.
    """
    # run topic modeling for Congressional legislation
    logging.info("Starting topic modeling for Congressional legislation...")
    congress_tm = TopicModeling(
        file_path=CONGRESS_DATA_POS_TAGGED_FILE,
        save_name=CONGRESS_DATA_FILE_TOPIC_MODELED_NAME,
        testing=True,
        column="text_pos_tags_of_interest",
    )
    congress_tm.gen_topic_model(True)

    # run topic modeling for SCOTUS decisions
    logging.info("Starting topic modeling for SCOTUS decisions...")
    scotus_tm = TopicModeling(
        file_path=SCOTUS_DATA_FILE_POS_TAGGED,
        save_name=SCOTUS_DATA_FILE_TOPIC_MODELED_NAME,
        testing=True,
        column="text_pos_tags_of_interest",
        min_df=NUM_SCOTUS_CASES // 5,
    )
    scotus_tm.gen_topic_model(True)
