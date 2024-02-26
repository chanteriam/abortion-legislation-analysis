"""
Services for topic modeling.
"""

import logging

from legislation_analysis.topic_modeling.topic_modeling import TopicModeling
from legislation_analysis.utils.constants import (
    CONGRESS_DATA_FILE_POS_TAGGED,
    CONGRESS_TOPIC_MODEL,
    MAX_NUM_TOPICS_CONGRESS,
    MAX_NUM_TOPICS_SCOTUS,
    MIN_NUM_TOPICS_CONGRESS,
    MIN_NUM_TOPICS_SCOTUS,
    SCOTUS_DATA_FILE_POS_TAGGED,
    SCOTUS_MIN_DF,
    SCOTUS_TOPIC_MODEL,
)


def run_topic_modeling() -> None:
    """
    Runs topic modeling for both Congressional legislation and SCOTUS decisions.
    """
    # run topic modeling for Congressional legislation
    logging.info("Running topic modeling for Congressional legislation...")
    congress_topic_modeling = TopicModeling(
        file_path=CONGRESS_DATA_FILE_POS_TAGGED,
        save_name=CONGRESS_TOPIC_MODEL,
        topic_ranges=(MIN_NUM_TOPICS_CONGRESS, MAX_NUM_TOPICS_CONGRESS),
    )
    congress_topic_modeling.gen_topic_model()

    # run topic modeling for SCOTUS decisions
    logging.info("Running topic modeling for SCOTUS decisions...")
    scotus_topic_modeling = TopicModeling(
        file_path=SCOTUS_DATA_FILE_POS_TAGGED,
        save_name=SCOTUS_TOPIC_MODEL,
        topic_ranges=(MIN_NUM_TOPICS_SCOTUS, MAX_NUM_TOPICS_SCOTUS),
        min_df=SCOTUS_MIN_DF,
    )
    scotus_topic_modeling.gen_topic_model()
