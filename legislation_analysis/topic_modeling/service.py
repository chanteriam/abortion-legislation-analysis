"""
Services for topic modeling.
"""

import logging

from legislation_analysis.topic_modeling.topic_modeling import TopicModeling
from legislation_analysis.utils.constants import (
    CONGRESS_DATA_FILE_CLEANED,
    CONGRESS_DATA_FILE_TOPIC_MODELED_NAME,
    MAX_NUM_TOPICS_CONGRESS,
    MAX_NUM_TOPICS_SCOTUS,
    MIN_NUM_TOPICS_CONGRESS,
    MIN_NUM_TOPICS_SCOTUS,
    SCOTUS_DATA_FILE_CLEANED,
    SCOTUS_DATA_FILE_TOPIC_MODELED_NAME,
)


def run_topic_modeling() -> None:
    """
    Runs topic modeling for both Congressional legislation and SCOTUS decisions.
    """
    # run topic modeling for Congressional legislation
    logging.info("Running topic modeling for Congressional legislation...")
    congress_topic_modeling = TopicModeling(
        file_path=CONGRESS_DATA_FILE_CLEANED,
        save_name=CONGRESS_DATA_FILE_TOPIC_MODELED_NAME,
        topic_ranges=(MIN_NUM_TOPICS_CONGRESS, MAX_NUM_TOPICS_CONGRESS),
    )
    congress_topic_modeling.gen_topic_model()

    # run topic modeling for SCOTUS decisions
    logging.info("Running topic modeling for SCOTUS decisions...")
    scotus_topic_modeling = TopicModeling(
        file_path=SCOTUS_DATA_FILE_CLEANED,
        save_name=SCOTUS_DATA_FILE_TOPIC_MODELED_NAME,
        topic_ranges=(MIN_NUM_TOPICS_SCOTUS, MAX_NUM_TOPICS_SCOTUS),
    )
    scotus_topic_modeling.gen_topic_model()
