"""
Services for topic modeling.
"""

import logging
import os

from legislation_analysis.topic_modeling.topic_modeling import TopicModeling
from legislation_analysis.utils.constants import (
    SCOTUS_DATA_FILE_TOPIC_MODELED_NAME,
    MODELED_DATA_PATH,
    CONGRESS_DATA_FILE_TOPIC_MODELED_NAME,
    CONGRESS_DATA_POS_TAGGED_FILE,
    SCOTUS_DATA_FILE_POS_TAGGED,
)


def run_topic_modeling() -> None:
    """
    Runs topic modeling for both Congressional legislation and SCOTUS decisions.
    """
    # run topic modeling for Congressional legislation
    logging.info("Starting topic modeling for Congressional legislation...")
    congress_tm = TopicModeling(
        CONGRESS_DATA_POS_TAGGED_FILE,
        os.path.join(MODELED_DATA_PATH, CONGRESS_DATA_FILE_TOPIC_MODELED_NAME),
        "text_pos_tags_of_interest",
    )
    congress_tm.gen_topic_model(True)

    # run topic modeling for SCOTUS decisions
    logging.info("Starting topic modeling for SCOTUS decisions...")
    scotus_tm = TopicModeling(
        SCOTUS_DATA_FILE_POS_TAGGED,
        os.path.join(MODELED_DATA_PATH, SCOTUS_DATA_FILE_TOPIC_MODELED_NAME),
        "text_pos_tags_of_interest",
    )
    scotus_tm.gen_topic_model(True)
