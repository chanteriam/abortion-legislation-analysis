"""
Services for topic modeling.
"""

import logging
import os

from legislation_analysis.topic_modeling.dynamic_topic_modeling import (
    DynamicTopicModeling,
)
from legislation_analysis.topic_modeling.topic_modeling import TopicModeling
from legislation_analysis.utils.constants import (
    CONGRESS_DATA_FILE_POS_TAGGED,
    CONGRESS_TOPIC_MODEL,
    MAX_NUM_TOPICS_CONGRESS,
    MAX_NUM_TOPICS_SCOTUS,
    MIN_NUM_TOPICS_CONGRESS,
    MIN_NUM_TOPICS_SCOTUS,
    MODELED_DATA_PATH,
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
    congress_model_fp = (
        os.path.join(MODELED_DATA_PATH, CONGRESS_TOPIC_MODEL)
        if os.path.exists(os.path.join(MODELED_DATA_PATH, CONGRESS_TOPIC_MODEL))
        else None
    )
    congress_topic_modeling = TopicModeling(
        file_path=CONGRESS_DATA_FILE_POS_TAGGED,
        save_name=CONGRESS_TOPIC_MODEL,
        topic_ranges=(MIN_NUM_TOPICS_CONGRESS, MAX_NUM_TOPICS_CONGRESS),
        model_fp=congress_model_fp,
    )
    congress_topic_modeling.gen_topic_model()

    # run topic modeling for SCOTUS decisions
    logging.info("Running topic modeling for SCOTUS decisions...")
    scotus_model_fp = (
        os.path.join(MODELED_DATA_PATH, SCOTUS_TOPIC_MODEL)
        if os.path.exists(os.path.join(MODELED_DATA_PATH, SCOTUS_TOPIC_MODEL))
        else None
    )
    scotus_topic_modeling = TopicModeling(
        file_path=SCOTUS_DATA_FILE_POS_TAGGED,
        save_name=SCOTUS_TOPIC_MODEL,
        topic_ranges=(MIN_NUM_TOPICS_SCOTUS, MAX_NUM_TOPICS_SCOTUS),
        min_df=SCOTUS_MIN_DF,
        model_fp=scotus_model_fp,
    )
    scotus_topic_modeling.gen_topic_model()


def run_dynamic_topic_modeling() -> None:
    """
    Runs dynamic topic modeling for Congressional legislation.
    """
    logging.info(
        """Running dynamic topic modeling for Congressional
                 legislation..."""
    )
    model_fp = (
        os.path.join(MODELED_DATA_PATH, f"{CONGRESS_TOPIC_MODEL}_dynamic")
        if os.path.exists(
            os.path.join(MODELED_DATA_PATH, f"{CONGRESS_TOPIC_MODEL}_dynamic")
        )
        else None
    )
    dynamic_topic_modeling = DynamicTopicModeling(
        file_path=CONGRESS_DATA_FILE_POS_TAGGED,
        save_name=CONGRESS_TOPIC_MODEL,
        model_fp=model_fp,
    )
    dynamic_topic_modeling.gen_topic_model()
