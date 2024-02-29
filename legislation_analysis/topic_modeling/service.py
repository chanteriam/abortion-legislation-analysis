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
    CONGRESS_DATA_POS_TAGGED_FILE,
    CONGRESS_DYN_TOPIC_MODEL_FILE,
    CONGRESS_DYN_TOPIC_MODEL_FILE_NAME,
    CONGRESS_TOPIC_MODEL_FILE,
    CONGRESS_TOPIC_MODEL_FILE_NAME,
    MAX_NUM_TOPICS_CONGRESS,
    MAX_NUM_TOPICS_SCOTUS,
    MIN_NUM_TOPICS_CONGRESS,
    MIN_NUM_TOPICS_SCOTUS,
    SCOTUS_DATA_POS_TAGGED_FILE,
    SCOTUS_MIN_DF,
    SCOTUS_TOPIC_MODEL_FILE,
    SCOTUS_TOPIC_MODEL_FILE_NAME,
)


def run_topic_modeling() -> None:
    """
    Runs topic modeling for both Congressional legislation and SCOTUS decisions.
    """
    # run topic modeling for Congressional legislation
    logging.info("Running topic modeling for Congressional legislation...")
    congress_model_fp = (
        CONGRESS_TOPIC_MODEL_FILE
        if os.path.exists(CONGRESS_TOPIC_MODEL_FILE)
        else None
    )
    congress_topic_modeling = TopicModeling(
        file_path=CONGRESS_DATA_POS_TAGGED_FILE,
        save_name=CONGRESS_TOPIC_MODEL_FILE_NAME,
        topic_ranges=(MIN_NUM_TOPICS_CONGRESS, MAX_NUM_TOPICS_CONGRESS),
        model_fp=congress_model_fp,
    )
    congress_topic_modeling.gen_topic_model()

    # run topic modeling for SCOTUS decisions
    logging.info("Running topic modeling for SCOTUS decisions...")
    scotus_model_fp = (
        SCOTUS_TOPIC_MODEL_FILE
        if os.path.exists(SCOTUS_TOPIC_MODEL_FILE)
        else None
    )
    scotus_topic_modeling = TopicModeling(
        file_path=SCOTUS_DATA_POS_TAGGED_FILE,
        save_name=SCOTUS_TOPIC_MODEL_FILE_NAME,
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
        "Running dynamic topic modeling for Congressional legislation..."
    )
    model_fp = (
        CONGRESS_DYN_TOPIC_MODEL_FILE
        if os.path.exists(CONGRESS_DYN_TOPIC_MODEL_FILE)
        else None
    )
    dynamic_topic_modeling = DynamicTopicModeling(
        file_path=CONGRESS_DATA_POS_TAGGED_FILE,
        save_name=CONGRESS_DYN_TOPIC_MODEL_FILE_NAME,
        model_fp=model_fp,
    )
    dynamic_topic_modeling.gen_topic_model()
