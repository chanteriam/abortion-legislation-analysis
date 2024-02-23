"""
Services for topic modeling.
"""

from legislation_analysis.topic_modeling.topic_modeling import TopicModeling
from legislation_analysis.utils.constants import (
    CONGRESS_DATA_FILE_CLEANED,
    CONGRESS_DATA_FILE_TOPIC_MODELED_NAME,
    SCOTUS_DATA_FILE_CLEANED,
    SCOTUS_DATA_FILE_TOPIC_MODELED_NAME,
    MIN_NUM_TOPICS_CONGRESS,
    MAX_NUM_TOPICS_CONGRESS,
    MIN_NUM_TOPICS_SCOTUS,
    MAX_NUM_TOPICS_SCOTUS,
)


def run_topic_modeling() -> None:
    """
    Runs topic modeling for both Congressional legislation and SCOTUS decisions.
    """
    # run topic modeling for Congressional legislation
    congress_topic_modeling = TopicModeling(
        file_path=CONGRESS_DATA_FILE_CLEANED,
        save_name=CONGRESS_DATA_FILE_TOPIC_MODELED_NAME,
        topic_ranges=(MIN_NUM_TOPICS_CONGRESS, MAX_NUM_TOPICS_CONGRESS),
    )
    congress_topic_modeling.gen_topic_model()

    # run topic modeling for SCOTUS decisions
    scotus_topic_modeling = TopicModeling(
        file_path=SCOTUS_DATA_FILE_CLEANED,
        save_name=SCOTUS_DATA_FILE_TOPIC_MODELED_NAME,
        topic_ranges=(MIN_NUM_TOPICS_SCOTUS, MAX_NUM_TOPICS_SCOTUS),
    )
    scotus_topic_modeling.gen_topic_model()
