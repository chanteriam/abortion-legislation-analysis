import logging
import os

from legislation_analysis.clustering.hierarchy_complete import HierarchyComplete
from legislation_analysis.clustering.knn import KNN
from legislation_analysis.utils.constants import PROCESSED_DATA_PATH


CONGRESS_CLUSTERED_FILE = "congress_legislation_clustered.fea"
CONGRESS_POS_TAGGED_FILE = os.path.join(
    PROCESSED_DATA_PATH, "congress_legislation_pos.fea"
)

SCOTUS_CLUSTERED_FILE = "scotus_cases_clustered.fea"
SCOTUS_POS_TAGGED_FILE = os.path.join(
    PROCESSED_DATA_PATH, "scotus_cases_pos.fea"
)


def run_hierarchy_complete_clustering() -> None:
    """
    Runs Hierarchy Complete clustering.
    """
    logging.info(
        "Starting Hierarchy Complete clustering for Congressional legislation."
    )
    # congress_hc = HierarchyComplete(
    #     CONGRESS_POS_TAGGED_FILE, CONGRESS_CLUSTERED_FILE
    # )
    # congress_hc.cluster_parts_of_speech()
    logging.info(
        "Finished Hierarchy Complete clustering for Congressional legislation."
    )

    logging.info("Starting Hierarchy Complete clustering for SCOTUS decisions.")
    scotus_hc = HierarchyComplete(SCOTUS_POS_TAGGED_FILE, SCOTUS_CLUSTERED_FILE)
    scotus_hc.cluster_parts_of_speech()
    logging.info("Finished Hierarchy Complete clustering for SCOTUS decisions.")


def run_hierarchy_ward_clustering() -> None:
    """
    Runs Hierarchy Ward clustering.
    """

    logging.info(
        "Starting Hierarchy Ward clustering for Congressional legislation."
    )
    # congress_hw = HierarchyComplete(
    #     CONGRESS_POS_TAGGED_FILE, CONGRESS_CLUSTERED_FILE
    # )
    # congress_hw.cluster_parts_of_speech()
    logging.info(
        "Finished Hierarchy Ward clustering for Congressional legislation."
    )

    logging.info("Starting Hierarchy Ward clustering for SCOTUS decisions.")
    scotus_hw = HierarchyComplete(SCOTUS_POS_TAGGED_FILE, SCOTUS_CLUSTERED_FILE)
    scotus_hw.cluster_parts_of_speech()
    logging.info("Finished Hierarchy Ward clustering for SCOTUS decisions.")


def run_knn_clustering() -> None:
    """
    Runs K-Nearest Neighbor clustering.
    """
    logging.info(
        "Starting K-Nearest Neighbor clustering for Congressional legislation."
    )
    # congress_knn = KNN(CONGRESS_POS_TAGGED_FILE, CONGRESS_CLUSTERED_FILE)
    # congress_knn.cluster_parts_of_speech()
    logging.info(
        "Finished K-Nearest Neighbor clustering for Congressional legislation."
    )

    logging.info("Starting K-Nearest Neighbor clustering for SCOTUS decisions.")
    scotus_knn = KNN(SCOTUS_POS_TAGGED_FILE, SCOTUS_CLUSTERED_FILE)
    scotus_knn.cluster_parts_of_speech()
    logging.info("Finished K-Nearest Neighbor clustering for SCOTUS decisions.")
