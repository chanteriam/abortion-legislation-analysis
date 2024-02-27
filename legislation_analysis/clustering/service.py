import logging
import os.path

from legislation_analysis.clustering.hierarchy_complete import HierarchyComplete
from legislation_analysis.clustering.hierarchy_ward import HierarchyWard
from legislation_analysis.clustering.knn import KNN
from legislation_analysis.utils.constants import (
    CONGRESS_DATA_CLUSTERED_FILE,
    CONGRESS_DATA_CLUSTERED_FILE_NAME,
    CONGRESS_DATA_POS_TAGGED_FILE,
    SCOTUS_DATA_CLUSTERED_FILE,
    SCOTUS_DATA_CLUSTERED_FILE_NAME,
    SCOTUS_DATA_POS_TAGGED_FILE,
)


def run_hierarchy_complete_clustering() -> None:
    """
    Runs Hierarchy Complete clustering.
    """
    logging.info(
        "Starting Hierarchy Complete clustering for "
        "Congressional legislation..."
    )

    congress_hc = HierarchyComplete(
        CONGRESS_DATA_POS_TAGGED_FILE, CONGRESS_DATA_CLUSTERED_FILE_NAME
    )
    congress_hc.cluster_parts_of_speech()

    logging.info(
        "Finished Hierarchy Complete clustering for "
        "Congressional legislation..."
    )

    logging.info(
        "Starting Hierarchy Complete clustering for SCOTUS decisions..."
    )

    scotus_hc = HierarchyComplete(
        SCOTUS_DATA_POS_TAGGED_FILE, SCOTUS_DATA_CLUSTERED_FILE_NAME
    )
    scotus_hc.cluster_parts_of_speech()

    logging.info(
        "Finished Hierarchy Complete clustering for SCOTUS decisions..."
    )


def run_hierarchy_ward_clustering() -> None:
    """
    Runs Hierarchy Ward clustering.
    """

    logging.info(
        "Starting Hierarchy Ward clustering for Congressional legislation..."
    )

    # Appending Hierarchy Ward clusters on to pre-existing cluster file
    if os.path.isfile(CONGRESS_DATA_CLUSTERED_FILE):
        congress_src_file = CONGRESS_DATA_CLUSTERED_FILE
    else:
        congress_src_file = CONGRESS_DATA_POS_TAGGED_FILE

    congress_hw = HierarchyWard(
        congress_src_file, CONGRESS_DATA_CLUSTERED_FILE_NAME
    )
    congress_hw.cluster_parts_of_speech()

    logging.info(
        "Finished Hierarchy Ward clustering for Congressional legislation..."
    )

    logging.info("Starting Hierarchy Ward clustering for SCOTUS decisions...")

    # Appending Hierarchy Ward clusters on to pre-existing cluster file
    if os.path.isfile(SCOTUS_DATA_CLUSTERED_FILE):
        scotus_src_file = SCOTUS_DATA_CLUSTERED_FILE
    else:
        scotus_src_file = SCOTUS_DATA_POS_TAGGED_FILE

    scotus_hw = HierarchyWard(scotus_src_file, SCOTUS_DATA_CLUSTERED_FILE_NAME)
    scotus_hw.cluster_parts_of_speech()

    logging.info("Finished Hierarchy Ward clustering for SCOTUS decisions...")


def run_knn_clustering() -> None:
    """
    Runs K-Nearest Neighbor clustering.
    """
    logging.info(
        "Starting K-Nearest Neighbor clustering for "
        "Congressional legislation..."
    )

    # Appending K-Nearest Neighbor clusters on to pre-existing cluster file
    if os.path.isfile(CONGRESS_DATA_CLUSTERED_FILE):
        congress_src_file = CONGRESS_DATA_CLUSTERED_FILE
    else:
        congress_src_file = CONGRESS_DATA_POS_TAGGED_FILE

    congress_knn = KNN(congress_src_file, CONGRESS_DATA_CLUSTERED_FILE_NAME)
    congress_knn.cluster_parts_of_speech()

    logging.info(
        "Finished K-Nearest Neighbor clustering for "
        "Congressional legislation..."
    )

    logging.info(
        "Starting K-Nearest Neighbor clustering for SCOTUS decisions..."
    )

    # Appending K-Nearest Neighbor clusters on to pre-existing cluster file
    if os.path.isfile(SCOTUS_DATA_CLUSTERED_FILE):
        scotus_src_file = SCOTUS_DATA_CLUSTERED_FILE
    else:
        scotus_src_file = SCOTUS_DATA_POS_TAGGED_FILE

    scotus_knn = KNN(scotus_src_file, SCOTUS_DATA_CLUSTERED_FILE_NAME)
    scotus_knn.cluster_parts_of_speech()

    logging.info(
        "Finished K-Nearest Neighbor clustering for SCOTUS decisions..."
    )
