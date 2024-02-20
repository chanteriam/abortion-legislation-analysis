import logging

from legislation_analysis.clustering.hierarchy_complete import HierarchyComplete
from legislation_analysis.clustering.hierarchy_ward import HierarchyWard
from legislation_analysis.clustering.knn import KNN
from legislation_analysis.utils.constants import (
    CONGRESS_DATA_CLUSTERED_FILE_NAME,
    CONGRESS_DATA_POS_TAGGED_FILE,
    SCOTUS_DATA_FILE_CLUSTERED_NAME,
    SCOTUS_DATA_FILE_POS_TAGGED,
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
        SCOTUS_DATA_FILE_POS_TAGGED, SCOTUS_DATA_FILE_CLUSTERED_NAME
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
    congress_hw = HierarchyWard(
        CONGRESS_DATA_POS_TAGGED_FILE, CONGRESS_DATA_CLUSTERED_FILE_NAME
    )
    congress_hw.cluster_parts_of_speech()
    logging.info(
        "Finished Hierarchy Ward clustering for Congressional legislation..."
    )

    logging.info("Starting Hierarchy Ward clustering for SCOTUS decisions...")
    scotus_hw = HierarchyWard(
        SCOTUS_DATA_FILE_POS_TAGGED, SCOTUS_DATA_FILE_CLUSTERED_NAME
    )
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
    congress_knn = KNN(
        CONGRESS_DATA_POS_TAGGED_FILE, CONGRESS_DATA_CLUSTERED_FILE_NAME
    )
    congress_knn.cluster_parts_of_speech()
    logging.info(
        "Finished K-Nearest Neighbor clustering for "
        "Congressional legislation..."
    )

    logging.info(
        "Starting K-Nearest Neighbor clustering for SCOTUS decisions..."
    )
    scotus_knn = KNN(
        SCOTUS_DATA_FILE_POS_TAGGED, SCOTUS_DATA_FILE_CLUSTERED_NAME
    )
    scotus_knn.cluster_parts_of_speech()
    logging.info(
        "Finished K-Nearest Neighbor clustering for SCOTUS decisions..."
    )
