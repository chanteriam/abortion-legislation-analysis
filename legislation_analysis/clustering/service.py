import logging
import os.path

from legislation_analysis.clustering.hierarchy_complete import HierarchyComplete
from legislation_analysis.clustering.hierarchy_ward import HierarchyWard
from legislation_analysis.clustering.k_means import KMeansClustering
from legislation_analysis.utils.constants import (
    CONGRESS_DATA_CLUSTERED_FILE,
    CONGRESS_DATA_CLUSTERED_FILE_NAME,
    CONGRESS_DATA_FILE_POS_TAGGED,
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
        CONGRESS_DATA_FILE_POS_TAGGED, CONGRESS_DATA_CLUSTERED_FILE_NAME
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
        CONGRESS_DATA_FILE_POS_TAGGED, SCOTUS_DATA_FILE_CLUSTERED_NAME
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
        pass
    else:
        pass

    congress_hw = HierarchyWard(
        CONGRESS_DATA_FILE_POS_TAGGED, CONGRESS_DATA_CLUSTERED_FILE_NAME
    )
    congress_hw.cluster_parts_of_speech()

    logging.info(
        "Finished Hierarchy Ward clustering for Congressional legislation..."
    )

    logging.info("Starting Hierarchy Ward clustering for SCOTUS decisions...")

    # Appending Hierarchy Ward clusters on to pre-existing cluster file
    if os.path.isfile(SCOTUS_DATA_FILE_CLUSTERED_NAME):
        scotus_src_file = SCOTUS_DATA_FILE_CLUSTERED_NAME
    else:
        scotus_src_file = CONGRESS_DATA_FILE_POS_TAGGED

    scotus_hw = HierarchyWard(scotus_src_file, SCOTUS_DATA_FILE_CLUSTERED_NAME)
    scotus_hw.cluster_parts_of_speech()

    logging.info("Finished Hierarchy Ward clustering for SCOTUS decisions...")


def run_k_means_clustering() -> None:
    """
    Runs K-Means clustering.
    """
    logging.info(
        "Starting K-Means clustering for " "Congressional legislation..."
    )
    congress_k_means = KMeansClustering(
        CONGRESS_DATA_FILE_POS_TAGGED, CONGRESS_DATA_CLUSTERED_FILE_NAME
    )
    congress_k_means.cluster_parts_of_speech()
    logging.info(
        "Finished K-Means clustering for " "Congressional legislation..."
    )

    logging.info("Starting K-Means clustering for SCOTUS decisions...")
    scotus_k_means = KMeansClustering(
        SCOTUS_DATA_FILE_POS_TAGGED, SCOTUS_DATA_FILE_CLUSTERED_NAME
    )
    scotus_k_means.cluster_parts_of_speech()
    logging.info("Finished K-Means clustering for SCOTUS decisions...")
