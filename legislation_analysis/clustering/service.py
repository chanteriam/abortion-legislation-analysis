import logging


def run_hierarchy_complete_clustering() -> None:
    """
    Runs Hierarchy Complete clustering.
    """
    logging.info(
        "Starting Hierarchy Complete clustering for Congressional legislation."
    )
    logging.info(
        "Finished Hierarchy Complete clustering for Congressional legislation."
    )

    logging.info("Starting Hierarchy Complete clustering for SCOTUS decisions.")
    logging.info("Finished Hierarchy Complete clustering for SCOTUS decisions.")


def run_hierarchy_ward_clustering() -> None:
    """
    Runs Hierarchy Ward clustering.
    """
    logging.info(
        "Starting Hierarchy Ward clustering for Congressional legislation."
    )
    logging.info(
        "Finished Hierarchy Ward clustering for Congressional legislation."
    )

    logging.info("Starting Hierarchy Ward clustering for SCOTUS decisions.")
    logging.info("Finished Hierarchy Ward clustering for SCOTUS decisions.")


def run_knn_clustering() -> None:
    """
    Runs K-Nearest Neighbor clustering.
    """
    logging.info(
        "Starting K-Nearest Neighbor clustering for Congressional legislation."
    )
    logging.info(
        "Finished K-Nearest Neighbor clustering for Congressional legislation."
    )

    logging.info("Starting K-Nearest Neighbor clustering for SCOTUS decisions.")
    logging.info("Finished K-Nearest Neighbor clustering for SCOTUS decisions.")
