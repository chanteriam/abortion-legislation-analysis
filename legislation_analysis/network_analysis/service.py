"""
Implements the NetworkAnalysis class, which applies graph-based analysis to
legislation data.
"""

import logging
import os

import pandas as pd

from legislation_analysis.network_analysis.network import NetworkAnalysis
from legislation_analysis.utils.constants import (
    CONGRESS_DATA_FILE_NER_NAME,
    NETWORK_DATA_PATH,
    PROCESSED_DATA_PATH,
    SCOTUS_DATA_FILE_NER_NAME,
)
from legislation_analysis.utils.functions import save_df_to_file


def run_network_analysis() -> None:
    """
    Runs network analysis on the POS-tagged data.
    """
    logging.debug("Starting network analysis...")
    legislation_network = NetworkAnalysis(
        congress_file=os.path.join(
            PROCESSED_DATA_PATH, CONGRESS_DATA_FILE_NER_NAME
        ),
        scotus_file=os.path.join(
            PROCESSED_DATA_PATH, SCOTUS_DATA_FILE_NER_NAME
        ),
    )
    legislation_network.process()
    logging.debug("Saving network analysis results...")
    congr_ref_matrix = pd.DataFrame(legislation_network.congress_refs)
    congr_ref_matrix.columns = [
        "congress_" + str(i) for i in range(len(congr_ref_matrix.columns))
    ]

    scotus_ref_matrix = pd.DataFrame(legislation_network.scotus_refs)
    scotus_ref_matrix.columns = [
        "scotus_" + str(i) for i in range(len(scotus_ref_matrix.columns))
    ]

    congr_scotus_ref_matrix = pd.DataFrame(
        legislation_network.congress_scotus_refs
    )
    congr_scotus_ref_matrix.columns = [
        "scotus_" + str(i) for i in range(len(congr_scotus_ref_matrix.columns))
    ]

    save_df_to_file(
        congr_ref_matrix,
        os.path.join(
            NETWORK_DATA_PATH, "congress_congress_references_matrix.fea"
        ),
    )

    save_df_to_file(
        scotus_ref_matrix,
        os.path.join(
            NETWORK_DATA_PATH, "congress_scotus_references_matrix.fea"
        ),
    )

    save_df_to_file(
        congr_scotus_ref_matrix,
        os.path.join(NETWORK_DATA_PATH, "scotus_scotus_references_matrix.fea"),
    )
