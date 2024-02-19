"""Serves as the entry point for the project module."""

import argparse
import logging

from legislation_analysis.api.service import (
    download_congress_data,
    download_scotus_data,
)
from legislation_analysis.clustering.service import (
    run_hierarchy_complete_clustering,
    run_hierarchy_ward_clustering,
    run_knn_clustering,
)
from legislation_analysis.processing.service import (
    run_data_cleaner,
    run_data_tokenizer,
    run_pos_tagger,
)


logging.basicConfig(
    format="%(asctime)s: %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p"
)


def main() -> None:
    """
    Collects and runs command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--congress",
        action="store_true",
        help="pulls legislation text from api.congress.gov",
    )

    parser.add_argument(
        "--scotus",
        action="store_true",
        help="pulls legislation text from SCOTUS decisions",
    )

    parser.add_argument(
        "--clean",
        action="store_true",
        help="cleans the data",
    )

    parser.add_argument(
        "--tokenize",
        action="store_true",
        help="tokenize the data",
    )

    parser.add_argument(
        "--pos-tag",
        action="store_true",
        help="PoS tag the data",
    )

    parser.add_argument(
        "--cluster",
        action="store_true",
        help="cluster the data",
    )

    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="print debugging messages",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    if args.congress:
        download_congress_data()

    if args.scotus:
        download_scotus_data()

    if args.clean:
        run_data_cleaner()

    if args.tokenize:
        run_data_tokenizer()

    if args.pos_tag:
        run_pos_tagger()

    if args.cluster:
        run_hierarchy_complete_clustering()
        run_hierarchy_ward_clustering()
        run_knn_clustering()


if __name__ == "__main__":
    main()
