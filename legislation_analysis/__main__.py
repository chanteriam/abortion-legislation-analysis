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
    run_k_means_clustering,
)
from legislation_analysis.processing.service import (
    run_data_cleaner,
    run_data_tokenizer,
    run_ner,
    run_pos_tagger,
)
from legislation_analysis.topic_modeling.service import (
    run_dynamic_topic_modeling,
    run_topic_modeling,
)


def main() -> None:
    """
    Collects and runs command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--all",
        "-a",
        action="store_true",
        help="runs all applications processes",
    )

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
        "--ner",
        action="store_true",
        help="NER the data",
    )

    parser.add_argument(
        "--cluster",
        action="store_true",
        help="cluster the data",
    )

    parser.add_argument(
        "--model",
        action="store_true",
        help="model the data",
    )

    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="print debugging messages",
    )

    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s: %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
        level=logging.DEBUG if args.debug else logging.INFO,
    )

    if args.all or args.congress:
        download_congress_data()

    if args.all or args.scotus:
        download_scotus_data()

    if args.all or args.clean:
        run_data_cleaner()

    if args.all or args.tokenize:
        run_data_tokenizer()

    if args.all or args.pos_tag:
        run_pos_tagger()

    if args.all or args.ner:
        run_ner()

    if args.all or args.cluster:
        run_hierarchy_complete_clustering()
        run_hierarchy_ward_clustering()
        run_k_means_clustering()

    if args.all or args.model:
        run_topic_modeling()
        run_dynamic_topic_modeling()


if __name__ == "__main__":
    main()
