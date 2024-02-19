"""Serves as the entry point for the project module."""

import argparse
import logging

from legislation_analysis.api import congress, scotus
from legislation_analysis.processing import clean, tokenizer


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
        "--debug",
        "-v",
        action="store_true",
        help="print debugging messages",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    if args.congress:
        congress.main()

    if args.scotus:
        scotus.main()

    if args.clean:
        clean.main()

    if args.tokenize:
        tokenizer.main()


if __name__ == "__main__":
    main()
