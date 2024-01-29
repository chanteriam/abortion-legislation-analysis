"""Serves as the entry point for the project module."""

import argparse

from legislation_analysis.api import congress, scotus
from legislation_analysis.processing import clean, tokenize


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
        "--verbose",
        "-v",
        action="store_true",
        help="prints status updates",
    )

    args = parser.parse_args()

    if args.congress:
        congress.main(args.verbose)

    if args.scotus:
        scotus.main(args.verbose)

    if args.clean:
        clean.main(args.verbose)

    if args.tokenize:
        tokenize.main(args.verbose)


if __name__ == "__main__":
    main()
