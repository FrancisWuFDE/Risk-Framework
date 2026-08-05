"""Split a dated portfolio CSV into long and short position files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from construct_port import (
    get_portfolio_date,
    get_portfolio_type,
    load_shares,
    resolve_portfolio_path,
)


def split_portfolio(
    portfolio_csv: str | Path,
    output_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    """Write separate long and short CSVs and return their paths."""
    portfolio_path = resolve_portfolio_path(portfolio_csv)
    if get_portfolio_type(portfolio_path) != "US_live_port":
        raise ValueError(
            "Only a US_live_port_YYYYMMDD.csv file can be divided."
        )

    portfolio = pd.read_csv(portfolio_path)
    shares = load_shares(portfolio_path)

    if shares.isna().any():
        raise ValueError("The shares column cannot contain missing values.")

    destination = (
        portfolio_path.parent
        if output_dir is None
        else Path(output_dir).resolve()
    )
    destination.mkdir(parents=True, exist_ok=True)

    portfolio_date = get_portfolio_date(portfolio_path)
    long_path = destination / f"long_positions_{portfolio_date:%Y%m%d}.csv"
    short_path = destination / f"short_positions_{portfolio_date:%Y%m%d}.csv"

    long_positions = portfolio.loc[shares.to_numpy() > 0].copy()
    short_positions = portfolio.loc[shares.to_numpy() < 0].copy()
    long_positions["Port"] = "long_positions"
    short_positions["Port"] = "short_positions"

    long_positions.to_csv(long_path, index=False)
    short_positions.to_csv(short_path, index=False)

    return long_path, short_path


def main() -> None:
    """Split one portfolio CSV and print the generated file locations."""
    parser = argparse.ArgumentParser(
        description="Split a portfolio CSV into long and short positions."
    )
    parser.add_argument(
        "portfolio_csv",
        type=Path,
        help=(
            "Portfolio filename in port_data, or an explicit path to a "
            "US_live_port_YYYYMMDD.csv file."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory; defaults to the input portfolio directory.",
    )
    args = parser.parse_args()

    long_path, short_path = split_portfolio(
        portfolio_csv=args.portfolio_csv,
        output_dir=args.output_dir,
    )

    long_count = len(pd.read_csv(long_path))
    short_count = len(pd.read_csv(short_path))
    print(f"Long positions ({long_count:,}) written to: {long_path}")
    print(f"Short positions ({short_count:,}) written to: {short_path}")


if __name__ == "__main__":
    main()
