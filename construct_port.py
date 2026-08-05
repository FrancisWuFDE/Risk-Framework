"""Load portfolio shares and prices from a dated CSV file."""

from __future__ import annotations

import argparse
import re
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from bloomberg_cache import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_RETRIES,
    PRICE_DATABASE,
    get_prices_for_date,
)


PORTFOLIO_TYPES = (
    "US_live_port",
    "long_positions",
    "short_positions",
)
PORTFOLIO_FILENAME_PATTERN = re.compile(
    r"^(?P<portfolio_type>US_live_port|long_positions|short_positions)_"
    r"(?P<date>\d{8})\.csv$"
)
PORTFOLIO_DATA_DIR = Path(__file__).resolve().parent / "port_data"
REQUIRED_COLUMNS = {"ticker", "shares"}


def _validate_portfolio_filename(csv_file: str | Path) -> Path:
    """Validate the portfolio filename and its YYYYMMDD date."""
    csv_path = Path(csv_file)
    match = PORTFOLIO_FILENAME_PATTERN.fullmatch(csv_path.name)

    if match is None:
        raise ValueError(
            "CSV filename must match 'US_live_port_YYYYMMDD.csv', "
            "'long_positions_YYYYMMDD.csv', or "
            "'short_positions_YYYYMMDD.csv'."
        )

    try:
        datetime.strptime(match.group("date"), "%Y%m%d")
    except ValueError as error:
        raise ValueError(
            "CSV filename must contain a valid date in YYYYMMDD format."
        ) from error

    return csv_path


def resolve_portfolio_path(csv_file: str | Path) -> Path:
    """Resolve a bare portfolio filename inside the port_data directory."""
    csv_path = _validate_portfolio_filename(csv_file)
    if not csv_path.is_absolute() and csv_path.parent == Path():
        csv_path = PORTFOLIO_DATA_DIR / csv_path

    if not csv_path.is_file():
        raise FileNotFoundError(f"Portfolio CSV does not exist: {csv_path}.")

    return csv_path


def get_portfolio_date(csv_file: str | Path) -> date:
    """Return the date encoded in a validated portfolio filename."""
    csv_path = _validate_portfolio_filename(csv_file)
    match = PORTFOLIO_FILENAME_PATTERN.fullmatch(csv_path.name)
    if match is None:
        raise ValueError(f"Invalid portfolio filename: {csv_path.name}.")

    return datetime.strptime(match.group("date"), "%Y%m%d").date()


def get_portfolio_type(csv_file: str | Path) -> str:
    """Return the portfolio type encoded in a validated filename."""
    csv_path = _validate_portfolio_filename(csv_file)
    match = PORTFOLIO_FILENAME_PATTERN.fullmatch(csv_path.name)
    if match is None:
        raise ValueError(f"Invalid portfolio filename: {csv_path.name}.")

    return match.group("portfolio_type")


def get_portfolio_output_filename(
    prefix: str,
    portfolio_type: str,
    as_of_date: date,
) -> str:
    """Return a dated output filename that distinguishes split portfolios."""
    if portfolio_type not in PORTFOLIO_TYPES:
        supported_types = ", ".join(PORTFOLIO_TYPES)
        raise ValueError(
            f"Unsupported portfolio type '{portfolio_type}'. "
            f"Expected one of: {supported_types}."
        )

    portfolio_suffix = (
        "" if portfolio_type == "US_live_port" else f"_{portfolio_type}"
    )
    return f"{prefix}{portfolio_suffix}_{as_of_date:%Y%m%d}.csv"


def _read_portfolio_csv(csv_file: str | Path) -> pd.DataFrame:
    """Read and validate the columns shared by portfolio series loaders."""
    csv_path = resolve_portfolio_path(csv_file)
    portfolio = pd.read_csv(csv_path)
    portfolio.columns = portfolio.columns.str.strip().str.lower()

    missing_columns = REQUIRED_COLUMNS.difference(portfolio.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"CSV file is missing required columns: {missing}.")

    if portfolio["ticker"].isna().any():
        raise ValueError("The ticker column cannot contain missing values.")

    portfolio["ticker"] = portfolio["ticker"].astype(str).str.strip()
    if portfolio["ticker"].eq("").any():
        raise ValueError("The ticker column cannot contain blank values.")

    if portfolio["ticker"].duplicated().any():
        duplicates = portfolio.loc[
            portfolio["ticker"].duplicated(keep=False),
            "ticker",
        ].unique()
        duplicate_list = ", ".join(duplicates)
        raise ValueError(f"Ticker values must be unique: {duplicate_list}.")

    return portfolio


def load_shares(csv_file: str | Path) -> pd.Series:
    """Return portfolio shares indexed by ticker."""
    portfolio = _read_portfolio_csv(csv_file)
    shares = pd.to_numeric(portfolio["shares"], errors="raise")
    shares.index = portfolio["ticker"]
    shares.index.name = "ticker"
    shares.name = "shares"
    return shares


def load_prices(
    csv_file: str | Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_retries: int = DEFAULT_MAX_RETRIES,
    database: str | Path = PRICE_DATABASE,
    allow_download: bool = True,
) -> pd.Series:
    """Return dated Bloomberg closes from the SQLite price cache."""
    portfolio = _read_portfolio_csv(csv_file)
    tickers = pd.Index(portfolio["ticker"], name="ticker")
    return get_prices_for_date(
        tickers=tickers,
        as_of_date=get_portfolio_date(csv_file),
        batch_size=batch_size,
        max_retries=max_retries,
        database=database,
        allow_download=allow_download,
    )


def main() -> None:
    """Load a portfolio CSV and display its shares and prices."""
    parser = argparse.ArgumentParser(
        description="Load shares and prices from a dated portfolio CSV."
    )
    parser.add_argument(
        "csv_file",
        type=Path,
        help=(
            "Portfolio filename in port_data, or an explicit path to a "
            "full, long-only, or short-only dated portfolio CSV."
        ),
    )
    args = parser.parse_args()

    portfolio = pd.concat(
        [
            load_shares(args.csv_file),
            load_prices(args.csv_file),
        ],
        axis="columns",
    )
    print(portfolio)


if __name__ == "__main__":
    main()
