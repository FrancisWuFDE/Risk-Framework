from __future__ import annotations

import argparse
import math
import re
import warnings
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    GoodFriday,
    Holiday,
    USLaborDay,
    USMartinLutherKingJr,
    USMemorialDay,
    USPresidentsDay,
    USThanksgivingDay,
    nearest_workday,
    sunday_to_monday,
)
from pandas.tseries.offsets import CustomBusinessDay


PORTFOLIO_FILENAME_PATTERN = re.compile(
    r"^US_live_port_(?P<date>\d{8})\.csv$"
)
REQUIRED_COLUMNS = {"ticker", "shares", "price"}


class NyseHolidayCalendar(AbstractHolidayCalendar):
    """Represent regular full-day NYSE holidays."""

    rules = [
        Holiday(
            "New Year's Day",
            month=1,
            day=1,
            observance=sunday_to_monday,
        ),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday(
            "Juneteenth",
            month=6,
            day=19,
            start_date="2022-06-19",
            observance=nearest_workday,
        ),
        Holiday(
            "Independence Day",
            month=7,
            day=4,
            observance=nearest_workday,
        ),
        USLaborDay,
        USThanksgivingDay,
        Holiday(
            "Christmas Day",
            month=12,
            day=25,
            observance=nearest_workday,
        ),
    ]


NYSE_TRADING_DAY = CustomBusinessDay(calendar=NyseHolidayCalendar())


@dataclass(frozen=True)
class PortfolioReturnResult:
    """Store a close-to-close portfolio return calculation."""

    holding_date: date
    return_date: date
    previous_csv: Path
    current_csv: Path
    profit_and_loss: float
    capital: float
    portfolio_return: float
    priced_tickers: int
    prior_tickers: int
    missing_tickers: tuple[str, ...]


def get_portfolio_date(csv_file: str | Path) -> date:
    """Return the YYYYMMDD date encoded in a portfolio filename."""
    csv_path = Path(csv_file)
    match = PORTFOLIO_FILENAME_PATTERN.fullmatch(csv_path.name)
    if match is None:
        raise ValueError(
            "Portfolio filename must match "
            "'US_live_port_YYYYMMDD.csv'."
        )

    try:
        return datetime.strptime(match.group("date"), "%Y%m%d").date()
    except ValueError as error:
        raise ValueError(
            "Portfolio filename must contain a valid YYYYMMDD date."
        ) from error


def find_previous_portfolio(current_csv: str | Path) -> Path:
    """Return the immediately preceding NYSE portfolio in the directory."""
    current_path = Path(current_csv).resolve()
    current_date = get_portfolio_date(current_path)
    previous_date = (
        pd.Timestamp(current_date) - NYSE_TRADING_DAY
    ).date()
    previous_path = current_path.with_name(
        f"US_live_port_{previous_date:%Y%m%d}.csv"
    )
    if not previous_path.is_file():
        raise FileNotFoundError(
            "Previous trading-day portfolio does not exist: "
            f"{previous_path}."
        )

    return previous_path


def load_portfolio(csv_file: str | Path) -> pd.DataFrame:
    """Load ticker, share, and closing-price columns from a portfolio."""
    csv_path = Path(csv_file).resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"Portfolio CSV does not exist: {csv_path}.")

    portfolio = pd.read_csv(csv_path)
    portfolio.columns = portfolio.columns.astype(str).str.strip().str.lower()
    if "price" not in portfolio.columns:
        unnamed_columns = [
            column
            for column in portfolio.columns
            if column.startswith("unnamed:")
        ]
        if len(unnamed_columns) == 1:
            portfolio = portfolio.rename(
                columns={unnamed_columns[0]: "price"}
            )

    missing_columns = REQUIRED_COLUMNS.difference(portfolio.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Portfolio is missing columns: {missing}.")

    portfolio = portfolio.loc[:, ["ticker", "shares", "price"]].copy()
    portfolio = portfolio.dropna(how="all")
    if portfolio["ticker"].isna().any():
        raise ValueError("Portfolio position rows cannot have blank tickers.")
    portfolio["ticker"] = (
        portfolio["ticker"].astype(str).str.strip().str.upper()
    )
    if portfolio["ticker"].eq("").any():
        raise ValueError("Portfolio tickers cannot be blank.")
    if portfolio["ticker"].duplicated().any():
        duplicates = portfolio.loc[
            portfolio["ticker"].duplicated(keep=False),
            "ticker",
        ].astype(str).unique()
        raise ValueError(
            "Portfolio tickers must be unique: "
            f"{', '.join(duplicates[:10])}."
        )

    portfolio["shares"] = pd.to_numeric(
        portfolio["shares"],
        errors="raise",
    )
    portfolio["price"] = pd.to_numeric(
        portfolio["price"],
        errors="raise",
    )
    if portfolio[["shares", "price"]].isna().any().any():
        raise ValueError("Portfolio shares and prices cannot be missing.")
    if portfolio["price"].le(0).any():
        raise ValueError("Portfolio prices must be positive.")

    return portfolio.set_index("ticker").sort_index()


def calculate_portfolio_return(
    current_csv: str | Path,
    previous_csv: str | Path | None = None,
    capital: float | None = None,
    strict: bool = False,
) -> PortfolioReturnResult:
    """Calculate prior holdings' close-to-close return.

    The numerator is prior shares multiplied by the change from the prior
    close to the current close. The denominator defaults to prior gross
    market value; pass ``capital`` to use a fixed capital amount.
    """
    current_path = Path(current_csv).resolve()
    previous_path = (
        find_previous_portfolio(current_path)
        if previous_csv is None
        else Path(previous_csv).resolve()
    )
    current_date = get_portfolio_date(current_path)
    previous_date = get_portfolio_date(previous_path)
    expected_previous_date = (
        pd.Timestamp(current_date) - NYSE_TRADING_DAY
    ).date()
    if previous_date != expected_previous_date:
        raise ValueError(
            f"Expected previous NYSE trading date {expected_previous_date}, "
            f"but received {previous_date}."
        )

    current = load_portfolio(current_path)
    previous = load_portfolio(previous_path)
    current_prices = current["price"].reindex(previous.index)
    missing_tickers = tuple(
        current_prices.index[current_prices.isna()].astype(str)
    )
    if missing_tickers and strict:
        preview = ", ".join(missing_tickers[:10])
        if len(missing_tickers) > 10:
            preview = f"{preview}, ..."
        raise ValueError(
            f"Current CSV has no price for {len(missing_tickers)} prior "
            f"holding(s): {preview}."
        )
    if missing_tickers:
        warnings.warn(
            f"Current CSV has no price for {len(missing_tickers)} prior "
            "holding(s). Their P&L is excluded; use --strict to reject "
            "incomplete coverage.",
            RuntimeWarning,
            stacklevel=2,
        )

    valid = current_prices.notna()
    if not valid.any():
        raise ValueError("The two portfolios have no overlapping tickers.")

    prior_market_values = previous["shares"].mul(previous["price"])
    prior_gross_market_value = float(prior_market_values.abs().sum())
    capital_used = (
        prior_gross_market_value
        if capital is None
        else float(capital)
    )
    if not math.isfinite(capital_used) or capital_used <= 0:
        raise ValueError(
            "Capital must be positive and finite. Supply --capital when "
            "the previous gross market value is zero."
        )

    price_changes = current_prices.loc[valid].sub(
        previous.loc[valid, "price"]
    )
    profit_and_loss = float(
        previous.loc[valid, "shares"].mul(price_changes).sum()
    )
    portfolio_return = profit_and_loss / capital_used

    return PortfolioReturnResult(
        holding_date=previous_date,
        return_date=current_date,
        previous_csv=previous_path,
        current_csv=current_path,
        profit_and_loss=profit_and_loss,
        capital=capital_used,
        portfolio_return=portfolio_return,
        priced_tickers=int(valid.sum()),
        prior_tickers=len(previous),
        missing_tickers=missing_tickers,
    )


def format_return_result(result: PortfolioReturnResult) -> str:
    """Return a readable portfolio-return summary."""
    rows = [
        ("Holdings date", result.holding_date.isoformat()),
        ("Return date", result.return_date.isoformat()),
        ("Previous portfolio", str(result.previous_csv)),
        ("Current portfolio", str(result.current_csv)),
        (
            "Price coverage",
            f"{result.priced_tickers:,}/{result.prior_tickers:,}",
        ),
        ("Daily P&L", f"${result.profit_and_loss:,.2f}"),
        ("Capital", f"${result.capital:,.2f}"),
        ("Return decimal", f"{result.portfolio_return:.10f}"),
        ("Portfolio return", f"{result.portfolio_return:.8%}"),
    ]
    label_width = max(len(label) for label, _ in rows)
    return "\n".join(
        f"{label:<{label_width}} : {value}"
        for label, value in rows
    )


def main() -> None:
    """Parse command-line arguments and print a daily portfolio return."""
    parser = argparse.ArgumentParser(
        description=(
            "Calculate close-to-close return using previous-day holdings "
            "and prices from two dated portfolio CSVs."
        )
    )
    parser.add_argument(
        "current_csv",
        type=Path,
        help="Current dated portfolio CSV containing current closing prices.",
    )
    parser.add_argument(
        "--previous-csv",
        type=Path,
        help=(
            "Previous trading-day portfolio CSV. By default, it is found "
            "in the current CSV's directory."
        ),
    )
    parser.add_argument(
        "--capital",
        type=float,
        help=(
            "Return denominator. Defaults to previous gross market value; "
            "use 300000000 for fixed $300M capital."
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any prior holding lacks a current price.",
    )
    args = parser.parse_args()

    result = calculate_portfolio_return(
        current_csv=args.current_csv,
        previous_csv=args.previous_csv,
        capital=args.capital,
        strict=args.strict,
    )
    print(format_return_result(result))


if __name__ == "__main__":
    main()
