"""Create a daily summary for a US live equity portfolio."""

from __future__ import annotations

import argparse
import math
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
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

from construct_port import (
    PORTFOLIO_DATA_DIR,
    get_portfolio_date,
    get_portfolio_output_filename,
    get_portfolio_type,
    load_prices,
    load_shares,
    resolve_portfolio_path,
)
from bloomberg_cache import (
    BLOOMBERG_DATABASE,
    DEFAULT_FACTOR_UNIVERSE,
    ensure_price_history,
    get_historical_market_caps,
    get_index_members,
    get_price_history,
    get_security_metadata,
    import_factor_universe_workbook,
)


DEFAULT_BATCH_SIZE = 50
DEFAULT_MAX_RETRIES = 3
DEFAULT_ADV_LOOKBACK_DAYS = 20
DEFAULT_LIQUIDATION_PARTICIPATION_RATE = 0.10
HISTORY_CALENDAR_MULTIPLIER = 2.25
MarketCapProvider = Callable[[pd.Index, pd.Series, date], pd.Series]


class NyseHolidayCalendar(AbstractHolidayCalendar):
    """Represent regular full-day NYSE market holidays."""

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
class PortfolioSummary:
    """Store portfolio-level metrics and ticker-level share changes."""

    as_of_date: date
    previous_date: date
    portfolio_type: str
    number_of_stocks: int
    weighted_average_market_cap: float
    ticker_coverage: int
    capital: float
    daily_return: float
    daily_turnover: float
    market_value: float
    market_value_change: float
    position_changes: pd.Series
    market_caps: pd.Series
    adv_lookback_days: int
    median_adv: float
    median_dollar_adv: float
    median_position_pct_adv: float
    maximum_position_pct_adv: float
    liquidation_participation_rate: float
    median_days_to_liquidate: float
    maximum_days_to_liquidate: float
    adv_by_position: pd.DataFrame


def find_previous_portfolio(today_csv: str | Path) -> Path:
    """Return the portfolio from the immediately preceding NYSE session."""
    today_path = resolve_portfolio_path(today_csv)
    today_date = get_portfolio_date(today_path)
    portfolio_type = get_portfolio_type(today_path)
    previous_date = (pd.Timestamp(today_date) - NYSE_TRADING_DAY).date()
    previous_path = today_path.with_name(
        f"{portfolio_type}_{previous_date:%Y%m%d}.csv"
    )

    if not previous_path.is_file():
        raise FileNotFoundError(
            "Previous trading-day portfolio does not exist: "
            f"{previous_path}."
        )

    return previous_path


def calculate_position_changes(
    today_shares: pd.Series,
    previous_shares: pd.Series,
) -> pd.Series:
    """Return nonzero share changes across both portfolios."""
    all_tickers = today_shares.index.union(previous_shares.index)
    aligned_today_shares = today_shares.reindex(
        all_tickers,
        fill_value=0,
    )
    aligned_previous_shares = previous_shares.reindex(
        all_tickers,
        fill_value=0,
    )
    changes = aligned_today_shares.sub(aligned_previous_shares)
    changes = changes.loc[changes.ne(0)].sort_index()
    changes.name = "share_change"
    changes.index.name = "ticker"
    return changes


def calculate_average_daily_volumes(
    close_prices: pd.DataFrame,
    volumes: pd.DataFrame,
    lookback_days: int = DEFAULT_ADV_LOOKBACK_DAYS,
) -> pd.DataFrame:
    """Return trailing share ADV and average daily dollar volume."""
    if lookback_days <= 0:
        raise ValueError("lookback_days must be positive.")

    aligned_close_prices = close_prices.reindex(
        index=volumes.index,
        columns=volumes.columns,
    )
    valid_volumes = volumes.where(volumes.gt(0))
    volume_window = valid_volumes.tail(lookback_days)
    dollar_volume_window = aligned_close_prices.mul(valid_volumes).tail(
        lookback_days
    )
    average_daily_volume = volume_window.mean().where(
        volume_window.count().eq(lookback_days)
    )
    average_daily_dollar_volume = dollar_volume_window.mean().where(
        dollar_volume_window.count().eq(lookback_days)
    )

    return pd.DataFrame(
        {
            "average_daily_volume": average_daily_volume,
            "average_daily_dollar_volume": average_daily_dollar_volume,
        }
    )


def calculate_adv_by_position(
    shares: pd.Series,
    prices: pd.Series,
    as_of_date: date,
    lookback_days: int = DEFAULT_ADV_LOOKBACK_DAYS,
    liquidation_participation_rate: float = (
        DEFAULT_LIQUIDATION_PARTICIPATION_RATE
    ),
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_retries: int = DEFAULT_MAX_RETRIES,
    database: str | Path = BLOOMBERG_DATABASE,
    allow_download: bool = True,
) -> pd.DataFrame:
    """Return position ADV, dollar ADV, and participation ratios."""
    if lookback_days <= 0:
        raise ValueError("lookback_days must be positive.")
    if not 0 < liquidation_participation_rate <= 1:
        raise ValueError(
            "liquidation_participation_rate must be greater than zero "
            "and no greater than one."
        )

    start_date = as_of_date - timedelta(
        days=math.ceil(
            lookback_days * HISTORY_CALENDAR_MULTIPLIER
        )
    )
    history = get_price_history(
        tickers=shares.index,
        start_date=start_date,
        end_date=as_of_date,
        batch_size=batch_size,
        max_retries=max_retries,
        database=database,
        allow_download=allow_download,
    )
    close_prices = history.pivot(
        index="date",
        columns="ticker",
        values="close",
    ).sort_index()
    volumes = history.pivot(
        index="date",
        columns="ticker",
        values="volume",
    ).sort_index()
    average_volumes = calculate_average_daily_volumes(
        close_prices=close_prices,
        volumes=volumes,
        lookback_days=lookback_days,
    )

    aligned_prices = prices.reindex(shares.index)
    position_market_value = shares.mul(aligned_prices)
    results = pd.DataFrame(
        {
            "shares": shares,
            "price": aligned_prices,
            "position_market_value": position_market_value,
            "average_daily_volume": average_volumes[
                "average_daily_volume"
            ].reindex(shares.index),
            "average_daily_dollar_volume": average_volumes[
                "average_daily_dollar_volume"
            ].reindex(shares.index),
        },
        index=shares.index,
    )
    results["position_shares_pct_adv"] = shares.abs().div(
        results["average_daily_volume"]
    )
    results["position_pct_of_daily_market_volume"] = (
        results["position_shares_pct_adv"] * 100.0
    )
    results["position_value_pct_dollar_adv"] = (
        position_market_value.abs().div(
            results["average_daily_dollar_volume"]
        )
    )
    results["liquidation_participation_rate"] = (
        liquidation_participation_rate
    )
    results["estimated_days_to_liquidate"] = (
        results["position_shares_pct_adv"]
        / liquidation_participation_rate
    )
    results["estimated_full_trading_days"] = results[
        "estimated_days_to_liquidate"
    ].map(
        lambda value: math.ceil(value) if pd.notna(value) else pd.NA
    ).astype("Int64")
    results.index.name = "ticker"
    return results.sort_index()


def write_position_changes(
    summary: PortfolioSummary,
    output_dir: str | Path = PORTFOLIO_DATA_DIR,
) -> Path:
    """Write ticker-level share changes to a dated CSV file."""
    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / get_portfolio_output_filename(
        prefix="position_diff",
        portfolio_type=summary.portfolio_type,
        as_of_date=summary.as_of_date,
    )
    changes = summary.position_changes.rename("shares_difference").to_frame()
    changes.to_csv(output_path, index=True, index_label="ticker")
    return output_path


def write_adv_by_position(
    summary: PortfolioSummary,
    output_dir: str | Path = PORTFOLIO_DATA_DIR,
) -> Path:
    """Write dated ticker-level ADV and participation calculations."""
    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / get_portfolio_output_filename(
        prefix="adv_by_position",
        portfolio_type=summary.portfolio_type,
        as_of_date=summary.as_of_date,
    )
    summary.adv_by_position.to_csv(
        output_path,
        index=True,
        index_label="ticker",
    )
    return output_path


def calculate_daily_turnover(
    today_shares: pd.Series,
    previous_shares: pd.Series,
    today_prices: pd.Series,
    previous_prices: pd.Series,
) -> float:
    """Return one-half the absolute change in daily portfolio weights."""
    today_market_values = today_shares.mul(today_prices).abs()
    previous_market_values = previous_shares.mul(previous_prices).abs()
    today_capital = float(today_market_values.sum(min_count=1))
    previous_capital = float(previous_market_values.sum(min_count=1))

    if not math.isfinite(today_capital) or today_capital <= 0:
        raise ValueError(
            "Cannot calculate turnover without positive current gross "
            "market value."
        )
    if not math.isfinite(previous_capital) or previous_capital <= 0:
        raise ValueError(
            "Cannot calculate turnover without positive previous gross "
            "market value."
        )

    today_weights = calculate_portfolio_weights(
        shares=today_shares,
        prices=today_prices,
        capital=today_capital,
    )
    previous_weights = calculate_portfolio_weights(
        shares=previous_shares,
        prices=previous_prices,
        capital=previous_capital,
    )

    all_tickers = today_weights.index.union(previous_weights.index)
    aligned_today_weights = today_weights.reindex(
        all_tickers,
        fill_value=0.0,
    ).fillna(0.0)
    aligned_previous_weights = previous_weights.reindex(
        all_tickers,
        fill_value=0.0,
    ).fillna(0.0)
    absolute_weight_changes = (
        aligned_today_weights.sub(aligned_previous_weights).abs()
    )
    return float(0.5 * absolute_weight_changes.sum())


def calculate_daily_return(
    previous_shares: pd.Series,
    today_shares: pd.Series,
    today_prices: pd.Series,
    previous_prices: pd.Series,
) -> float:
    """Return the daily change in gross portfolio capital."""
    today_capital = float(
        today_shares.mul(today_prices).abs().sum(min_count=1)
    )
    previous_capital = float(
        previous_shares.mul(previous_prices).abs().sum(min_count=1)
    )

    if not math.isfinite(today_capital) or today_capital <= 0:
        raise ValueError(
            "Cannot calculate daily return without positive current capital."
        )
    if not math.isfinite(previous_capital) or previous_capital <= 0:
        raise ValueError(
            "Cannot calculate daily return without positive previous capital."
        )

    return float(today_capital / previous_capital - 1.0)


def get_market_caps(
    tickers: pd.Index,
    prices: pd.Series,
    as_of_date: date,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_retries: int = DEFAULT_MAX_RETRIES,
    database: str | Path = BLOOMBERG_DATABASE,
    allow_download: bool = True,
) -> pd.Series:
    """Return Bloomberg market caps for a historical portfolio date."""
    del prices
    return get_historical_market_caps(
        tickers=tickers,
        as_of_date=as_of_date,
        batch_size=batch_size,
        max_retries=max_retries,
        database=database,
        allow_download=allow_download,
    )


def calculate_weighted_average_market_cap(
    shares: pd.Series,
    prices: pd.Series,
    market_caps: pd.Series,
    capital: float,
) -> float:
    """Weight company market caps by absolute position value over capital."""
    weights = calculate_portfolio_weights(
        shares=shares,
        prices=prices,
        capital=capital,
    )
    aligned_market_caps = market_caps.reindex(weights.index)
    valid = aligned_market_caps.notna() & weights.notna()

    if not valid.any():
        raise ValueError(
            "Cannot calculate weighted average market cap without "
            "valid position weights and market caps."
        )

    return float(aligned_market_caps.loc[valid].mul(weights.loc[valid]).sum())


def calculate_portfolio_weights(
    shares: pd.Series,
    prices: pd.Series,
    capital: float,
) -> pd.Series:
    """Return weights as absolute position market value over capital."""
    if not math.isfinite(capital) or capital <= 0:
        raise ValueError("capital must be a positive finite number.")

    weights = shares.mul(prices).abs().div(capital)
    weights.name = "weight"
    weights.index.name = "ticker"
    return weights


def create_portfolio_summary(
    today_csv: str | Path,
    market_cap_provider: MarketCapProvider = get_market_caps,
    capital: float | None = None,
    price_batch_size: int = DEFAULT_BATCH_SIZE,
    price_max_retries: int = DEFAULT_MAX_RETRIES,
    allow_price_download: bool = True,
    adv_lookback_days: int = DEFAULT_ADV_LOOKBACK_DAYS,
    liquidation_participation_rate: float = (
        DEFAULT_LIQUIDATION_PARTICIPATION_RATE
    ),
) -> PortfolioSummary:
    """Calculate current and day-over-day portfolio summary metrics."""
    today_path = resolve_portfolio_path(today_csv)
    previous_path = find_previous_portfolio(today_path)

    today_shares = load_shares(today_path)
    today_prices = load_prices(
        today_path,
        batch_size=price_batch_size,
        max_retries=price_max_retries,
        allow_download=allow_price_download,
    )
    previous_shares = load_shares(previous_path)
    previous_prices = load_prices(
        previous_path,
        batch_size=price_batch_size,
        max_retries=price_max_retries,
        allow_download=allow_price_download,
    )
    as_of_date = get_portfolio_date(today_path)
    portfolio_type = get_portfolio_type(today_path)
    if today_shares.isna().any():
        raise ValueError("Today's shares cannot contain missing values.")
    if previous_shares.isna().any():
        raise ValueError(
            "Previous-day shares cannot contain missing values."
        )

    adv_by_position = calculate_adv_by_position(
        shares=today_shares,
        prices=today_prices,
        as_of_date=as_of_date,
        lookback_days=adv_lookback_days,
        liquidation_participation_rate=(
            liquidation_participation_rate
        ),
        batch_size=price_batch_size,
        max_retries=price_max_retries,
        allow_download=allow_price_download,
    )

    missing_today_prices = int(today_prices.isna().sum())
    missing_previous_prices = int(previous_prices.isna().sum())
    if missing_today_prices or missing_previous_prices:
        warnings.warn(
            "Market values exclude positions with missing prices: "
            f"{missing_today_prices} current and "
            f"{missing_previous_prices} previous.",
            RuntimeWarning,
            stacklevel=2,
        )

    today_position_values = today_shares.mul(today_prices)
    capital_used = (
        float(today_position_values.abs().sum(min_count=1))
        if capital is None
        else float(capital)
    )
    if not math.isfinite(capital_used) or capital_used <= 0:
        raise ValueError(
            "Capital must be positive. Supply capital explicitly if the "
            "portfolio has no priced exposure."
        )

    market_caps = market_cap_provider(
        today_shares.index,
        today_prices,
        as_of_date,
    )
    weighted_average_market_cap = calculate_weighted_average_market_cap(
        shares=today_shares,
        prices=today_prices,
        market_caps=market_caps,
        capital=capital_used,
    )
    today_market_value = float(
        today_shares.mul(today_prices).sum(min_count=1)
    )
    previous_market_value = float(
        previous_shares.mul(previous_prices).sum(min_count=1)
    )
    valid_adv = (
        adv_by_position["average_daily_volume"].notna()
        & adv_by_position["average_daily_dollar_volume"].notna()
    )

    return PortfolioSummary(
        as_of_date=as_of_date,
        previous_date=get_portfolio_date(previous_path),
        portfolio_type=portfolio_type,
        number_of_stocks=len(today_shares),
        weighted_average_market_cap=weighted_average_market_cap,
        ticker_coverage=int(
            (
                today_prices.notna()
                & market_caps.reindex(today_shares.index).notna()
                & valid_adv.reindex(today_shares.index, fill_value=False)
            ).sum()
        ),
        capital=capital_used,
        daily_return=calculate_daily_return(
            previous_shares=previous_shares,
            today_shares=today_shares,
            today_prices=today_prices,
            previous_prices=previous_prices,
        ),
        daily_turnover=calculate_daily_turnover(
            today_shares=today_shares,
            previous_shares=previous_shares,
            today_prices=today_prices,
            previous_prices=previous_prices,
        ),
        market_value=today_market_value,
        market_value_change=today_market_value - previous_market_value,
        position_changes=calculate_position_changes(
            today_shares=today_shares,
            previous_shares=previous_shares,
        ),
        market_caps=market_caps,
        adv_lookback_days=adv_lookback_days,
        median_adv=float(
            adv_by_position.loc[
                valid_adv,
                "average_daily_volume",
            ].median()
        ),
        median_dollar_adv=float(
            adv_by_position.loc[
                valid_adv,
                "average_daily_dollar_volume",
            ].median()
        ),
        median_position_pct_adv=float(
            adv_by_position.loc[
                valid_adv,
                "position_shares_pct_adv",
            ].median()
        ),
        maximum_position_pct_adv=float(
            adv_by_position.loc[
                valid_adv,
                "position_shares_pct_adv",
            ].max()
        ),
        liquidation_participation_rate=(
            liquidation_participation_rate
        ),
        median_days_to_liquidate=float(
            adv_by_position.loc[
                valid_adv,
                "estimated_days_to_liquidate",
            ].median()
        ),
        maximum_days_to_liquidate=float(
            adv_by_position.loc[
                valid_adv,
                "estimated_days_to_liquidate",
            ].max()
        ),
        adv_by_position=adv_by_position,
    )


def _format_dollars(value: float) -> str:
    """Format a dollar value with a sign and thousands separators."""
    return f"${value:,.2f}"


def _format_market_cap(value: float) -> str:
    """Format company market capitalization in readable units."""
    if abs(value) >= 1_000_000_000_000:
        return f"${value / 1_000_000_000_000:,.2f}T"
    if abs(value) >= 1_000_000_000:
        return f"${value / 1_000_000_000:,.2f}B"
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:,.2f}M"
    return _format_dollars(value)


def format_portfolio_summary(summary: PortfolioSummary) -> str:
    """Return a printable portfolio summary."""
    summary_rows = [
        ("As of", summary.as_of_date.isoformat()),
        (
            "Portfolio type",
            {
                "US_live_port": "Full portfolio",
                "long_positions": "Long only",
                "short_positions": "Short only",
            }[summary.portfolio_type],
        ),
        ("Number of stocks", f"{summary.number_of_stocks:,}"),
        (
            "Weighted average market cap",
            _format_market_cap(summary.weighted_average_market_cap),
        ),
        (
            "Ticker coverage",
            f"{summary.ticker_coverage:,}/{summary.number_of_stocks:,}",
        ),
        ("ADV lookback", f"{summary.adv_lookback_days} trading days"),
        ("Median share ADV", f"{summary.median_adv:,.0f}"),
        (
            "Median dollar ADV",
            _format_dollars(summary.median_dollar_adv),
        ),
        (
            "Median % daily volume",
            f"{summary.median_position_pct_adv:.2%}",
        ),
        (
            "Maximum % daily volume",
            f"{summary.maximum_position_pct_adv:.2%}",
        ),
        (
            "Liquidation participation cap",
            f"{summary.liquidation_participation_rate:.2%}",
        ),
        (
            "Median days to liquidate",
            f"{summary.median_days_to_liquidate:,.2f}",
        ),
        (
            "Maximum days to liquidate",
            f"{summary.maximum_days_to_liquidate:,.2f}",
        ),
        ("Capital", _format_dollars(summary.capital)),
        ("Daily return", f"{summary.daily_return:.2%}"),
        ("Daily turnover", f"{summary.daily_turnover:.2%}"),
        ("Market value", _format_dollars(summary.market_value)),
        (
            "Market value change",
            _format_dollars(summary.market_value_change),
        ),
    ]
    label_width = max(len(label) for label, _ in summary_rows)
    lines = ["PORTFOLIO SUMMARY", "=" * 72]
    lines.extend(
        f"{label:<{label_width}} : {value}"
        for label, value in summary_rows
    )
    return "\n".join(lines)


def main() -> None:
    """Parse a current portfolio CSV and print its daily summary."""
    parser = argparse.ArgumentParser(
        description="Print a current and day-over-day portfolio summary."
    )
    parser.add_argument(
        "today_csv",
        type=Path,
        help=(
            "Today's portfolio filename in port_data, or an explicit path "
            "to a full, long-only, or short-only dated portfolio CSV."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Sequential Bloomberg request batch size (default: 50).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help="Retries for failed Bloomberg requests (default: 3).",
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help=(
            "Skip every Bloomberg data request and use only data already "
            "stored in port_data/bloomberg_data.db."
        ),
    )
    parser.add_argument(
        "--capital",
        type=float,
        help=(
            "Portfolio capital used as the weight denominator. Defaults "
            "to current gross market value."
        ),
    )
    parser.add_argument(
        "--benchmark",
        default="VTHR",
        help=(
            "Benchmark ticker used for beta and relative strength "
            "(default: VTHR)."
        ),
    )
    parser.add_argument(
        "--risk-lookback-days",
        type=int,
        default=252,
        help="Trading-day lookback for risk metrics (default: 252).",
    )
    parser.add_argument(
        "--adv-lookback-days",
        type=int,
        default=DEFAULT_ADV_LOOKBACK_DAYS,
        help="Trading-day lookback for ADV calculations (default: 20).",
    )
    parser.add_argument(
        "--liquidation-participation-rate",
        type=float,
        default=DEFAULT_LIQUIDATION_PARTICIPATION_RATE,
        help=(
            "Maximum fraction of ADV traded per day when estimating "
            "liquidation time (default: 0.10)."
        ),
    )
    parser.add_argument(
        "--factor-universe",
        default=DEFAULT_FACTOR_UNIVERSE,
        help=(
            "Bloomberg index used to normalize custom factor scores "
            f"(default: {DEFAULT_FACTOR_UNIVERSE})."
        ),
    )
    parser.add_argument(
        "--factor-universe-workbook",
        type=Path,
        help=(
            "Bloomberg PORT XLSX export for today's factor universe. "
            "Imports positive Bmrk-weight tickers before data collection."
        ),
    )
    parser.add_argument(
        "--minimum-observations",
        type=int,
        default=60,
        help="Minimum returns required per stock (default: 60).",
    )
    parser.add_argument(
        "--downside-threshold",
        type=float,
        default=0.0,
        help=(
            "Daily minimum acceptable return used as downside-risk tau "
            "in decimal form (default: 0)."
        ),
    )
    parser.add_argument(
        "--var-confidence",
        type=float,
        default=0.95,
        help="One-day parametric VaR confidence (default: 0.95).",
    )
    parser.add_argument(
        "--mac3-model",
        help=(
            "Bloomberg MAC3 model identifier to report from the local "
            "Risk Model Files cache. Omit to skip the MAC3 report."
        ),
    )
    parser.add_argument(
        "--mac3-horizon",
        default="quarterly",
        help="Cached MAC3 forecast horizon (default: quarterly).",
    )
    args = parser.parse_args()

    from risk_metrics import (
        create_risk_report,
        format_risk_report,
        write_risk_contributions,
    )

    today_path = resolve_portfolio_path(args.today_csv)
    previous_path = find_previous_portfolio(today_path)
    as_of_date = get_portfolio_date(today_path)
    portfolio_type = get_portfolio_type(today_path)
    today_shares = load_shares(today_path)
    previous_shares = load_shares(previous_path)
    if args.factor_universe_workbook is not None:
        try:
            imported_snapshot = import_factor_universe_workbook(
                workbook_path=args.factor_universe_workbook,
                index_ticker=args.factor_universe,
                database=BLOOMBERG_DATABASE,
                expected_as_of_date=as_of_date,
            )
        except (FileNotFoundError, ValueError) as error:
            parser.error(str(error))
        print(
            "Factor universe imported: "
            f"{len(imported_snapshot.members):,} tickers for "
            f"{imported_snapshot.as_of_date}"
        )
    try:
        factor_universe_members = get_index_members(
            index_ticker=args.factor_universe,
            as_of_date=as_of_date,
            max_retries=args.max_retries,
            allow_download=not args.cache_only,
        )
    except LookupError as error:
        if not args.cache_only:
            raise
        warnings.warn(
            f"{error}. Falling back to portfolio tickers for factor "
            "normalization.",
            RuntimeWarning,
            stacklevel=2,
        )
        factor_universe_members = pd.Series(
            index=pd.Index([], dtype="object", name="ticker"),
            dtype="float64",
            name="index_weight",
        )
    factor_tickers = factor_universe_members.index.union(
        today_shares.index
    )
    price_tickers = (
        factor_tickers.union(previous_shares.index)
        .union(pd.Index([args.benchmark]))
    )
    price_start_date = as_of_date - timedelta(
        days=math.ceil(
            max(args.risk_lookback_days, args.adv_lookback_days)
            * HISTORY_CALENDAR_MULTIPLIER
        )
    )
    if args.cache_only:
        print(
            "Bloomberg data collection skipped; using the local cache only."
        )
    else:
        ensure_price_history(
            tickers=price_tickers,
            start_date=price_start_date,
            end_date=as_of_date,
            batch_size=args.batch_size,
            max_retries=args.max_retries,
        )
        get_security_metadata(
            tickers=factor_tickers,
            batch_size=args.batch_size,
            max_retries=args.max_retries,
        )
        get_historical_market_caps(
            tickers=factor_tickers,
            as_of_date=as_of_date,
            batch_size=args.batch_size,
            max_retries=args.max_retries,
        )

    market_cap_provider = lambda tickers, prices, as_of_date: get_market_caps(
        tickers=tickers,
        prices=prices,
        as_of_date=as_of_date,
        batch_size=args.batch_size,
        max_retries=args.max_retries,
        allow_download=not args.cache_only,
    )
    summary = create_portfolio_summary(
        today_csv=args.today_csv,
        market_cap_provider=market_cap_provider,
        capital=args.capital,
        price_batch_size=args.batch_size,
        price_max_retries=args.max_retries,
        allow_price_download=False,
        adv_lookback_days=args.adv_lookback_days,
        liquidation_participation_rate=(
            args.liquidation_participation_rate
        ),
    )
    print(format_portfolio_summary(summary))
    output_path = write_position_changes(summary)
    print(f"\nPosition differences written to: {output_path}")
    adv_output_path = write_adv_by_position(summary)
    print(f"\nADV by position written to: {adv_output_path}")

    today_prices = load_prices(
        today_path,
        batch_size=args.batch_size,
        max_retries=args.max_retries,
        allow_download=False,
    )
    risk_report = create_risk_report(
        shares=today_shares,
        prices=today_prices,
        market_caps=summary.market_caps,
        as_of_date=summary.as_of_date,
        capital=summary.capital,
        benchmark=args.benchmark,
        lookback_days=args.risk_lookback_days,
        minimum_observations=args.minimum_observations,
        downside_threshold=args.downside_threshold,
        var_confidence=args.var_confidence,
        batch_size=args.batch_size,
        max_retries=args.max_retries,
        allow_price_download=False,
        portfolio_type=portfolio_type,
        factor_universe=args.factor_universe,
    )
    print(f"\n{format_risk_report(risk_report)}")
    risk_output_path = write_risk_contributions(risk_report)
    print(f"\nRisk contributions written to: {risk_output_path}")

    if args.mac3_model:
        from mac3_cache import (
            calculate_mac3_factor_risk,
            format_mac3_risk_report,
            write_mac3_factor_risk,
        )

        mac3_weights = today_shares.mul(today_prices).div(summary.capital)
        mac3_report = calculate_mac3_factor_risk(
            weights=mac3_weights,
            capital=summary.capital,
            as_of_date=summary.as_of_date,
            model=args.mac3_model,
            horizon=args.mac3_horizon,
            database=BLOOMBERG_DATABASE,
        )
        print(f"\n{format_mac3_risk_report(mac3_report)}")
        mac3_output_path = write_mac3_factor_risk(
            mac3_report,
            PORTFOLIO_DATA_DIR,
            portfolio_type=portfolio_type,
        )
        print(f"\nMAC3 factor risks written to: {mac3_output_path}")


if __name__ == "__main__":
    main()
