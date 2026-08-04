"""Create a daily summary for a US live equity portfolio."""

from __future__ import annotations

import argparse
import math
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf
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
    get_portfolio_type,
    load_prices,
    load_shares,
    resolve_portfolio_path,
)
from price_cache import (
    YFINANCE_DATABASE,
    ensure_price_history,
    get_price_history,
    get_prices_for_date,
    load_historical_market_caps,
    upsert_historical_market_caps,
)


DEFAULT_BATCH_SIZE = 50
DEFAULT_MAX_RETRIES = 3
DEFAULT_ADV_LOOKBACK_DAYS = 20
HISTORY_CALENDAR_MULTIPLIER = 2.25
RETRY_BACKOFF_SECONDS = 1.0
SHARES_LOOKBACK_DAYS = 730
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

    portfolio_type: str
    as_of_date: date
    previous_date: date
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
    adv_by_position: pd.DataFrame


def _portfolio_date(csv_file: str | Path) -> date:
    """Return the date encoded in a validated portfolio filename."""
    return get_portfolio_date(csv_file)


def find_previous_portfolio(today_csv: str | Path) -> Path:
    """Return the portfolio from the immediately preceding NYSE session."""
    today_path = resolve_portfolio_path(today_csv)
    today_date = _portfolio_date(today_path)
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
    raw_close: pd.DataFrame,
    volumes: pd.DataFrame,
    lookback_days: int = DEFAULT_ADV_LOOKBACK_DAYS,
) -> pd.DataFrame:
    """Return trailing share ADV and average daily dollar volume."""
    if lookback_days <= 0:
        raise ValueError("lookback_days must be positive.")

    aligned_raw_close = raw_close.reindex(
        index=volumes.index,
        columns=volumes.columns,
    )
    volume_window = volumes.tail(lookback_days)
    dollar_volume_window = aligned_raw_close.mul(volumes).tail(
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
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_retries: int = DEFAULT_MAX_RETRIES,
    allow_download: bool = True,
) -> pd.DataFrame:
    """Return share and dollar ADV with position participation ratios."""
    if lookback_days <= 0:
        raise ValueError("lookback_days must be positive.")

    start_date = as_of_date - timedelta(
        days=math.ceil(lookback_days * HISTORY_CALENDAR_MULTIPLIER)
    )
    history = get_price_history(
        tickers=shares.index,
        start_date=start_date,
        end_date=as_of_date,
        batch_size=batch_size,
        max_retries=max_retries,
        allow_download=allow_download,
    )
    raw_close = history.pivot(
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
        raw_close=raw_close,
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
    results["position_value_pct_dollar_adv"] = (
        position_market_value.abs().div(
            results["average_daily_dollar_volume"]
        )
    )
    results.index.name = "ticker"
    return results.sort_index()


def write_position_changes(
    summary: PortfolioSummary,
    output_dir: str | Path = PORTFOLIO_DATA_DIR,
) -> Path:
    """Write ticker-level share changes to a dated CSV file."""
    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    portfolio_label = (
        ""
        if summary.portfolio_type == "US_live_port"
        else f"_{summary.portfolio_type}"
    )
    output_path = output_directory / (
        f"position_diff{portfolio_label}_{summary.as_of_date:%Y%m%d}.csv"
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
    portfolio_label = (
        ""
        if summary.portfolio_type == "US_live_port"
        else f"_{summary.portfolio_type}"
    )
    output_path = output_directory / (
        f"adv_by_position{portfolio_label}_"
        f"{summary.as_of_date:%Y%m%d}.csv"
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
    today_prices: pd.Series,
    previous_prices: pd.Series,
) -> float:
    """Return prior holdings' daily P&L over prior gross capital."""
    aligned_today_prices = today_prices.reindex(previous_shares.index)
    aligned_previous_prices = previous_prices.reindex(
        previous_shares.index
    )
    valid = aligned_today_prices.notna() & aligned_previous_prices.notna()
    if not valid.any():
        raise ValueError(
            "Cannot calculate daily return without overlapping prices."
        )

    priced_shares = previous_shares.loc[valid]
    prior_market_values = priced_shares.mul(
        aligned_previous_prices.loc[valid]
    )
    prior_capital = float(prior_market_values.abs().sum())
    if not math.isfinite(prior_capital) or prior_capital <= 0:
        raise ValueError(
            "Cannot calculate daily return without positive prior capital."
        )

    daily_profit_and_loss = priced_shares.mul(
        aligned_today_prices.loc[valid].sub(
            aligned_previous_prices.loc[valid]
        )
    ).sum()
    return float(daily_profit_and_loss / prior_capital)


def _extract_historical_shares(
    ticker: yf.Ticker,
    as_of_date: date,
) -> float:
    """Return the latest reported shares outstanding on or before a date."""
    start_date = as_of_date - timedelta(days=SHARES_LOOKBACK_DAYS)
    end_date = as_of_date + timedelta(days=1)
    shares_history = ticker.get_shares_full(
        start=start_date.isoformat(),
        end=end_date.isoformat(),
    )
    if shares_history is None or shares_history.empty:
        return math.nan

    shares_history = pd.to_numeric(shares_history, errors="coerce").dropna()
    observation_dates = pd.Index(
        [timestamp.date() for timestamp in shares_history.index]
    )
    shares_history = shares_history.loc[observation_dates <= as_of_date]
    if shares_history.empty:
        return math.nan

    shares_outstanding = float(shares_history.iloc[-1])
    if not math.isfinite(shares_outstanding) or shares_outstanding <= 0:
        return math.nan

    return shares_outstanding


def _fetch_market_cap_batch(
    tickers: list[str],
    prices: pd.Series,
    as_of_date: date,
    max_retries: int,
    backoff_seconds: float,
) -> pd.Series:
    """Fetch historical shares and calculate one dated market-cap batch."""
    results = pd.Series(
        index=pd.Index(tickers, name="ticker"),
        dtype="float64",
        name="market_cap",
    )
    pending = list(tickers)

    for attempt in range(max_retries + 1):
        yahoo_symbols = {
            ticker: ticker.replace(".", "-").replace("/", "-").upper()
            for ticker in pending
        }
        ticker_group = yf.Tickers(" ".join(yahoo_symbols.values()))
        failed: list[str] = []

        for ticker in pending:
            yahoo_symbol = yahoo_symbols[ticker]
            price = prices.get(ticker, math.nan)
            try:
                shares_outstanding = _extract_historical_shares(
                    ticker=ticker_group.tickers[yahoo_symbol],
                    as_of_date=as_of_date,
                )
            except Exception:
                shares_outstanding = math.nan

            if pd.isna(price) or math.isnan(shares_outstanding):
                failed.append(ticker)
            else:
                market_cap = float(price) * shares_outstanding
                if math.isfinite(market_cap) and market_cap > 0:
                    results.loc[ticker] = market_cap
                else:
                    failed.append(ticker)

        pending = failed
        if not pending or attempt == max_retries:
            break

        time.sleep(backoff_seconds * (2**attempt))

    return results


def get_market_caps(
    tickers: pd.Index,
    prices: pd.Series,
    as_of_date: date,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_retries: int = DEFAULT_MAX_RETRIES,
    database: str | Path = YFINANCE_DATABASE,
) -> pd.Series:
    """Return Yahoo market caps calculated for a historical portfolio date."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if max_retries < 0:
        raise ValueError("max_retries cannot be negative.")

    unique_tickers = pd.Index(tickers.astype(str).unique(), name="ticker")
    cached_market_caps = load_historical_market_caps(
        tickers=unique_tickers,
        as_of_date=as_of_date,
        database=database,
    )
    market_caps = cached_market_caps.reindex(unique_tickers)
    missing_tickers = market_caps.index[market_caps.isna()]
    aligned_prices = prices.reindex(unique_tickers)
    downloadable_tickers = missing_tickers[
        aligned_prices.reindex(missing_tickers).notna()
    ]

    for start in range(0, len(downloadable_tickers), batch_size):
        batch = downloadable_tickers[start : start + batch_size].tolist()
        downloaded = _fetch_market_cap_batch(
            tickers=batch,
            prices=aligned_prices,
            as_of_date=as_of_date,
            max_retries=max_retries,
            backoff_seconds=RETRY_BACKOFF_SECONDS,
        )
        market_caps.update(downloaded)
        upsert_historical_market_caps(
            market_caps=downloaded,
            as_of_date=as_of_date,
            prices=aligned_prices,
            database=database,
        )

    missing_tickers = market_caps.index[market_caps.isna()]
    if len(missing_tickers) == len(market_caps):
        raise RuntimeError(
            "Yahoo Finance did not return historical shares data for any "
            "ticker."
        )

    if len(missing_tickers) > 0:
        preview = ", ".join(missing_tickers[:10])
        if len(missing_tickers) > 10:
            preview = f"{preview}, ..."
        warnings.warn(
            "Yahoo Finance did not return historical shares data for "
            f"{len(missing_tickers)} ticker(s): {preview}",
            RuntimeWarning,
            stacklevel=2,
        )

    return market_caps


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
    as_of_date = _portfolio_date(today_path)
    today_prices_for_prior_holdings = get_prices_for_date(
        tickers=previous_shares.index,
        as_of_date=as_of_date,
        batch_size=price_batch_size,
        max_retries=price_max_retries,
        allow_download=allow_price_download,
    )
    adv_by_position = calculate_adv_by_position(
        shares=today_shares,
        prices=today_prices,
        as_of_date=as_of_date,
        lookback_days=adv_lookback_days,
        batch_size=price_batch_size,
        max_retries=price_max_retries,
        allow_download=allow_price_download,
    )

    if today_shares.isna().any():
        raise ValueError("Today's shares cannot contain missing values.")
    if previous_shares.isna().any():
        raise ValueError(
            "Previous-day shares cannot contain missing values."
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
        portfolio_type=get_portfolio_type(today_path),
        as_of_date=as_of_date,
        previous_date=_portfolio_date(previous_path),
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
            today_prices=today_prices_for_prior_holdings,
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
            adv_by_position.loc[valid_adv, "average_daily_volume"].median()
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
            "Median position / ADV",
            f"{summary.median_position_pct_adv:.2%}",
        ),
        (
            "Maximum position / ADV",
            f"{summary.maximum_position_pct_adv:.2%}",
        ),
        ("Capital", _format_dollars(summary.capital)),
        ("Daily return", f"{summary.daily_return:.8%}"),
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
            "Today's dated full, long, or short portfolio filename in "
            "port_data, or an explicit path."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Sequential Yahoo Finance batch size (default: 50).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help="Retries for failed Yahoo tickers (default: 3).",
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
        help="Benchmark ticker used for beta (default: VTHR).",
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
        "--minimum-observations",
        type=int,
        default=60,
        help="Minimum returns required per stock (default: 60).",
    )
    parser.add_argument(
        "--var-confidence",
        type=float,
        default=0.95,
        help="One-day parametric VaR confidence (default: 0.95).",
    )
    args = parser.parse_args()

    from risk_metrics import (
        DEFAULT_RELATIVE_STRENGTH_LOOKBACK_DAYS,
        create_risk_report,
        format_risk_report,
        write_risk_contributions,
    )

    today_path = resolve_portfolio_path(args.today_csv)
    previous_path = find_previous_portfolio(today_path)
    as_of_date = _portfolio_date(today_path)
    today_shares = load_shares(today_path)
    previous_shares = load_shares(previous_path)
    price_tickers = (
        today_shares.index.union(previous_shares.index)
        .union(pd.Index([args.benchmark]))
    )
    price_start_date = as_of_date - timedelta(
        days=math.ceil(
            max(
                args.risk_lookback_days,
                args.adv_lookback_days,
                DEFAULT_RELATIVE_STRENGTH_LOOKBACK_DAYS,
            )
            * HISTORY_CALENDAR_MULTIPLIER
        )
    )
    ensure_price_history(
        tickers=price_tickers,
        start_date=price_start_date,
        end_date=as_of_date,
        batch_size=args.batch_size,
        max_retries=args.max_retries,
    )

    market_cap_provider = lambda tickers, prices, as_of_date: get_market_caps(
        tickers=tickers,
        prices=prices,
        as_of_date=as_of_date,
        batch_size=args.batch_size,
        max_retries=args.max_retries,
    )
    summary = create_portfolio_summary(
        today_csv=args.today_csv,
        market_cap_provider=market_cap_provider,
        capital=args.capital,
        price_batch_size=args.batch_size,
        price_max_retries=args.max_retries,
        allow_price_download=False,
        adv_lookback_days=args.adv_lookback_days,
    )
    print(format_portfolio_summary(summary))
    output_path = write_position_changes(summary)
    print(f"\nPosition differences written to: {output_path}")
    adv_output_path = write_adv_by_position(summary)
    print(f"\nADV by position written to: {adv_output_path}")

    risk_report = create_risk_report(
        shares=today_shares,
        prices=load_prices(
            today_path,
            batch_size=args.batch_size,
            max_retries=args.max_retries,
            allow_download=False,
        ),
        market_caps=summary.market_caps,
        as_of_date=summary.as_of_date,
        portfolio_type=summary.portfolio_type,
        capital=summary.capital,
        benchmark=args.benchmark,
        lookback_days=args.risk_lookback_days,
        minimum_observations=args.minimum_observations,
        var_confidence=args.var_confidence,
        batch_size=args.batch_size,
        max_retries=args.max_retries,
        allow_price_download=False,
    )
    print(f"\n{format_risk_report(risk_report)}")
    risk_output_path = write_risk_contributions(risk_report)
    print(f"\nRisk contributions written to: {risk_output_path}")


if __name__ == "__main__":
    main()
