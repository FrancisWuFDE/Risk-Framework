"""Calculate portfolio exposures and market-risk metrics."""

from __future__ import annotations

import argparse
import math
import time
import warnings
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
import yfinance as yf

from construct_port import (
    PORTFOLIO_DATA_DIR,
    get_portfolio_date,
    get_portfolio_type,
    load_prices,
    load_shares,
    resolve_portfolio_path,
)
from price_cache import (
    PRICE_DATABASE,
    get_price_history,
    load_security_metadata,
    load_short_interest,
    upsert_security_metadata,
    upsert_short_interest,
)


TRADING_DAYS_PER_YEAR = 252
DEFAULT_LOOKBACK_DAYS = 252
DEFAULT_MINIMUM_OBSERVATIONS = 60
DEFAULT_BATCH_SIZE = 50
DEFAULT_MAX_RETRIES = 3
DEFAULT_VAR_CONFIDENCE = 0.95
DEFAULT_BENCHMARK = "VTHR"
DEFAULT_LIQUIDITY_LOOKBACK_DAYS = 63
DEFAULT_RELATIVE_STRENGTH_LOOKBACK_DAYS = 63
DEFAULT_RELATIVE_STRENGTH_SKIP_DAYS = 5
DEFAULT_SHORT_TERM_REVERSAL_MAX_LAG = 6
DEFAULT_SHORT_TERM_REVERSAL_HALF_LIFE_DAYS = 3.0
SHORT_INTEREST_MAX_AGE_DAYS = 45
RETRY_BACKOFF_SECONDS = 1.0
HISTORY_CALENDAR_MULTIPLIER = 2.25


@dataclass(frozen=True)
class RiskReport:
    """Store portfolio exposure, factor, and covariance-risk results."""

    portfolio_type: str
    as_of_date: date
    capital: float
    long_exposure: float
    short_exposure: float
    gross_exposure: float
    net_exposure: float
    beta_exposure: float
    sector_exposures: pd.DataFrame
    factor_exposures: pd.Series
    factor_coverage: pd.Series
    annualized_volatility: float
    daily_volatility: float
    systematic_volatility: float
    idiosyncratic_volatility: float
    factor_model_volatility: float
    systematic_variance_share: float
    idiosyncratic_variance_share: float
    parametric_var: float
    var_confidence: float
    covariance_coverage: int
    total_positions: int
    risk_contributions: pd.DataFrame


def _portfolio_date(csv_file: str | Path) -> date:
    """Extract the YYYYMMDD date from a resolved portfolio filename."""
    return get_portfolio_date(csv_file)


def _yahoo_symbol(ticker: str) -> str:
    """Translate common share-class notation to Yahoo's symbol format."""
    return ticker.replace(".", "-").replace("/", "-").upper()


def calculate_position_exposures(
    shares: pd.Series,
    prices: pd.Series,
) -> pd.Series:
    """Return long, short, gross, and net dollar exposures."""
    market_values = shares.mul(prices)
    long_exposure = float(market_values.clip(lower=0).sum(min_count=1))
    short_exposure = float(market_values.clip(upper=0).sum(min_count=1))

    return pd.Series(
        {
            "long": long_exposure,
            "short": short_exposure,
            "gross": long_exposure + abs(short_exposure),
            "net": long_exposure + short_exposure,
        },
        dtype="float64",
        name="exposure",
    )


def calculate_signed_weights(
    shares: pd.Series,
    prices: pd.Series,
    capital: float,
) -> pd.Series:
    """Return signed position-market-value weights."""
    if not math.isfinite(capital) or capital <= 0:
        raise ValueError("capital must be a positive finite number.")

    weights = shares.mul(prices).div(capital)
    weights.name = "weight"
    weights.index.name = "ticker"
    return weights


def download_market_history(
    tickers: pd.Index,
    as_of_date: date,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_retries: int = DEFAULT_MAX_RETRIES,
    database: str | Path = PRICE_DATABASE,
    allow_download: bool = True,
) -> pd.DataFrame:
    """Return raw closes, adjusted closes, and volumes from SQLite."""
    if lookback_days <= 1:
        raise ValueError("lookback_days must be greater than one.")

    unique_tickers = pd.Index(
        tickers.astype(str).unique(),
        name="ticker",
    )
    calendar_days = math.ceil(
        lookback_days * HISTORY_CALENDAR_MULTIPLIER
    )
    start_date = as_of_date - timedelta(days=calendar_days)
    history = get_price_history(
        tickers=unique_tickers,
        start_date=start_date,
        end_date=as_of_date,
        batch_size=batch_size,
        max_retries=max_retries,
        database=database,
        allow_download=allow_download,
    )
    history = history.rename(
        columns={
            "close": "raw_close",
            "adjusted_close": "close",
        }
    )
    return history[
        ["date", "ticker", "raw_close", "close", "volume"]
    ]


def _download_metadata_batch(
    tickers: list[str],
    max_retries: int,
) -> pd.DataFrame:
    """Download sectors and valuation metadata sequentially."""
    pending = list(tickers)
    records: list[dict[str, object]] = []

    for attempt in range(max_retries + 1):
        yahoo_symbols = {
            ticker: _yahoo_symbol(ticker)
            for ticker in pending
        }
        ticker_group = yf.Tickers(" ".join(yahoo_symbols.values()))
        failed: list[str] = []

        for ticker in pending:
            yahoo_symbol = yahoo_symbols[ticker]
            try:
                info = ticker_group.tickers[yahoo_symbol].get_info()
            except Exception:
                info = {}

            sector = info.get("sector")
            price_to_book = info.get("priceToBook")
            if sector is None and price_to_book is None:
                failed.append(ticker)
                continue

            records.append(
                {
                    "ticker": ticker,
                    "sector": sector or "Unclassified",
                    "price_to_book": price_to_book,
                    "retrieved_at": datetime.now()
                    .astimezone()
                    .isoformat(timespec="seconds"),
                }
            )

        pending = failed
        if not pending or attempt == max_retries:
            break

        time.sleep(RETRY_BACKOFF_SECONDS * (2**attempt))

    return pd.DataFrame.from_records(
        records,
        columns=["ticker", "sector", "price_to_book", "retrieved_at"],
    )


def get_risk_metadata(
    tickers: pd.Index,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_retries: int = DEFAULT_MAX_RETRIES,
    database: str | Path = PRICE_DATABASE,
) -> pd.DataFrame:
    """Return cached or newly downloaded sector and valuation metadata."""
    unique_tickers = pd.Index(tickers.astype(str).unique(), name="ticker")
    metadata = load_security_metadata(
        tickers=unique_tickers,
        database=database,
    )
    cached = metadata[["sector", "price_to_book"]].notna().any(axis=1)
    missing_tickers = metadata.index[~cached].tolist()

    for start in range(0, len(missing_tickers), batch_size):
        batch = missing_tickers[start : start + batch_size]
        downloaded = _download_metadata_batch(
            tickers=batch,
            max_retries=max_retries,
        )
        upsert_security_metadata(
            metadata=downloaded,
            database=database,
        )

    return load_security_metadata(
        tickers=unique_tickers,
        database=database,
    )


def _short_interest_report_date(value: object) -> date | None:
    """Convert a Yahoo short-interest timestamp into a date."""
    if value is None or pd.isna(value):
        return None

    try:
        if isinstance(value, (int, float)):
            return pd.to_datetime(value, unit="s", utc=True).date()
        return pd.Timestamp(value).date()
    except (TypeError, ValueError, OverflowError):
        return None


def _download_short_interest_batch(
    tickers: list[str],
    max_retries: int,
) -> pd.DataFrame:
    """Download latest Yahoo short interest sequentially with retries."""
    pending = list(tickers)
    records: list[dict[str, object]] = []

    for attempt in range(max_retries + 1):
        yahoo_symbols = {
            ticker: _yahoo_symbol(ticker)
            for ticker in pending
        }
        ticker_group = yf.Tickers(" ".join(yahoo_symbols.values()))
        failed: list[str] = []

        for ticker in pending:
            yahoo_symbol = yahoo_symbols[ticker]
            try:
                info = ticker_group.tickers[yahoo_symbol].get_info()
            except Exception:
                info = {}

            report_date = _short_interest_report_date(
                info.get("dateShortInterest")
            )
            shares_short = pd.to_numeric(
                info.get("sharesShort"),
                errors="coerce",
            )
            float_shares = pd.to_numeric(
                info.get("floatShares"),
                errors="coerce",
            )
            short_percent_float = pd.to_numeric(
                info.get("shortPercentOfFloat"),
                errors="coerce",
            )
            if (
                pd.isna(short_percent_float)
                and not pd.isna(shares_short)
                and not pd.isna(float_shares)
                and float_shares > 0
            ):
                short_percent_float = shares_short / float_shares

            if report_date is None or pd.isna(short_percent_float):
                failed.append(ticker)
                continue

            records.append(
                {
                    "ticker": ticker,
                    "report_date": report_date,
                    "shares_short": shares_short,
                    "float_shares": float_shares,
                    "short_percent_float": short_percent_float,
                }
            )

        pending = failed
        if not pending or attempt == max_retries:
            break

        time.sleep(RETRY_BACKOFF_SECONDS * (2**attempt))

    return pd.DataFrame.from_records(
        records,
        columns=[
            "ticker",
            "report_date",
            "shares_short",
            "float_shares",
            "short_percent_float",
        ],
    )


def get_short_interest_data(
    tickers: pd.Index,
    as_of_date: date,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_retries: int = DEFAULT_MAX_RETRIES,
    database: str | Path = PRICE_DATABASE,
    allow_download: bool = True,
) -> pd.DataFrame:
    """Return point-in-time short interest without future observations."""
    unique_tickers = pd.Index(tickers.astype(str).unique(), name="ticker")
    short_interest = load_short_interest(
        tickers=unique_tickers,
        as_of_date=as_of_date,
        database=database,
    )
    report_dates = pd.to_datetime(short_interest["report_date"])
    age_days = (
        pd.Timestamp(as_of_date) - report_dates
    ).dt.days
    stale = (
        short_interest["short_percent_float"].isna()
        | age_days.isna()
        | age_days.gt(SHORT_INTEREST_MAX_AGE_DAYS)
    )

    historical_cutoff = date.today() - timedelta(
        days=SHORT_INTEREST_MAX_AGE_DAYS
    )
    if allow_download and as_of_date >= historical_cutoff:
        stale_tickers = short_interest.index[stale].tolist()
        for start in range(0, len(stale_tickers), batch_size):
            batch = stale_tickers[start : start + batch_size]
            downloaded = _download_short_interest_batch(
                tickers=batch,
                max_retries=max_retries,
            )
            upsert_short_interest(
                short_interest=downloaded,
                database=database,
            )

        short_interest = load_short_interest(
            tickers=unique_tickers,
            as_of_date=as_of_date,
            database=database,
        )

    return short_interest


def calculate_sector_exposures(
    weights: pd.Series,
    sectors: pd.Series,
) -> pd.DataFrame:
    """Return long, short, gross, and net weights by Yahoo sector."""
    aligned_sectors = sectors.reindex(weights.index).fillna("Unclassified")
    exposure = pd.DataFrame(
        {
            "weight": weights,
            "sector": aligned_sectors,
        }
    )
    grouped = exposure.groupby("sector")["weight"]
    sector_exposures = pd.DataFrame(
        {
            "long": grouped.apply(lambda values: values.clip(lower=0).sum()),
            "short": grouped.apply(lambda values: values.clip(upper=0).sum()),
            "net": grouped.sum(),
            "gross": grouped.apply(lambda values: values.abs().sum()),
        }
    )
    return sector_exposures.sort_values("gross", ascending=False)


def _zscore(values: pd.Series) -> pd.Series:
    """Return a winsorized cross-sectional z-score."""
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.dropna()
    if len(valid) < 2:
        return pd.Series(index=values.index, dtype="float64")

    lower = valid.quantile(0.01)
    upper = valid.quantile(0.99)
    clipped = numeric.clip(lower=lower, upper=upper)
    standard_deviation = clipped.std(ddof=0)
    if standard_deviation == 0 or pd.isna(standard_deviation):
        return pd.Series(0.0, index=values.index)

    return clipped.sub(clipped.mean()).div(standard_deviation)


def calculate_amihud_illiquidity(
    adjusted_close: pd.DataFrame,
    raw_close: pd.DataFrame,
    volumes: pd.DataFrame,
    lookback_days: int = DEFAULT_LIQUIDITY_LOOKBACK_DAYS,
) -> pd.Series:
    """Return average absolute return per dollar of trading volume."""
    if lookback_days <= 0:
        raise ValueError("lookback_days must be positive.")

    aligned_raw_close = raw_close.reindex(
        index=adjusted_close.index,
        columns=adjusted_close.columns,
    )
    aligned_volumes = volumes.reindex(
        index=adjusted_close.index,
        columns=adjusted_close.columns,
    )
    returns = adjusted_close.pct_change(fill_method=None)
    dollar_volume = aligned_raw_close.mul(aligned_volumes)
    dollar_volume = dollar_volume.where(dollar_volume.gt(0))
    daily_illiquidity = returns.abs().div(dollar_volume)
    illiquidity_window = daily_illiquidity.tail(lookback_days)

    illiquidity = illiquidity_window.mean()
    required_observations = min(lookback_days, len(illiquidity_window))
    return illiquidity.where(
        illiquidity_window.count().eq(required_observations)
    ).rename("liquidity")


def calculate_relative_strength(
    adjusted_close: pd.DataFrame,
    benchmark_prices: pd.Series,
    lookback_days: int = DEFAULT_RELATIVE_STRENGTH_LOOKBACK_DAYS,
    skip_recent_days: int = DEFAULT_RELATIVE_STRENGTH_SKIP_DAYS,
) -> pd.Series:
    """Return stock performance relative to the benchmark over 63–5 days."""
    if lookback_days <= 0:
        raise ValueError("lookback_days must be positive.")
    if skip_recent_days < 0:
        raise ValueError("skip_recent_days cannot be negative.")
    if skip_recent_days >= lookback_days:
        raise ValueError(
            "skip_recent_days must be less than lookback_days."
        )

    required_prices = lookback_days + 1
    if len(adjusted_close) < required_prices:
        raise ValueError(
            f"Relative strength requires at least {required_prices} "
            "price observations."
        )

    aligned_benchmark = benchmark_prices.reindex(adjusted_close.index)
    ending_index = -(skip_recent_days + 1)
    starting_index = -required_prices
    benchmark_start = aligned_benchmark.iloc[starting_index]
    benchmark_end = aligned_benchmark.iloc[ending_index]
    if (
        pd.isna(benchmark_start)
        or pd.isna(benchmark_end)
        or benchmark_start <= 0
        or benchmark_end <= 0
    ):
        raise ValueError(
            "Benchmark prices are unavailable at the relative-strength "
            "endpoints."
        )

    stock_start = adjusted_close.iloc[starting_index]
    stock_end = adjusted_close.iloc[ending_index]
    stock_growth = stock_end.where(stock_end.gt(0)).div(
        stock_start.where(stock_start.gt(0))
    )
    benchmark_growth = benchmark_end / benchmark_start
    relative_strength = stock_growth.div(benchmark_growth).sub(1.0)
    return relative_strength.rename("relative_strength")


def calculate_short_term_reversal(
    adjusted_close: pd.DataFrame,
    maximum_lag: int = DEFAULT_SHORT_TERM_REVERSAL_MAX_LAG,
    weight_half_life_days: float = (
        DEFAULT_SHORT_TERM_REVERSAL_HALF_LIFE_DAYS
    ),
) -> pd.Series:
    """Return weighted reversal from lags two through six."""
    if maximum_lag < 2:
        raise ValueError("maximum_lag must be at least 2.")
    if weight_half_life_days <= 0:
        raise ValueError("weight_half_life_days must be positive.")

    required_prices = maximum_lag + 2
    if len(adjusted_close) < required_prices:
        raise ValueError(
            f"Short-term reversal requires at least {required_prices} "
            "price observations."
        )

    returns = adjusted_close.pct_change(fill_method=None)
    lags = np.arange(2, maximum_lag + 1)
    decay_factor = 0.5 ** (1.0 / weight_half_life_days)
    weights = decay_factor ** (lags - 2)
    weights /= weights.sum()
    lagged_returns = pd.DataFrame(
        {
            lag: returns.shift(lag).iloc[-1]
            for lag in lags
        }
    )
    reversal = lagged_returns.mul(
        weights,
        axis="columns",
    ).sum(
        axis="columns",
        min_count=len(lags),
    ).mul(-1.0)
    return reversal.rename("short_term_reversal")


def calculate_factor_scores(
    close_prices: pd.DataFrame,
    volumes: pd.DataFrame,
    market_caps: pd.Series,
    price_to_book: pd.Series,
    benchmark_prices: pd.Series | None = None,
    dollar_volume_prices: pd.DataFrame | None = None,
    short_percent_float: pd.Series | None = None,
    liquidity_lookback_days: int = DEFAULT_LIQUIDITY_LOOKBACK_DAYS,
) -> pd.DataFrame:
    """Return standardized style scores, including Amihud illiquidity."""
    if liquidity_lookback_days <= 0:
        raise ValueError("liquidity_lookback_days must be positive.")

    if dollar_volume_prices is None:
        dollar_volume_prices = close_prices
    if benchmark_prices is None:
        raise ValueError(
            "benchmark_prices are required for relative strength."
        )
    if short_percent_float is None:
        short_percent_float = pd.Series(
            index=close_prices.columns,
            dtype="float64",
        )

    numeric_market_caps = pd.to_numeric(
        market_caps,
        errors="coerce",
    ).astype("float64")
    numeric_price_to_book = pd.to_numeric(
        price_to_book,
        errors="coerce",
    ).astype("float64")
    returns = close_prices.pct_change(fill_method=None)
    relative_strength = calculate_relative_strength(
        adjusted_close=close_prices,
        benchmark_prices=benchmark_prices,
    )
    short_term_reversal = calculate_short_term_reversal(close_prices)
    numeric_short_percent_float = pd.to_numeric(
        short_percent_float,
        errors="coerce",
    ).astype("float64")

    realized_volatility = returns.std().mul(
        math.sqrt(TRADING_DAYS_PER_YEAR)
    )
    illiquidity = calculate_amihud_illiquidity(
        adjusted_close=close_prices,
        raw_close=dollar_volume_prices,
        volumes=volumes,
        lookback_days=liquidity_lookback_days,
    )
    raw_factors = pd.DataFrame(
        {
            "size": np.log(
                numeric_market_caps.where(numeric_market_caps.gt(0))
            ),
            "value": numeric_price_to_book.where(
                numeric_price_to_book.gt(0)
            ).pow(-1),
            "relative_strength": relative_strength,
            "short_interest": numeric_short_percent_float.where(
                numeric_short_percent_float.ge(0)
            ),
            "short_term_reversal": short_term_reversal,
            "volatility": realized_volatility,
            "liquidity": illiquidity,
        }
    )
    return raw_factors.apply(_zscore)


def calculate_factor_exposures(
    weights: pd.Series,
    factor_scores: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """Return signed portfolio factor exposures and ticker coverage."""
    aligned_scores = factor_scores.reindex(weights.index)
    exposures = aligned_scores.mul(weights, axis="index").sum(
        min_count=1
    )
    exposures.name = "exposure"
    coverage = aligned_scores.notna().sum()
    coverage.name = "ticker_coverage"
    return exposures, coverage


def calculate_asset_betas(
    asset_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    minimum_observations: int = DEFAULT_MINIMUM_OBSERVATIONS,
) -> pd.Series:
    """Calculate regression betas against the benchmark return series."""
    benchmark_variance = benchmark_returns.var()
    if pd.isna(benchmark_variance) or benchmark_variance <= 0:
        raise ValueError("Benchmark returns have no usable variance.")

    betas: dict[str, float] = {}
    for ticker in asset_returns.columns:
        paired = pd.concat(
            [asset_returns[ticker], benchmark_returns],
            axis="columns",
        ).dropna()
        if len(paired) < minimum_observations:
            betas[ticker] = math.nan
            continue

        betas[ticker] = paired.iloc[:, 0].cov(paired.iloc[:, 1]) / (
            paired.iloc[:, 1].var()
        )

    return pd.Series(betas, dtype="float64", name="beta")


def calculate_factor_risk_decomposition(
    weights: pd.Series,
    asset_returns: pd.DataFrame,
    factor_scores: pd.DataFrame,
    betas: pd.Series,
    sectors: pd.Series,
    minimum_observations: int = DEFAULT_MINIMUM_OBSERVATIONS,
) -> tuple[float, float, float, float, float]:
    """Decompose risk into defined-factor and residual components.

    Style scores, benchmark betas, and sector indicators form the factor
    loading matrix. Daily factor returns are estimated with cross-sectional
    least squares. Residual stock variances are treated as independent.
    """
    if minimum_observations < 2:
        raise ValueError("minimum_observations must be at least 2.")

    valid_tickers = weights.index.intersection(asset_returns.columns)
    if valid_tickers.empty:
        raise ValueError("No weighted positions have return history.")

    style_loadings = factor_scores.reindex(valid_tickers).copy()
    style_loadings = style_loadings.loc[
        :,
        style_loadings.notna().any(axis="index"),
    ]
    style_loadings = style_loadings.apply(
        pd.to_numeric,
        errors="coerce",
    ).fillna(0.0)
    style_loadings = style_loadings.add_prefix("style::")

    beta_loading = pd.to_numeric(
        betas.reindex(valid_tickers),
        errors="coerce",
    )
    loading_parts = [style_loadings]
    if beta_loading.notna().any():
        loading_parts.append(
            beta_loading.fillna(0.0).rename("beta").to_frame()
        )

    aligned_sectors = sectors.reindex(valid_tickers).fillna("Unclassified")
    sector_loadings = pd.get_dummies(
        aligned_sectors,
        prefix="sector",
        prefix_sep="::",
        dtype="float64",
    )
    loading_parts.append(sector_loadings)
    factor_loadings = pd.concat(loading_parts, axis="columns")
    factor_loadings = factor_loadings.loc[
        :,
        factor_loadings.abs().sum(axis="index").gt(0),
    ]
    if factor_loadings.empty:
        raise ValueError("No usable factor loadings are available.")

    aligned_returns = asset_returns.reindex(columns=valid_tickers)
    factor_returns = pd.DataFrame(
        index=aligned_returns.index,
        columns=factor_loadings.columns,
        dtype="float64",
    )
    residual_returns = pd.DataFrame(
        index=aligned_returns.index,
        columns=valid_tickers,
        dtype="float64",
    )

    for observation_date, daily_returns in aligned_returns.iterrows():
        usable_tickers = daily_returns.index[daily_returns.notna()]
        if len(usable_tickers) <= factor_loadings.shape[1]:
            continue

        daily_loadings = factor_loadings.loc[usable_tickers]
        estimated_returns, _, _, _ = np.linalg.lstsq(
            daily_loadings.to_numpy(dtype="float64"),
            daily_returns.loc[usable_tickers].to_numpy(dtype="float64"),
            rcond=None,
        )
        factor_returns.loc[observation_date] = estimated_returns
        fitted_returns = daily_loadings.to_numpy() @ estimated_returns
        residual_returns.loc[observation_date, usable_tickers] = (
            daily_returns.loc[usable_tickers].to_numpy() - fitted_returns
        )

    factor_returns = factor_returns.dropna(how="any")
    if len(factor_returns) < minimum_observations:
        raise ValueError(
            "Insufficient return history to estimate factor covariance."
        )

    factor_covariance = factor_returns.cov()
    aligned_weights = weights.reindex(valid_tickers).fillna(0.0)
    portfolio_factor_loadings = factor_loadings.mul(
        aligned_weights,
        axis="index",
    ).sum(axis="index")
    factor_vector = portfolio_factor_loadings.to_numpy(dtype="float64")
    systematic_variance = float(
        factor_vector
        @ factor_covariance.to_numpy(dtype="float64")
        @ factor_vector
        * TRADING_DAYS_PER_YEAR
    )

    specific_variances = residual_returns.var(
        axis="index",
        ddof=1,
    ).where(
        residual_returns.count(axis="index").ge(minimum_observations)
    )
    specific_variances = specific_variances.dropna()
    if specific_variances.empty:
        raise ValueError(
            "No positions have sufficient residual history for "
            "idiosyncratic-risk estimation."
        )

    specific_weights = aligned_weights.reindex(
        specific_variances.index
    ).fillna(0.0)
    idiosyncratic_variance = float(
        specific_weights.pow(2).mul(specific_variances).sum()
        * TRADING_DAYS_PER_YEAR
    )
    systematic_variance = max(systematic_variance, 0.0)
    idiosyncratic_variance = max(idiosyncratic_variance, 0.0)
    factor_model_variance = systematic_variance + idiosyncratic_variance

    systematic_volatility = math.sqrt(systematic_variance)
    idiosyncratic_volatility = math.sqrt(idiosyncratic_variance)
    factor_model_volatility = math.sqrt(factor_model_variance)
    if factor_model_variance == 0:
        systematic_share = 0.0
        idiosyncratic_share = 0.0
    else:
        systematic_share = systematic_variance / factor_model_variance
        idiosyncratic_share = idiosyncratic_variance / factor_model_variance

    return (
        systematic_volatility,
        idiosyncratic_volatility,
        factor_model_volatility,
        systematic_share,
        idiosyncratic_share,
    )


def calculate_covariance_risk(
    weights: pd.Series,
    asset_returns: pd.DataFrame,
    minimum_observations: int = DEFAULT_MINIMUM_OBSERVATIONS,
) -> tuple[float, pd.DataFrame]:
    """Return annual volatility and stock-level component contributions."""
    valid_tickers = asset_returns.columns[
        asset_returns.count().ge(minimum_observations)
    ]
    valid_tickers = valid_tickers.intersection(weights.index)
    if len(valid_tickers) == 0:
        raise ValueError("No positions have sufficient return history.")

    returns = asset_returns.loc[:, valid_tickers]
    returns = returns.sub(returns.mean()).fillna(0.0)
    covariance = np.cov(
        returns.to_numpy(),
        rowvar=False,
        ddof=1,
    )
    covariance = np.atleast_2d(covariance) * TRADING_DAYS_PER_YEAR
    aligned_weights = weights.reindex(valid_tickers).fillna(0.0)
    weight_vector = aligned_weights.to_numpy()
    portfolio_variance = float(
        weight_vector @ covariance @ weight_vector
    )
    portfolio_variance = max(portfolio_variance, 0.0)
    annualized_volatility = math.sqrt(portfolio_variance)

    if annualized_volatility == 0:
        component = np.zeros(len(valid_tickers))
        percentage = np.zeros(len(valid_tickers))
    else:
        marginal = covariance @ weight_vector / annualized_volatility
        component = weight_vector * marginal
        percentage = component / annualized_volatility

    contributions = pd.DataFrame(
        {
            "weight": aligned_weights,
            "component_contribution": component,
            "percentage_contribution": percentage,
        },
        index=pd.Index(valid_tickers, name="ticker"),
    ).sort_values("percentage_contribution", ascending=False)
    return annualized_volatility, contributions


def create_risk_report(
    shares: pd.Series,
    prices: pd.Series,
    market_caps: pd.Series,
    as_of_date: date,
    capital: float | None = None,
    benchmark: str = DEFAULT_BENCHMARK,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    minimum_observations: int = DEFAULT_MINIMUM_OBSERVATIONS,
    var_confidence: float = DEFAULT_VAR_CONFIDENCE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_retries: int = DEFAULT_MAX_RETRIES,
    allow_price_download: bool = True,
    portfolio_type: str = "US_live_port",
    allow_short_interest_download: bool = True,
) -> RiskReport:
    """Calculate exposures, factors, covariance risk, and parametric VaR."""
    if not 0 < var_confidence < 1:
        raise ValueError("var_confidence must be between zero and one.")

    position_exposures = calculate_position_exposures(shares, prices)
    capital_used = (
        float(position_exposures["gross"])
        if capital is None
        else float(capital)
    )
    weights = calculate_signed_weights(shares, prices, capital_used)
    history_tickers = weights.index.union(pd.Index([benchmark]))
    history_lookback_days = max(
        lookback_days,
        DEFAULT_LIQUIDITY_LOOKBACK_DAYS,
        DEFAULT_RELATIVE_STRENGTH_LOOKBACK_DAYS,
    )
    history = download_market_history(
        tickers=history_tickers,
        as_of_date=as_of_date,
        lookback_days=history_lookback_days,
        batch_size=batch_size,
        max_retries=max_retries,
        allow_download=allow_price_download,
    )
    close_prices = history.pivot(
        index="date",
        columns="ticker",
        values="close",
    ).sort_index()
    raw_close_prices = history.pivot(
        index="date",
        columns="ticker",
        values="raw_close",
    ).sort_index()
    volumes = history.pivot(
        index="date",
        columns="ticker",
        values="volume",
    ).sort_index()
    close_prices = close_prices.tail(history_lookback_days + 1)
    raw_close_prices = raw_close_prices.reindex(close_prices.index)
    volumes = volumes.reindex(close_prices.index)

    if benchmark not in close_prices:
        raise ValueError(
            f"Benchmark history is unavailable for {benchmark}."
        )

    returns = close_prices.pct_change(fill_method=None).tail(lookback_days)
    benchmark_returns = returns[benchmark]
    asset_returns = returns.drop(columns=benchmark, errors="ignore")
    betas = calculate_asset_betas(
        asset_returns=asset_returns,
        benchmark_returns=benchmark_returns,
        minimum_observations=minimum_observations,
    )
    beta_exposure = float(
        weights.reindex(betas.index).mul(betas).sum(min_count=1)
    )

    metadata = get_risk_metadata(
        tickers=weights.index,
        batch_size=batch_size,
        max_retries=max_retries,
    )
    short_interest = get_short_interest_data(
        tickers=weights.index,
        as_of_date=as_of_date,
        batch_size=batch_size,
        max_retries=max_retries,
        allow_download=allow_short_interest_download,
    )
    short_interest_coverage = int(
        short_interest["short_percent_float"].notna().sum()
    )
    if short_interest_coverage == 0:
        warnings.warn(
            "No point-in-time short-interest observations are available "
            f"on or before {as_of_date:%Y-%m-%d}; short-interest "
            "exposure will be unavailable.",
            RuntimeWarning,
            stacklevel=2,
        )
    sector_exposures = calculate_sector_exposures(
        weights=weights,
        sectors=metadata["sector"],
    )
    factor_scores = calculate_factor_scores(
        close_prices=close_prices.drop(columns=benchmark, errors="ignore"),
        volumes=volumes.drop(columns=benchmark, errors="ignore"),
        market_caps=market_caps,
        price_to_book=metadata["price_to_book"],
        benchmark_prices=close_prices[benchmark],
        short_percent_float=short_interest["short_percent_float"],
        dollar_volume_prices=raw_close_prices.drop(
            columns=benchmark,
            errors="ignore",
        ),
    )
    factor_exposures, factor_coverage = calculate_factor_exposures(
        weights=weights,
        factor_scores=factor_scores,
    )
    (
        systematic_volatility,
        idiosyncratic_volatility,
        factor_model_volatility,
        systematic_variance_share,
        idiosyncratic_variance_share,
    ) = calculate_factor_risk_decomposition(
        weights=weights,
        asset_returns=asset_returns,
        factor_scores=factor_scores,
        betas=betas,
        sectors=metadata["sector"],
        minimum_observations=minimum_observations,
    )

    annualized_volatility, risk_contributions = calculate_covariance_risk(
        weights=weights,
        asset_returns=asset_returns,
        minimum_observations=minimum_observations,
    )
    risk_contributions["beta"] = betas.reindex(
        risk_contributions.index
    )
    risk_contributions["beta_contribution"] = (
        risk_contributions["weight"]
        * risk_contributions["beta"]
    )
    risk_contributions = risk_contributions[
        [
            "weight",
            "beta",
            "beta_contribution",
            "component_contribution",
            "percentage_contribution",
        ]
    ]
    daily_volatility = annualized_volatility / math.sqrt(
        TRADING_DAYS_PER_YEAR
    )
    z_score = NormalDist().inv_cdf(var_confidence)
    parametric_var = z_score * daily_volatility * capital_used

    return RiskReport(
        portfolio_type=portfolio_type,
        as_of_date=as_of_date,
        capital=capital_used,
        long_exposure=float(position_exposures["long"]),
        short_exposure=float(position_exposures["short"]),
        gross_exposure=float(position_exposures["gross"]),
        net_exposure=float(position_exposures["net"]),
        beta_exposure=beta_exposure,
        sector_exposures=sector_exposures,
        factor_exposures=factor_exposures,
        factor_coverage=factor_coverage,
        annualized_volatility=annualized_volatility,
        daily_volatility=daily_volatility,
        systematic_volatility=systematic_volatility,
        idiosyncratic_volatility=idiosyncratic_volatility,
        factor_model_volatility=factor_model_volatility,
        systematic_variance_share=systematic_variance_share,
        idiosyncratic_variance_share=idiosyncratic_variance_share,
        parametric_var=parametric_var,
        var_confidence=var_confidence,
        covariance_coverage=len(risk_contributions),
        total_positions=len(weights),
        risk_contributions=risk_contributions,
    )


def _format_dollars(value: float) -> str:
    """Format a signed dollar amount."""
    return f"${value:,.2f}"


def format_risk_report(
    report: RiskReport,
    top_contributors: int = 20,
) -> str:
    """Return a formatted portfolio risk report."""
    exposure_rows = [
        ("Long exposure", report.long_exposure),
        ("Short exposure", report.short_exposure),
        ("Gross exposure", report.gross_exposure),
        ("Net exposure", report.net_exposure),
    ]
    lines = ["RISK METRICS", "=" * 88]
    for label, value in exposure_rows:
        lines.append(
            f"{label:<24} : {_format_dollars(value):>20} "
            f"({value / report.capital:>8.2%})"
        )

    lines.extend(
        [
            f"{'Beta exposure':<24} : {report.beta_exposure:>20.4f}",
            (
                f"{'Annualized volatility':<24} : "
                f"{report.annualized_volatility:>20.2%}"
            ),
            (
                f"{'Daily volatility':<24} : "
                f"{report.daily_volatility:>20.2%}"
            ),
            (
                f"{'Systematic volatility':<24} : "
                f"{report.systematic_volatility:>20.2%} "
                f"({report.systematic_variance_share:>7.2%} of "
                "factor-model variance)"
            ),
            (
                f"{'Idiosyncratic volatility':<24} : "
                f"{report.idiosyncratic_volatility:>20.2%} "
                f"({report.idiosyncratic_variance_share:>7.2%} of "
                "factor-model variance)"
            ),
            (
                f"{'Factor-model volatility':<24} : "
                f"{report.factor_model_volatility:>20.2%}"
            ),
            (
                f"{f'1-day {report.var_confidence:.0%} VaR':<24} : "
                f"{_format_dollars(report.parametric_var):>20}"
            ),
            (
                f"{'Covariance coverage':<24} : "
                f"{report.covariance_coverage:>10,}/"
                f"{report.total_positions:,}"
            ),
            "",
            "SECTOR EXPOSURES",
            "-" * 88,
            report.sector_exposures.to_string(
                formatters={
                    column: lambda value: f"{value:.2%}"
                    for column in report.sector_exposures.columns
                }
            ),
            "",
            "FACTOR EXPOSURES",
            "-" * 88,
        ]
    )
    factor_table = pd.concat(
        [report.factor_exposures, report.factor_coverage],
        axis="columns",
    )
    lines.append(
        factor_table.to_string(
            formatters={
                "exposure": lambda value: f"{value:+.4f}",
                "ticker_coverage": lambda value: f"{value:,.0f}",
            }
        )
    )
    lines.extend(
        [
            "",
            f"TOP {top_contributors} COMPONENT RISK CONTRIBUTIONS",
            "-" * 88,
        ]
    )
    top_risk = report.risk_contributions.head(top_contributors).copy()
    lines.append(
        top_risk.to_string(
            formatters={
                "weight": lambda value: f"{value:+.3%}",
                "beta": lambda value: f"{value:.3f}",
                "beta_contribution": lambda value: f"{value:+.4f}",
                "component_contribution": lambda value: f"{value:+.3%}",
                "percentage_contribution": lambda value: f"{value:+.2%}",
            }
        )
    )
    return "\n".join(lines)


def write_risk_contributions(
    report: RiskReport,
    output_dir: str | Path = PORTFOLIO_DATA_DIR,
) -> Path:
    """Write the complete ticker-level risk contribution table."""
    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    portfolio_label = (
        ""
        if report.portfolio_type == "US_live_port"
        else f"_{report.portfolio_type}"
    )
    output_path = output_directory / (
        f"risk_contribution{portfolio_label}_"
        f"{report.as_of_date:%Y%m%d}.csv"
    )
    report.risk_contributions.to_csv(
        output_path,
        index=True,
        index_label="ticker",
    )
    return output_path


def main() -> None:
    """Calculate and print risk metrics for a dated portfolio CSV."""
    parser = argparse.ArgumentParser(
        description="Calculate portfolio exposures and market-risk metrics."
    )
    parser.add_argument(
        "portfolio_csv",
        type=Path,
        help="Portfolio filename in port_data, or an explicit path.",
    )
    parser.add_argument("--capital", type=float)
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=DEFAULT_LOOKBACK_DAYS,
    )
    parser.add_argument(
        "--var-confidence",
        type=float,
        default=DEFAULT_VAR_CONFIDENCE,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
    )
    args = parser.parse_args()

    portfolio_path = resolve_portfolio_path(args.portfolio_csv)
    shares = load_shares(portfolio_path)
    prices = load_prices(
        portfolio_path,
        batch_size=args.batch_size,
        max_retries=args.max_retries,
    )
    as_of_date = _portfolio_date(portfolio_path)

    from port_summary import get_market_caps

    market_caps = get_market_caps(
        tickers=shares.index,
        prices=prices,
        as_of_date=as_of_date,
        batch_size=args.batch_size,
        max_retries=args.max_retries,
    )
    report = create_risk_report(
        shares=shares,
        prices=prices,
        market_caps=market_caps,
        as_of_date=as_of_date,
        portfolio_type=get_portfolio_type(portfolio_path),
        capital=args.capital,
        benchmark=args.benchmark,
        lookback_days=args.lookback_days,
        var_confidence=args.var_confidence,
        batch_size=args.batch_size,
        max_retries=args.max_retries,
    )
    print(format_risk_report(report))
    output_path = write_risk_contributions(report)
    print(f"\nRisk contributions written to: {output_path}")


if __name__ == "__main__":
    main()
