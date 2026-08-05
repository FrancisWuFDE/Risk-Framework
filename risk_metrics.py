"""Calculate portfolio exposures and market-risk metrics."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

from bloomberg_cache import (
    DEFAULT_FACTOR_UNIVERSE,
    PRICE_DATABASE,
    get_historical_market_caps,
    get_index_members,
    get_price_history,
    get_security_metadata,
)
from construct_port import (
    PORTFOLIO_DATA_DIR,
    get_portfolio_date,
    get_portfolio_output_filename,
    get_portfolio_type,
    load_prices,
    load_shares,
    resolve_portfolio_path,
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
DEFAULT_SHORT_TERM_REVERSAL_LOOKBACK_DAYS = 5
DEFAULT_SHORT_TERM_REVERSAL_SKIP_DAYS = 1
DEFAULT_SHORT_TERM_REVERSAL_HALF_LIFE_DAYS = 3.0
DEFAULT_SENTIMENT_LOOKBACK_DAYS = 20
DEFAULT_DOWNSIDE_THRESHOLD = 0.0
HISTORY_CALENDAR_MULTIPLIER = 2.25


@dataclass(frozen=True)
class RiskReport:
    """Store portfolio exposure, factor, and covariance-risk results."""

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
    parametric_var: float
    var_confidence: float
    covariance_coverage: int
    total_positions: int
    risk_contributions: pd.DataFrame
    portfolio_type: str = "US_live_port"
    factor_universe: str = DEFAULT_FACTOR_UNIVERSE
    factor_universe_size: int = 0


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
    """Return Bloomberg closes, total-return levels, and volumes."""
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
    return history[
        [
            "date",
            "ticker",
            "close",
            "total_return_index",
            "volume",
            "short_interest",
            "equity_float",
            "analyst_sentiment",
        ]
    ]


def get_risk_metadata(
    tickers: pd.Index,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_retries: int = DEFAULT_MAX_RETRIES,
    database: str | Path = PRICE_DATABASE,
    allow_download: bool = True,
) -> pd.DataFrame:
    """Return Bloomberg GICS sector and price-to-book metadata."""
    return get_security_metadata(
        tickers=tickers,
        batch_size=batch_size,
        max_retries=max_retries,
        database=database,
        allow_download=allow_download,
    )


def calculate_sector_exposures(
    weights: pd.Series,
    sectors: pd.Series,
) -> pd.DataFrame:
    """Return long, short, gross, and net weights by Bloomberg GICS sector."""
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


def _zscore(
    values: pd.Series,
    normalization_tickers: pd.Index | None = None,
) -> pd.Series:
    """Return a z-score using a fixed cross-sectional reference universe."""
    numeric = pd.to_numeric(values, errors="coerce")
    reference = (
        numeric
        if normalization_tickers is None
        else numeric.reindex(normalization_tickers)
    )
    valid_reference = reference.dropna()
    if len(valid_reference) < 2:
        return pd.Series(index=values.index, dtype="float64")

    lower = valid_reference.quantile(0.01)
    upper = valid_reference.quantile(0.99)
    clipped = numeric.clip(lower=lower, upper=upper)
    clipped_reference = valid_reference.clip(lower=lower, upper=upper)
    reference_mean = clipped_reference.mean()
    standard_deviation = clipped_reference.std(ddof=0)
    if standard_deviation == 0 or pd.isna(standard_deviation):
        return pd.Series(0.0, index=values.index)

    return clipped.sub(reference_mean).div(standard_deviation)


def calculate_amihud_illiquidity(
    total_return_levels: pd.DataFrame,
    raw_close: pd.DataFrame,
    volumes: pd.DataFrame,
    lookback_days: int = DEFAULT_LIQUIDITY_LOOKBACK_DAYS,
) -> pd.Series:
    """Return average absolute return per dollar of trading volume."""
    if lookback_days <= 0:
        raise ValueError("lookback_days must be positive.")

    aligned_close = raw_close.reindex_like(total_return_levels)
    aligned_volumes = volumes.reindex_like(total_return_levels)
    returns = total_return_levels.pct_change(fill_method=None)
    dollar_volume = aligned_close.mul(aligned_volumes)
    dollar_volume = dollar_volume.where(dollar_volume.gt(0))
    daily_illiquidity = returns.abs().div(dollar_volume)
    window = daily_illiquidity.tail(lookback_days)
    required_observations = min(lookback_days, len(window))
    return window.mean().where(
        window.count().eq(required_observations)
    ).rename("liquidity")


def calculate_relative_strength(
    total_return_levels: pd.DataFrame,
    benchmark_levels: pd.Series,
    lookback_days: int = DEFAULT_RELATIVE_STRENGTH_LOOKBACK_DAYS,
    skip_recent_days: int = DEFAULT_RELATIVE_STRENGTH_SKIP_DAYS,
) -> pd.Series:
    """Return stock total return relative to the benchmark over 63–5 days."""
    if lookback_days <= 0:
        raise ValueError("lookback_days must be positive.")
    if skip_recent_days < 0:
        raise ValueError("skip_recent_days cannot be negative.")
    if skip_recent_days >= lookback_days:
        raise ValueError(
            "skip_recent_days must be less than lookback_days."
        )

    required_levels = lookback_days + 1
    if len(total_return_levels) < required_levels:
        raise ValueError(
            f"Relative strength requires at least {required_levels} "
            "price observations."
        )

    aligned_benchmark = benchmark_levels.reindex(
        total_return_levels.index
    )
    starting_index = -required_levels
    ending_index = -(skip_recent_days + 1)
    benchmark_start = aligned_benchmark.iloc[starting_index]
    benchmark_end = aligned_benchmark.iloc[ending_index]
    if (
        pd.isna(benchmark_start)
        or pd.isna(benchmark_end)
        or benchmark_start <= 0
        or benchmark_end <= 0
    ):
        raise ValueError(
            "Benchmark levels are unavailable at the relative-strength "
            "endpoints."
        )

    stock_start = total_return_levels.iloc[starting_index]
    stock_end = total_return_levels.iloc[ending_index]
    stock_growth = stock_end.where(stock_end.gt(0)).div(
        stock_start.where(stock_start.gt(0))
    )
    benchmark_growth = benchmark_end / benchmark_start
    return stock_growth.div(benchmark_growth).sub(1.0).rename(
        "relative_strength"
    )


def calculate_one_day_reversal(
    total_return_levels: pd.DataFrame,
) -> pd.Series:
    """Return the negative of each stock's latest completed daily return."""
    if len(total_return_levels) < 2:
        raise ValueError(
            "One-day reversal requires at least two price observations."
        )

    latest_return = total_return_levels.pct_change(
        fill_method=None
    ).iloc[-1]
    return latest_return.mul(-1.0).rename("one_day_reversal")


def calculate_short_term_reversal(
    total_return_levels: pd.DataFrame,
    lookback_days: int = DEFAULT_SHORT_TERM_REVERSAL_LOOKBACK_DAYS,
    skip_recent_days: int = DEFAULT_SHORT_TERM_REVERSAL_SKIP_DAYS,
    weight_half_life_days: float = (
        DEFAULT_SHORT_TERM_REVERSAL_HALF_LIFE_DAYS
    ),
) -> pd.Series:
    """Return negative weighted returns preceding one-day reversal."""
    if lookback_days <= 0:
        raise ValueError("lookback_days must be positive.")
    if skip_recent_days < 0:
        raise ValueError("skip_recent_days cannot be negative.")
    if weight_half_life_days <= 0:
        raise ValueError("weight_half_life_days must be positive.")

    required_levels = lookback_days + skip_recent_days + 1
    if len(total_return_levels) < required_levels:
        raise ValueError(
            f"Short-term reversal requires at least {required_levels} "
            "price observations."
        )

    returns = total_return_levels.pct_change(fill_method=None)
    lags = np.arange(
        skip_recent_days,
        skip_recent_days + lookback_days,
    )
    decay_factor = 0.5 ** (1.0 / weight_half_life_days)
    weights = decay_factor ** np.arange(lookback_days)
    weights /= weights.sum()
    lagged_returns = pd.DataFrame(
        {lag: returns.shift(lag).iloc[-1] for lag in lags}
    )
    return lagged_returns.mul(weights, axis="columns").sum(
        axis="columns",
        min_count=len(lags),
    ).mul(-1.0).rename("short_term_reversal")


def latest_historical_values(
    values: pd.DataFrame,
) -> pd.Series:
    """Return each ticker's latest non-null value in the history window."""
    return values.ffill().iloc[-1]


def calculate_recommendation_sentiment(
    analyst_consensus: pd.DataFrame,
    lookback_days: int = DEFAULT_SENTIMENT_LOOKBACK_DAYS,
) -> pd.Series:
    """Return the change in analyst consensus over 20 trading days.

    Bloomberg's EQY_REC_CONS level is carried forward between changes. A
    positive result means the consensus recommendation improved over the
    lookback; a negative result means analyst opinion deteriorated.
    """
    if lookback_days <= 0:
        raise ValueError("lookback_days must be positive.")

    required_observations = lookback_days + 1
    if len(analyst_consensus) < required_observations:
        raise ValueError(
            "Recommendation sentiment requires at least "
            f"{required_observations} observations."
        )

    numeric_consensus = analyst_consensus.apply(
        pd.to_numeric,
        errors="coerce",
    ).ffill()
    current_consensus = numeric_consensus.iloc[-1]
    prior_consensus = numeric_consensus.iloc[-required_observations]
    return current_consensus.sub(prior_consensus).rename("sentiment")


def calculate_downside_variance(
    daily_returns: pd.DataFrame,
    threshold: float = DEFAULT_DOWNSIDE_THRESHOLD,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    minimum_observations: int = DEFAULT_MINIMUM_OBSERVATIONS,
) -> pd.Series:
    """Return mean squared daily shortfall below a target return."""
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite.")
    if lookback_days <= 0:
        raise ValueError("lookback_days must be positive.")
    if minimum_observations <= 0:
        raise ValueError("minimum_observations must be positive.")

    recent_returns = daily_returns.tail(lookback_days).apply(
        pd.to_numeric,
        errors="coerce",
    )
    squared_shortfall = recent_returns.sub(threshold).clip(
        upper=0.0
    ).pow(2)
    downside_variance = squared_shortfall.mean()
    valid_observations = recent_returns.notna().sum()
    return downside_variance.where(
        valid_observations.ge(minimum_observations)
    ).rename("downside_risk")


def calculate_factor_scores(
    close_prices: pd.DataFrame,
    volumes: pd.DataFrame,
    market_caps: pd.Series,
    price_to_book: pd.Series,
    benchmark_prices: pd.Series,
    short_interest: pd.Series,
    equity_float: pd.Series,
    analyst_sentiment: pd.Series,
    liquidity_prices: pd.DataFrame | None = None,
    downside_threshold: float = DEFAULT_DOWNSIDE_THRESHOLD,
    downside_lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    minimum_downside_observations: int = DEFAULT_MINIMUM_OBSERVATIONS,
    normalization_tickers: pd.Index | None = None,
) -> pd.DataFrame:
    """Return standardized Bloomberg-backed style-factor scores."""
    returns = close_prices.pct_change(fill_method=None)
    dollar_volume_prices = (
        close_prices
        if liquidity_prices is None
        else liquidity_prices.reindex_like(close_prices)
    )
    realized_volatility = returns.std().mul(
        math.sqrt(TRADING_DAYS_PER_YEAR)
    )
    relative_strength = calculate_relative_strength(
        total_return_levels=close_prices,
        benchmark_levels=benchmark_prices,
    )
    one_day_reversal = calculate_one_day_reversal(close_prices)
    short_term_reversal = calculate_short_term_reversal(close_prices)
    downside_risk = calculate_downside_variance(
        daily_returns=returns,
        threshold=downside_threshold,
        lookback_days=downside_lookback_days,
        minimum_observations=minimum_downside_observations,
    )
    illiquidity = calculate_amihud_illiquidity(
        total_return_levels=close_prices,
        raw_close=dollar_volume_prices,
        volumes=volumes,
    )
    numeric_short_interest = pd.to_numeric(
        short_interest,
        errors="coerce",
    )
    numeric_equity_float = pd.to_numeric(
        equity_float,
        errors="coerce",
    )
    short_percent_float = numeric_short_interest.where(
        numeric_short_interest.ge(0)
    ).div(numeric_equity_float.where(numeric_equity_float.gt(0)))
    raw_factors = pd.DataFrame(
        {
            "size": np.log(market_caps.where(market_caps.gt(0))),
            "value": price_to_book.where(price_to_book.gt(0)).pow(-1),
            "relative_strength": relative_strength,
            "one_day_reversal": one_day_reversal,
            "short_term_reversal": short_term_reversal,
            "short_interest": short_percent_float.where(
                short_percent_float.ge(0)
            ),
            "sentiment": pd.to_numeric(
                analyst_sentiment,
                errors="coerce",
            ),
            "volatility": realized_volatility,
            "downside_risk": downside_risk,
            "liquidity": illiquidity,
        }
    )
    return raw_factors.apply(
        _zscore,
        normalization_tickers=normalization_tickers,
    )


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
    downside_threshold: float = DEFAULT_DOWNSIDE_THRESHOLD,
    var_confidence: float = DEFAULT_VAR_CONFIDENCE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_retries: int = DEFAULT_MAX_RETRIES,
    allow_price_download: bool = True,
    portfolio_type: str = "US_live_port",
    factor_universe: str = DEFAULT_FACTOR_UNIVERSE,
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
    universe_members = get_index_members(
        index_ticker=factor_universe,
        as_of_date=as_of_date,
        max_retries=max_retries,
        allow_download=allow_price_download,
    )
    normalization_tickers = universe_members.index
    factor_tickers = normalization_tickers.union(weights.index)
    history_tickers = factor_tickers.union(pd.Index([benchmark]))
    history = download_market_history(
        tickers=history_tickers,
        as_of_date=as_of_date,
        lookback_days=lookback_days,
        batch_size=batch_size,
        max_retries=max_retries,
        allow_download=allow_price_download,
    )
    close_prices = history.pivot(
        index="date",
        columns="ticker",
        values="close",
    ).sort_index()
    return_levels = history.pivot(
        index="date",
        columns="ticker",
        values="total_return_index",
    ).sort_index()
    volumes = history.pivot(
        index="date",
        columns="ticker",
        values="volume",
    ).sort_index()
    short_interest_history = history.pivot(
        index="date",
        columns="ticker",
        values="short_interest",
    ).sort_index()
    equity_float_history = history.pivot(
        index="date",
        columns="ticker",
        values="equity_float",
    ).sort_index()
    sentiment_history = history.pivot(
        index="date",
        columns="ticker",
        values="analyst_sentiment",
    ).sort_index()
    close_prices = close_prices.tail(lookback_days + 1)
    return_levels = return_levels.reindex(close_prices.index).reindex(
        columns=close_prices.columns
    )
    volumes = volumes.reindex(close_prices.index)
    short_interest_history = short_interest_history.reindex(
        close_prices.index
    )
    equity_float_history = equity_float_history.reindex(
        close_prices.index
    )
    sentiment_history = sentiment_history.reindex(close_prices.index)

    if benchmark not in return_levels:
        raise ValueError(
            f"Benchmark history is unavailable for {benchmark}."
        )

    returns = return_levels.pct_change(
        fill_method=None
    ).tail(lookback_days)
    benchmark_returns = returns[benchmark]
    asset_returns = returns.reindex(columns=weights.index)
    betas = calculate_asset_betas(
        asset_returns=asset_returns,
        benchmark_returns=benchmark_returns,
        minimum_observations=minimum_observations,
    )
    beta_exposure = float(
        weights.reindex(betas.index).mul(betas).sum(min_count=1)
    )

    metadata = get_risk_metadata(
        tickers=factor_tickers,
        batch_size=batch_size,
        max_retries=max_retries,
        allow_download=allow_price_download,
    )
    sector_exposures = calculate_sector_exposures(
        weights=weights,
        sectors=metadata["sector"],
    )
    factor_market_caps = get_historical_market_caps(
        tickers=factor_tickers,
        as_of_date=as_of_date,
        batch_size=batch_size,
        max_retries=max_retries,
        allow_download=allow_price_download,
    )
    factor_market_caps.update(
        pd.to_numeric(market_caps, errors="coerce")
    )
    factor_return_levels = return_levels.reindex(columns=factor_tickers)
    factor_close_prices = close_prices.reindex(columns=factor_tickers)
    factor_volumes = volumes.reindex(columns=factor_tickers)
    factor_short_interest = short_interest_history.reindex(
        columns=factor_tickers
    )
    factor_equity_float = equity_float_history.reindex(
        columns=factor_tickers
    )
    factor_sentiment = sentiment_history.reindex(columns=factor_tickers)
    factor_scores = calculate_factor_scores(
        close_prices=factor_return_levels,
        volumes=factor_volumes,
        market_caps=factor_market_caps,
        price_to_book=metadata["price_to_book"],
        benchmark_prices=return_levels[benchmark],
        short_interest=latest_historical_values(
            factor_short_interest
        ),
        equity_float=latest_historical_values(
            factor_equity_float
        ),
        analyst_sentiment=calculate_recommendation_sentiment(
            factor_sentiment
        ),
        downside_threshold=downside_threshold,
        downside_lookback_days=lookback_days,
        minimum_downside_observations=minimum_observations,
        liquidity_prices=factor_close_prices,
        normalization_tickers=normalization_tickers,
    )
    factor_exposures, factor_coverage = calculate_factor_exposures(
        weights=weights,
        factor_scores=factor_scores,
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
        parametric_var=parametric_var,
        var_confidence=var_confidence,
        covariance_coverage=len(risk_contributions),
        total_positions=len(weights),
        risk_contributions=risk_contributions,
        portfolio_type=portfolio_type,
        factor_universe=factor_universe,
        factor_universe_size=len(normalization_tickers),
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
                f"{f'1-day {report.var_confidence:.0%} VaR':<24} : "
                f"{_format_dollars(report.parametric_var):>20}"
            ),
            (
                f"{'Covariance coverage':<24} : "
                f"{report.covariance_coverage:>10,}/"
                f"{report.total_positions:,}"
            ),
            (
                f"{'Factor universe':<24} : "
                f"{report.factor_universe:>20} "
                f"({report.factor_universe_size:,} members)"
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
    output_path = output_directory / get_portfolio_output_filename(
        prefix="risk_contribution",
        portfolio_type=report.portfolio_type,
        as_of_date=report.as_of_date,
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
        "--factor-universe",
        default=DEFAULT_FACTOR_UNIVERSE,
        help=(
            "Bloomberg index used to normalize custom factor scores "
            f"(default: {DEFAULT_FACTOR_UNIVERSE})."
        ),
    )
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
        "--downside-threshold",
        type=float,
        default=DEFAULT_DOWNSIDE_THRESHOLD,
        help=(
            "Daily minimum acceptable return used as downside-risk tau "
            "in decimal form (default: 0)."
        ),
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
    as_of_date = get_portfolio_date(portfolio_path)
    portfolio_type = get_portfolio_type(portfolio_path)

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
        capital=args.capital,
        benchmark=args.benchmark,
        lookback_days=args.lookback_days,
        downside_threshold=args.downside_threshold,
        var_confidence=args.var_confidence,
        batch_size=args.batch_size,
        max_retries=args.max_retries,
        portfolio_type=portfolio_type,
        factor_universe=args.factor_universe,
    )
    print(format_risk_report(report))
    output_path = write_risk_contributions(report)
    print(f"\nRisk contributions written to: {output_path}")


if __name__ == "__main__":
    main()
