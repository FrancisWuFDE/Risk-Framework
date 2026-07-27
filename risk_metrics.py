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
    load_prices,
    load_shares,
    resolve_portfolio_path,
)
from price_cache import (
    PRICE_DATABASE,
    get_price_history,
    load_security_metadata,
    upsert_security_metadata,
)


TRADING_DAYS_PER_YEAR = 252
DEFAULT_LOOKBACK_DAYS = 252
DEFAULT_MINIMUM_OBSERVATIONS = 60
DEFAULT_BATCH_SIZE = 50
DEFAULT_MAX_RETRIES = 3
DEFAULT_VAR_CONFIDENCE = 0.95
DEFAULT_BENCHMARK = "SPY"
RETRY_BACKOFF_SECONDS = 1.0
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


def _portfolio_date(csv_file: str | Path) -> date:
    """Extract the YYYYMMDD date from a resolved portfolio filename."""
    portfolio_path = resolve_portfolio_path(csv_file)
    date_text = portfolio_path.stem.removeprefix("US_live_port_")
    return datetime.strptime(date_text, "%Y%m%d").date()


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
    """Return adjusted closes and volumes from the SQLite price cache."""
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
    history = history.drop(columns="close").rename(
        columns={"adjusted_close": "close"}
    )
    return history[["date", "ticker", "close", "volume"]]


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


def calculate_factor_scores(
    close_prices: pd.DataFrame,
    volumes: pd.DataFrame,
    market_caps: pd.Series,
    price_to_book: pd.Series,
) -> pd.DataFrame:
    """Return standardized size, value, momentum, vol, and liquidity."""
    returns = close_prices.pct_change(fill_method=None)
    momentum: dict[str, float] = {}

    for ticker in close_prices.columns:
        prices = close_prices[ticker].dropna()
        if len(prices) < 126:
            momentum[ticker] = math.nan
            continue

        recent_index = -22 if len(prices) >= 22 else -1
        old_index = -253 if len(prices) >= 253 else 0
        momentum[ticker] = (
            prices.iloc[recent_index] / prices.iloc[old_index] - 1
        )

    realized_volatility = returns.std().mul(
        math.sqrt(TRADING_DAYS_PER_YEAR)
    )
    average_dollar_volume = close_prices.mul(volumes).tail(63).mean()
    raw_factors = pd.DataFrame(
        {
            "size": np.log(market_caps.where(market_caps.gt(0))),
            "value": price_to_book.where(price_to_book.gt(0)).pow(-1),
            "momentum": pd.Series(momentum),
            "volatility": realized_volatility,
            "liquidity": np.log(
                average_dollar_volume.where(average_dollar_volume.gt(0))
            ),
        }
    )
    return raw_factors.apply(_zscore)


def calculate_factor_exposures(
    weights: pd.Series,
    factor_scores: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """Return signed portfolio factor exposures and ticker coverage."""
    aligned_scores = factor_scores.reindex(weights.index)
    exposures = aligned_scores.mul(weights, axis="index").sum()
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
    var_confidence: float = DEFAULT_VAR_CONFIDENCE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_retries: int = DEFAULT_MAX_RETRIES,
    allow_price_download: bool = True,
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
    volumes = history.pivot(
        index="date",
        columns="ticker",
        values="volume",
    ).sort_index()
    close_prices = close_prices.tail(lookback_days + 1)
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
    sector_exposures = calculate_sector_exposures(
        weights=weights,
        sectors=metadata["sector"],
    )
    factor_scores = calculate_factor_scores(
        close_prices=close_prices.drop(columns=benchmark, errors="ignore"),
        volumes=volumes.drop(columns=benchmark, errors="ignore"),
        market_caps=market_caps,
        price_to_book=metadata["price_to_book"],
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
    output_path = output_directory / (
        f"risk_contribution_{report.as_of_date:%Y%m%d}.csv"
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
