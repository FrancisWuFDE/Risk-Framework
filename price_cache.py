"""Cache yfinance daily prices and volumes in a local SQLite database."""

from __future__ import annotations

import sqlite3
import time
import warnings
from contextlib import closing
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf


DEFAULT_BATCH_SIZE = 50
DEFAULT_MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 1.0
YFINANCE_DATABASE = Path(__file__).resolve().parent / "port_data" / (
    "yfinance_data.db"
)
PRICE_DATABASE = YFINANCE_DATABASE


def _yahoo_symbol(ticker: str) -> str:
    """Translate common share-class notation to Yahoo's symbol format."""
    return ticker.replace(".", "-").replace("/", "-").upper()


def initialize_price_database(
    database: str | Path = PRICE_DATABASE,
) -> Path:
    """Create the SQLite daily-price table when it does not exist."""
    database_path = Path(database)
    database_path.parent.mkdir(parents=True, exist_ok=True)

    with closing(sqlite3.connect(database_path)) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS daily_prices (
                ticker TEXT NOT NULL,
                date TEXT NOT NULL,
                close REAL,
                adjusted_close REAL,
                volume REAL,
                retrieved_at TEXT NOT NULL,
                PRIMARY KEY (ticker, date)
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_daily_prices_date
            ON daily_prices (date)
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS historical_market_caps (
                ticker TEXT NOT NULL,
                as_of_date TEXT NOT NULL,
                shares_outstanding REAL,
                market_cap REAL NOT NULL,
                retrieved_at TEXT NOT NULL,
                PRIMARY KEY (ticker, as_of_date)
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_market_caps_as_of_date
            ON historical_market_caps (as_of_date)
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS security_metadata (
                ticker TEXT PRIMARY KEY,
                sector TEXT,
                price_to_book REAL,
                retrieved_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS short_interest (
                ticker TEXT NOT NULL,
                report_date TEXT NOT NULL,
                shares_short REAL,
                float_shares REAL,
                short_percent_float REAL NOT NULL,
                retrieved_at TEXT NOT NULL,
                PRIMARY KEY (ticker, report_date)
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_short_interest_report_date
            ON short_interest (report_date)
            """
        )
        connection.commit()

    return database_path


def _extract_download_field(
    downloaded: pd.DataFrame,
    yahoo_symbol: str,
    field: str,
) -> pd.Series:
    """Extract one ticker field from either yfinance column layout."""
    if downloaded.empty:
        return pd.Series(dtype="float64")

    if isinstance(downloaded.columns, pd.MultiIndex):
        candidates = [
            (yahoo_symbol, field),
            (field, yahoo_symbol),
        ]
        for candidate in candidates:
            if candidate in downloaded.columns:
                return pd.to_numeric(
                    downloaded[candidate],
                    errors="coerce",
                )

        return pd.Series(dtype="float64")

    if field in downloaded.columns:
        return pd.to_numeric(downloaded[field], errors="coerce")

    return pd.Series(dtype="float64")


def _download_price_batch(
    tickers: list[str],
    start_date: date,
    end_date: date,
    max_retries: int,
) -> pd.DataFrame:
    """Download a sequential ticker batch for an inclusive date range."""
    pending = list(tickers)
    records: list[pd.DataFrame] = []

    for attempt in range(max_retries + 1):
        yahoo_symbols = {
            ticker: _yahoo_symbol(ticker)
            for ticker in pending
        }
        downloaded = yf.download(
            tickers=list(yahoo_symbols.values()),
            start=start_date.isoformat(),
            end=(end_date + timedelta(days=1)).isoformat(),
            auto_adjust=False,
            repair=True,
            threads=False,
            progress=False,
            group_by="ticker",
            multi_level_index=True,
        )
        failed: list[str] = []

        for ticker in pending:
            yahoo_symbol = yahoo_symbols[ticker]
            close = _extract_download_field(
                downloaded=downloaded,
                yahoo_symbol=yahoo_symbol,
                field="Close",
            )
            adjusted_close = _extract_download_field(
                downloaded=downloaded,
                yahoo_symbol=yahoo_symbol,
                field="Adj Close",
            )
            volume = _extract_download_field(
                downloaded=downloaded,
                yahoo_symbol=yahoo_symbol,
                field="Volume",
            )
            if adjusted_close.empty:
                adjusted_close = close.copy()

            ticker_history = pd.DataFrame(
                {
                    "close": close,
                    "adjusted_close": adjusted_close,
                    "volume": volume,
                }
            ).dropna(subset=["close"])
            if ticker_history.empty:
                failed.append(ticker)
                continue

            ticker_history = ticker_history.reset_index()
            ticker_history = ticker_history.rename(
                columns={ticker_history.columns[0]: "date"}
            )
            ticker_history["date"] = pd.to_datetime(
                ticker_history["date"]
            ).dt.tz_localize(None)
            ticker_history.insert(1, "ticker", ticker)
            records.append(
                ticker_history[
                    [
                        "date",
                        "ticker",
                        "close",
                        "adjusted_close",
                        "volume",
                    ]
                ]
            )

        pending = failed
        if not pending or attempt == max_retries:
            break

        time.sleep(RETRY_BACKOFF_SECONDS * (2**attempt))

    if not records:
        return pd.DataFrame(
            columns=[
                "date",
                "ticker",
                "close",
                "adjusted_close",
                "volume",
            ]
        )

    return pd.concat(records, ignore_index=True)


def _upsert_price_history(
    history: pd.DataFrame,
    database: str | Path = PRICE_DATABASE,
) -> None:
    """Insert or update downloaded rows in the SQLite price cache."""
    if history.empty:
        return

    database_path = initialize_price_database(database)
    retrieved_at = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )
    rows = [
        (
            str(row.ticker),
            pd.Timestamp(row.date).date().isoformat(),
            None if pd.isna(row.close) else float(row.close),
            (
                None
                if pd.isna(row.adjusted_close)
                else float(row.adjusted_close)
            ),
            None if pd.isna(row.volume) else float(row.volume),
            retrieved_at,
        )
        for row in history.itertuples(index=False)
    ]

    with closing(sqlite3.connect(database_path)) as connection:
        connection.executemany(
            """
            INSERT INTO daily_prices (
                ticker,
                date,
                close,
                adjusted_close,
                volume,
                retrieved_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (ticker, date) DO UPDATE SET
                close = excluded.close,
                adjusted_close = excluded.adjusted_close,
                volume = excluded.volume,
                retrieved_at = excluded.retrieved_at
            """,
            rows,
        )
        connection.commit()


def _cached_date_bounds(
    tickers: pd.Index,
    database: str | Path = PRICE_DATABASE,
) -> dict[str, tuple[date, date]]:
    """Return minimum and maximum cached dates for each ticker."""
    database_path = initialize_price_database(database)
    bounds: dict[str, tuple[date, date]] = {}

    with closing(sqlite3.connect(database_path)) as connection:
        ticker_list = tickers.astype(str).unique().tolist()
        for start in range(0, len(ticker_list), 900):
            batch = ticker_list[start : start + 900]
            placeholders = ", ".join("?" for _ in batch)
            rows = connection.execute(
                f"""
                SELECT ticker, MIN(date), MAX(date)
                FROM daily_prices
                WHERE ticker IN ({placeholders})
                GROUP BY ticker
                """,
                batch,
            ).fetchall()

            for ticker, minimum_date, maximum_date in rows:
                bounds[ticker] = (
                    date.fromisoformat(minimum_date),
                    date.fromisoformat(maximum_date),
                )

    return bounds


def ensure_price_history(
    tickers: pd.Index,
    start_date: date,
    end_date: date,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_retries: int = DEFAULT_MAX_RETRIES,
    database: str | Path = PRICE_DATABASE,
) -> None:
    """Download only leading or trailing date ranges absent from SQLite."""
    if start_date > end_date:
        raise ValueError("start_date cannot be after end_date.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if max_retries < 0:
        raise ValueError("max_retries cannot be negative.")

    unique_tickers = pd.Index(
        tickers.astype(str).unique(),
        name="ticker",
    )
    cached_bounds = _cached_date_bounds(unique_tickers, database)
    requests_by_range: dict[tuple[date, date], list[str]] = {}

    for ticker in unique_tickers:
        bounds = cached_bounds.get(ticker)
        if bounds is None:
            requests_by_range.setdefault(
                (start_date, end_date),
                [],
            ).append(ticker)
            continue

        minimum_date, maximum_date = bounds
        if start_date < minimum_date:
            leading_end = min(
                end_date,
                minimum_date - timedelta(days=1),
            )
            requests_by_range.setdefault(
                (start_date, leading_end),
                [],
            ).append(ticker)

        if end_date > maximum_date:
            trailing_start = max(
                start_date,
                maximum_date + timedelta(days=1),
            )
            requests_by_range.setdefault(
                (trailing_start, end_date),
                [],
            ).append(ticker)

    for (request_start, request_end), request_tickers in sorted(
        requests_by_range.items()
    ):
        for start in range(0, len(request_tickers), batch_size):
            batch = request_tickers[start : start + batch_size]
            downloaded = _download_price_batch(
                tickers=batch,
                start_date=request_start,
                end_date=request_end,
                max_retries=max_retries,
            )
            _upsert_price_history(downloaded, database)


def get_price_history(
    tickers: pd.Index,
    start_date: date,
    end_date: date,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_retries: int = DEFAULT_MAX_RETRIES,
    database: str | Path = PRICE_DATABASE,
    allow_download: bool = True,
) -> pd.DataFrame:
    """Return an inclusive date range from the SQLite price cache."""
    unique_tickers = pd.Index(
        tickers.astype(str).unique(),
        name="ticker",
    )
    if allow_download:
        ensure_price_history(
            tickers=unique_tickers,
            start_date=start_date,
            end_date=end_date,
            batch_size=batch_size,
            max_retries=max_retries,
            database=database,
        )
    database_path = initialize_price_database(database)
    frames: list[pd.DataFrame] = []

    with closing(sqlite3.connect(database_path)) as connection:
        ticker_list = unique_tickers.tolist()
        for start in range(0, len(ticker_list), 900):
            batch = ticker_list[start : start + 900]
            placeholders = ", ".join("?" for _ in batch)
            frame = pd.read_sql_query(
                f"""
                SELECT
                    date,
                    ticker,
                    close,
                    adjusted_close,
                    volume
                FROM daily_prices
                WHERE ticker IN ({placeholders})
                    AND date BETWEEN ? AND ?
                ORDER BY date, ticker
                """,
                connection,
                params=[
                    *batch,
                    start_date.isoformat(),
                    end_date.isoformat(),
                ],
                parse_dates=["date"],
            )
            frames.append(frame)

    if not frames:
        return pd.DataFrame(
            columns=[
                "date",
                "ticker",
                "close",
                "adjusted_close",
                "volume",
            ]
        )

    return pd.concat(frames, ignore_index=True)


def get_prices_for_date(
    tickers: pd.Index,
    as_of_date: date,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_retries: int = DEFAULT_MAX_RETRIES,
    database: str | Path = PRICE_DATABASE,
    allow_download: bool = True,
) -> pd.Series:
    """Return unadjusted closing prices for one portfolio date."""
    unique_tickers = pd.Index(
        tickers.astype(str).unique(),
        name="ticker",
    )
    history = get_price_history(
        tickers=unique_tickers,
        start_date=as_of_date,
        end_date=as_of_date,
        batch_size=batch_size,
        max_retries=max_retries,
        database=database,
        allow_download=allow_download,
    )
    prices = history.drop_duplicates("ticker", keep="last").set_index(
        "ticker"
    )["close"]
    prices = prices.reindex(unique_tickers)
    prices.name = "price"

    missing_tickers = prices.index[prices.isna()]
    if allow_download:
        for start in range(0, len(missing_tickers), batch_size):
            batch = missing_tickers[start : start + batch_size].tolist()
            downloaded = _download_price_batch(
                tickers=batch,
                start_date=as_of_date,
                end_date=as_of_date,
                max_retries=max_retries,
            )
            _upsert_price_history(downloaded, database)

    if allow_download and len(missing_tickers) > 0:
        history = get_price_history(
            tickers=unique_tickers,
            start_date=as_of_date,
            end_date=as_of_date,
            batch_size=batch_size,
            max_retries=max_retries,
            database=database,
            allow_download=False,
        )
        prices = history.drop_duplicates(
            "ticker",
            keep="last",
        ).set_index("ticker")["close"]
        prices = prices.reindex(unique_tickers)
        prices.name = "price"
        missing_tickers = prices.index[prices.isna()]

    if len(missing_tickers) > 0:
        preview = ", ".join(missing_tickers[:10])
        if len(missing_tickers) > 10:
            preview = f"{preview}, ..."
        warnings.warn(
            "SQLite contains no yfinance closing price for "
            f"{len(missing_tickers)} ticker(s) on {as_of_date}: {preview}",
            RuntimeWarning,
            stacklevel=2,
        )

    return prices


def count_cached_prices(
    database: str | Path = PRICE_DATABASE,
) -> int:
    """Return the number of cached ticker-date observations."""
    database_path = initialize_price_database(database)
    with closing(sqlite3.connect(database_path)) as connection:
        result = connection.execute(
            "SELECT COUNT(*) FROM daily_prices"
        ).fetchone()

    return int(result[0]) if result is not None else 0


def load_historical_market_caps(
    tickers: pd.Index,
    as_of_date: date,
    database: str | Path = YFINANCE_DATABASE,
) -> pd.Series:
    """Return cached market capitalizations for one historical date."""
    unique_tickers = pd.Index(
        tickers.astype(str).unique(),
        name="ticker",
    )
    database_path = initialize_price_database(database)
    frames: list[pd.DataFrame] = []

    with closing(sqlite3.connect(database_path)) as connection:
        ticker_list = unique_tickers.tolist()
        for start in range(0, len(ticker_list), 900):
            batch = ticker_list[start : start + 900]
            placeholders = ", ".join("?" for _ in batch)
            frame = pd.read_sql_query(
                f"""
                SELECT ticker, market_cap
                FROM historical_market_caps
                WHERE ticker IN ({placeholders})
                    AND as_of_date = ?
                """,
                connection,
                params=[*batch, as_of_date.isoformat()],
            )
            frames.append(frame)

    if not frames:
        return pd.Series(
            index=unique_tickers,
            dtype="float64",
            name="market_cap",
        )

    cached = pd.concat(frames, ignore_index=True)
    market_caps = cached.drop_duplicates(
        "ticker",
        keep="last",
    ).set_index("ticker")["market_cap"]
    return market_caps.reindex(unique_tickers).rename("market_cap")


def upsert_historical_market_caps(
    market_caps: pd.Series,
    as_of_date: date,
    prices: pd.Series | None = None,
    database: str | Path = YFINANCE_DATABASE,
) -> None:
    """Store successful historical market-cap observations in SQLite."""
    successful = pd.to_numeric(market_caps, errors="coerce").dropna()
    successful = successful.loc[successful.gt(0)]
    if successful.empty:
        return

    aligned_prices = (
        prices.reindex(successful.index)
        if prices is not None
        else pd.Series(index=successful.index, dtype="float64")
    )
    retrieved_at = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )
    rows = []
    for ticker, market_cap in successful.items():
        price = aligned_prices.get(ticker)
        shares_outstanding = (
            float(market_cap / price)
            if pd.notna(price) and float(price) > 0
            else None
        )
        rows.append(
            (
                str(ticker),
                as_of_date.isoformat(),
                shares_outstanding,
                float(market_cap),
                retrieved_at,
            )
        )

    database_path = initialize_price_database(database)
    with closing(sqlite3.connect(database_path)) as connection:
        connection.executemany(
            """
            INSERT INTO historical_market_caps (
                ticker,
                as_of_date,
                shares_outstanding,
                market_cap,
                retrieved_at
            )
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (ticker, as_of_date) DO UPDATE SET
                shares_outstanding = excluded.shares_outstanding,
                market_cap = excluded.market_cap,
                retrieved_at = excluded.retrieved_at
            """,
            rows,
        )
        connection.commit()


def load_security_metadata(
    tickers: pd.Index,
    database: str | Path = YFINANCE_DATABASE,
) -> pd.DataFrame:
    """Return cached sector and price-to-book metadata."""
    unique_tickers = pd.Index(
        tickers.astype(str).unique(),
        name="ticker",
    )
    database_path = initialize_price_database(database)
    frames: list[pd.DataFrame] = []

    with closing(sqlite3.connect(database_path)) as connection:
        ticker_list = unique_tickers.tolist()
        for start in range(0, len(ticker_list), 900):
            batch = ticker_list[start : start + 900]
            placeholders = ", ".join("?" for _ in batch)
            frame = pd.read_sql_query(
                f"""
                SELECT ticker, sector, price_to_book, retrieved_at
                FROM security_metadata
                WHERE ticker IN ({placeholders})
                """,
                connection,
                params=batch,
            )
            frames.append(frame)

    if not frames:
        return pd.DataFrame(
            index=unique_tickers,
            columns=["sector", "price_to_book", "retrieved_at"],
        )

    metadata = pd.concat(frames, ignore_index=True)
    metadata = metadata.drop_duplicates(
        "ticker",
        keep="last",
    ).set_index("ticker")
    return metadata.reindex(unique_tickers)


def upsert_security_metadata(
    metadata: pd.DataFrame,
    database: str | Path = YFINANCE_DATABASE,
) -> None:
    """Store successful sector and valuation metadata in SQLite."""
    if metadata.empty:
        return

    frame = metadata.copy()
    if "ticker" not in frame.columns:
        frame = frame.reset_index()
        frame = frame.rename(columns={frame.columns[0]: "ticker"})
    retrieved_at = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )
    rows = [
        (
            str(row.ticker),
            None if pd.isna(row.sector) else str(row.sector),
            (
                None
                if pd.isna(row.price_to_book)
                else float(row.price_to_book)
            ),
            (
                retrieved_at
                if not hasattr(row, "retrieved_at")
                or pd.isna(row.retrieved_at)
                else str(row.retrieved_at)
            ),
        )
        for row in frame.itertuples(index=False)
    ]

    database_path = initialize_price_database(database)
    with closing(sqlite3.connect(database_path)) as connection:
        connection.executemany(
            """
            INSERT INTO security_metadata (
                ticker,
                sector,
                price_to_book,
                retrieved_at
            )
            VALUES (?, ?, ?, ?)
            ON CONFLICT (ticker) DO UPDATE SET
                sector = excluded.sector,
                price_to_book = excluded.price_to_book,
                retrieved_at = excluded.retrieved_at
            """,
            rows,
        )
        connection.commit()


def load_short_interest(
    tickers: pd.Index,
    as_of_date: date,
    database: str | Path = YFINANCE_DATABASE,
) -> pd.DataFrame:
    """Return each ticker's latest reported short interest by a date."""
    unique_tickers = pd.Index(
        tickers.astype(str).unique(),
        name="ticker",
    )
    database_path = initialize_price_database(database)
    frames: list[pd.DataFrame] = []

    with closing(sqlite3.connect(database_path)) as connection:
        ticker_list = unique_tickers.tolist()
        for start in range(0, len(ticker_list), 900):
            batch = ticker_list[start : start + 900]
            placeholders = ", ".join("?" for _ in batch)
            frame = pd.read_sql_query(
                f"""
                SELECT
                    ticker,
                    report_date,
                    shares_short,
                    float_shares,
                    short_percent_float,
                    retrieved_at
                FROM short_interest
                WHERE ticker IN ({placeholders})
                    AND report_date <= ?
                ORDER BY ticker, report_date
                """,
                connection,
                params=[*batch, as_of_date.isoformat()],
            )
            frames.append(frame)

    columns = [
        "report_date",
        "shares_short",
        "float_shares",
        "short_percent_float",
        "retrieved_at",
    ]
    if not frames:
        return pd.DataFrame(index=unique_tickers, columns=columns)

    short_interest = pd.concat(frames, ignore_index=True)
    if short_interest.empty:
        return pd.DataFrame(index=unique_tickers, columns=columns)

    short_interest = short_interest.drop_duplicates(
        "ticker",
        keep="last",
    ).set_index("ticker")
    short_interest["report_date"] = pd.to_datetime(
        short_interest["report_date"]
    ).dt.date
    return short_interest.reindex(unique_tickers)


def upsert_short_interest(
    short_interest: pd.DataFrame,
    database: str | Path = YFINANCE_DATABASE,
) -> None:
    """Store successful reported short-interest snapshots in SQLite."""
    if short_interest.empty:
        return

    frame = short_interest.copy()
    if "ticker" not in frame.columns:
        frame = frame.reset_index()
        frame = frame.rename(columns={frame.columns[0]: "ticker"})
    retrieved_at = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )
    rows = [
        (
            str(row.ticker),
            pd.Timestamp(row.report_date).date().isoformat(),
            None if pd.isna(row.shares_short) else float(row.shares_short),
            None if pd.isna(row.float_shares) else float(row.float_shares),
            float(row.short_percent_float),
            retrieved_at,
        )
        for row in frame.itertuples(index=False)
        if not pd.isna(row.report_date)
        and not pd.isna(row.short_percent_float)
    ]
    if not rows:
        return

    database_path = initialize_price_database(database)
    with closing(sqlite3.connect(database_path)) as connection:
        connection.executemany(
            """
            INSERT INTO short_interest (
                ticker,
                report_date,
                shares_short,
                float_shares,
                short_percent_float,
                retrieved_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (ticker, report_date) DO UPDATE SET
                shares_short = excluded.shares_short,
                float_shares = excluded.float_shares,
                short_percent_float = excluded.short_percent_float,
                retrieved_at = excluded.retrieved_at
            """,
            rows,
        )
        connection.commit()
