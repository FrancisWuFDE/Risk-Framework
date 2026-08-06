"""Retrieve Bloomberg data and cache it in a local SQLite database."""

from __future__ import annotations

import argparse
import importlib
import math
import re
import sqlite3
import time
import warnings
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterator
from xml.etree import ElementTree
from zipfile import BadZipFile, ZipFile

import pandas as pd


DEFAULT_BATCH_SIZE = 50
DEFAULT_MAX_RETRIES = 3
DEFAULT_CACHE_LOOKBACK_DAYS = 252
DEFAULT_BENCHMARK = "VTHR"
DEFAULT_FACTOR_UNIVERSE = "B3000 Index"
HISTORY_CALENDAR_MULTIPLIER = 2.25
RETRY_BACKOFF_SECONDS = 1.0
HISTORICAL_FIELDS_VERSION = 1
BLOOMBERG_HOST = "localhost"
BLOOMBERG_PORT = 8194
BLOOMBERG_SERVICE = "//blp/refdata"
RESPONSE_TIMEOUT_MILLISECONDS = 120_000
BLOOMBERG_DATABASE = (
    Path(__file__).resolve().parent / "port_data" / "bloomberg_data.db"
)
PRICE_DATABASE = BLOOMBERG_DATABASE

HISTORICAL_FIELDS = [
    "PX_LAST",
    "TOT_RETURN_INDEX_GROSS_DVDS",
    "PX_VOLUME",
    "SHORT_INT",
    "EQY_FLOAT",
    "EQY_REC_CONS",
]
MARKET_CAP_FIELD = "CUR_MKT_CAP"
MARKET_CAP_MULTIPLIER = 1_000_000.0
METADATA_FIELDS = ["GICS_SECTOR_NAME", "PX_TO_BOOK_RATIO"]
INDEX_MEMBERS_FIELD = "INDX_MWEIGHT_HIST"
INDEX_MEMBERS_DATE_OVERRIDE = "END_DATE_OVERRIDE"
KNOWN_SECURITY_TYPES = {
    "Comdty",
    "Corp",
    "Curncy",
    "Equity",
    "Govt",
    "Index",
    "M-Mkt",
    "Mtge",
    "Muni",
    "Pfd",
}
FIGI_PATTERN = re.compile(r"BBG[0-9A-Z]{9}")
SPREADSHEET_NAMESPACE = (
    "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
)
OFFICE_RELATIONSHIPS_NAMESPACE = (
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
)
PACKAGE_RELATIONSHIPS_NAMESPACE = (
    "http://schemas.openxmlformats.org/package/2006/relationships"
)


@dataclass(frozen=True)
class FactorUniverseSnapshot:
    """Represent one dated factor-universe membership snapshot."""

    as_of_date: date
    benchmark_name: str
    members: pd.Series


def _load_blpapi() -> Any:
    """Import Bloomberg's Python package or raise an actionable error."""
    try:
        return importlib.import_module("blpapi")
    except ImportError as error:
        raise RuntimeError(
            "Bloomberg's blpapi package is required. Install it from the "
            "Bloomberg package index and run with Bloomberg Terminal open."
        ) from error


def _bloomberg_security(ticker: str) -> str:
    """Translate a portfolio ticker into a Bloomberg security identifier."""
    normalized = str(ticker).strip()
    if not normalized:
        raise ValueError("Ticker values cannot be blank.")

    if normalized.startswith("/"):
        return normalized
    if FIGI_PATTERN.fullmatch(normalized.upper()):
        return f"/bbgid/{normalized.upper()}"

    final_token = normalized.rsplit(maxsplit=1)[-1]
    if final_token in KNOWN_SECURITY_TYPES:
        return normalized

    return f"{normalized} US Equity"


def _normalize_payload(value: Any) -> list[Any]:
    """Return one Bloomberg response element as a list."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _as_float(value: Any) -> float:
    """Convert a Bloomberg value to a finite float or NaN."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return math.nan

    return numeric if math.isfinite(numeric) else math.nan


def _resolve_requested_identifier(
    security_data: dict[str, Any],
    securities: dict[str, str],
    requested_identifiers: list[str],
) -> str | None:
    """Map a Bloomberg response back to the caller's identifier."""
    response_security = str(security_data.get("security", ""))
    identifier = securities.get(response_security)
    if identifier is not None:
        return identifier

    sequence_number = security_data.get("sequenceNumber")
    try:
        sequence_index = int(sequence_number)
    except (TypeError, ValueError):
        return None
    if 0 <= sequence_index < len(requested_identifiers):
        return requested_identifiers[sequence_index]
    return None


@contextmanager
def _reference_data_session() -> Iterator[tuple[Any, Any, Any]]:
    """Yield a running Bloomberg Desktop API reference-data session."""
    blpapi = _load_blpapi()
    options = blpapi.SessionOptions()
    options.setServerHost(BLOOMBERG_HOST)
    options.setServerPort(BLOOMBERG_PORT)
    options.setClientMode(blpapi.SessionOptions.DAPI)
    options.setNumStartAttempts(1)
    session = blpapi.Session(options)

    if not session.start():
        raise RuntimeError(
            "Could not start a Bloomberg Desktop API session. Confirm that "
            "Bloomberg Terminal is open and logged in."
        )

    try:
        if not session.openService(BLOOMBERG_SERVICE):
            raise RuntimeError(
                f"Could not open Bloomberg service {BLOOMBERG_SERVICE}."
            )
        yield blpapi, session, session.getService(BLOOMBERG_SERVICE)
    finally:
        session.stop()


def _send_request(
    blpapi: Any,
    session: Any,
    request: Any,
) -> list[dict[str, Any]]:
    """Send a synchronous Bloomberg request and decode its messages."""
    session.sendRequest(request)
    payloads: list[dict[str, Any]] = []

    while True:
        event = session.nextEvent(RESPONSE_TIMEOUT_MILLISECONDS)
        event_type = event.eventType()
        if event_type == blpapi.Event.TIMEOUT:
            raise TimeoutError("Bloomberg request timed out.")

        for message in event:
            payload = message.toPy()
            if not isinstance(payload, dict):
                continue
            if payload.get("responseError"):
                raise RuntimeError(
                    f"Bloomberg response error: {payload['responseError']}"
                )
            payloads.append(payload)

        if event_type == blpapi.Event.RESPONSE:
            return payloads


def _historical_data_request(
    tickers: list[str],
    fields: list[str],
    start_date: date,
    end_date: date,
) -> list[dict[str, Any]]:
    """Request daily historical Bloomberg fields for a ticker batch."""
    requested_identifiers = list(tickers)
    securities = {
        _bloomberg_security(ticker): ticker
        for ticker in requested_identifiers
    }

    with _reference_data_session() as (blpapi, session, service):
        request = service.createRequest("HistoricalDataRequest")
        for security in securities:
            request.append("securities", security)
        for field in fields:
            request.append("fields", field)
        request.set("startDate", start_date.strftime("%Y%m%d"))
        request.set("endDate", end_date.strftime("%Y%m%d"))
        request.set("periodicitySelection", "DAILY")
        request.set("maxDataPoints", 100_000)
        payloads = _send_request(blpapi, session, request)

    records: list[dict[str, Any]] = []
    for payload in payloads:
        for security_data in _normalize_payload(
            payload.get("securityData")
        ):
            if not isinstance(security_data, dict):
                continue
            ticker = _resolve_requested_identifier(
                security_data=security_data,
                securities=securities,
                requested_identifiers=requested_identifiers,
            )
            if ticker is None or security_data.get("securityError"):
                continue

            for field_data in _normalize_payload(
                security_data.get("fieldData")
            ):
                if not isinstance(field_data, dict):
                    continue
                record = {"ticker": ticker}
                record.update(field_data)
                records.append(record)

    return records


def _reference_data_request(
    tickers: list[str],
    fields: list[str],
) -> list[dict[str, Any]]:
    """Request Bloomberg reference fields for a ticker batch."""
    requested_identifiers = list(tickers)
    securities = {
        _bloomberg_security(ticker): ticker
        for ticker in requested_identifiers
    }

    with _reference_data_session() as (blpapi, session, service):
        request = service.createRequest("ReferenceDataRequest")
        for security in securities:
            request.append("securities", security)
        for field in fields:
            request.append("fields", field)
        payloads = _send_request(blpapi, session, request)

    records: list[dict[str, Any]] = []
    for payload in payloads:
        for security_data in _normalize_payload(
            payload.get("securityData")
        ):
            if not isinstance(security_data, dict):
                continue
            ticker = _resolve_requested_identifier(
                security_data=security_data,
                securities=securities,
                requested_identifiers=requested_identifiers,
            )
            field_data = security_data.get("fieldData", {})
            if (
                ticker is None
                or security_data.get("securityError")
                or not isinstance(field_data, dict)
            ):
                continue
            record = {"ticker": ticker}
            record.update(field_data)
            records.append(record)

    return records


def _index_members_request(
    index_ticker: str,
    as_of_date: date,
) -> list[dict[str, Any]]:
    """Request historical index membership from Bloomberg."""
    with _reference_data_session() as (blpapi, session, service):
        request = service.createRequest("ReferenceDataRequest")
        request.append("securities", _bloomberg_security(index_ticker))
        request.append("fields", INDEX_MEMBERS_FIELD)
        override = request.getElement("overrides").appendElement()
        override.setElement("fieldId", INDEX_MEMBERS_DATE_OVERRIDE)
        override.setElement("value", as_of_date.strftime("%Y%m%d"))
        payloads = _send_request(blpapi, session, request)

    members: list[dict[str, Any]] = []
    for payload in payloads:
        for security_data in _normalize_payload(
            payload.get("securityData")
        ):
            if (
                not isinstance(security_data, dict)
                or security_data.get("securityError")
            ):
                continue
            field_data = security_data.get("fieldData", {})
            if not isinstance(field_data, dict):
                continue
            for member in _normalize_payload(
                field_data.get(INDEX_MEMBERS_FIELD)
            ):
                if isinstance(member, dict):
                    members.append(member)

    return members


def _normalize_index_member_ticker(value: Any) -> str | None:
    """Convert a Bloomberg index-member identifier to a portfolio ticker."""
    if value is None:
        return None

    tokens = str(value).strip().split()
    if not tokens:
        return None
    if tokens[-1] in KNOWN_SECURITY_TYPES:
        tokens.pop()
    if len(tokens) > 1 and len(tokens[-1]) == 2:
        tokens.pop()
    return " ".join(tokens) or None


def _parse_index_members(records: list[dict[str, Any]]) -> pd.Series:
    """Return normalized tickers and weights from a Bloomberg bulk field."""
    member_fields = (
        "Member Ticker and Exchange Code",
        "Member Ticker",
        "Index Member",
        "Member",
        "Security",
    )
    weight_fields = (
        "Percent Weight",
        "Percentage Weight",
        "Weight",
    )
    weights: dict[str, float] = {}

    for record in records:
        raw_member = next(
            (record.get(field) for field in member_fields if field in record),
            None,
        )
        ticker = _normalize_index_member_ticker(raw_member)
        if ticker is None:
            continue
        raw_weight = next(
            (record.get(field) for field in weight_fields if field in record),
            None,
        )
        weights[ticker] = _as_float(raw_weight)

    return pd.Series(
        weights,
        index=pd.Index(weights, name="ticker"),
        dtype="float64",
        name="index_weight",
    )


def _xlsx_column_number(cell_reference: str) -> int:
    """Return a one-based column number from an Excel cell reference."""
    match = re.match(r"[A-Z]+", cell_reference.upper())
    if match is None:
        raise ValueError(
            f"Invalid Excel cell reference: {cell_reference!r}."
        )

    column_number = 0
    for character in match.group(0):
        column_number = column_number * 26 + ord(character) - 64
    return column_number


def _xlsx_cell_value(
    cell: ElementTree.Element,
    shared_strings: list[str],
) -> Any:
    """Decode one cell from the XML inside an XLSX workbook."""
    namespace = {"main": SPREADSHEET_NAMESPACE}
    cell_type = cell.get("t")
    if cell_type == "inlineStr":
        return "".join(
            text.text or ""
            for text in cell.findall(".//main:t", namespace)
        )

    value = cell.find("main:v", namespace)
    if value is None or value.text is None:
        return None
    if cell_type == "s":
        return shared_strings[int(value.text)]
    if cell_type in {"str", "e"}:
        return value.text
    if cell_type == "b":
        return value.text == "1"

    try:
        return float(value.text)
    except ValueError:
        return value.text


def _read_xlsx_worksheets(
    workbook_path: Path,
) -> list[tuple[str, list[tuple[int, dict[int, Any]]]]]:
    """Read worksheet cell values using only the Python standard library."""
    if not workbook_path.is_file():
        raise FileNotFoundError(
            f"Factor-universe workbook does not exist: {workbook_path}"
        )
    if workbook_path.suffix.lower() != ".xlsx":
        raise ValueError(
            "Factor-universe workbooks must use the .xlsx format."
        )

    spreadsheet_namespace = {"main": SPREADSHEET_NAMESPACE}
    package_namespace = {"package": PACKAGE_RELATIONSHIPS_NAMESPACE}
    relationship_attribute = (
        f"{{{OFFICE_RELATIONSHIPS_NAMESPACE}}}id"
    )

    try:
        with ZipFile(workbook_path) as workbook:
            shared_strings: list[str] = []
            if "xl/sharedStrings.xml" in workbook.namelist():
                shared_root = ElementTree.fromstring(
                    workbook.read("xl/sharedStrings.xml")
                )
                shared_strings = [
                    "".join(
                        text.text or ""
                        for text in item.findall(
                            ".//main:t",
                            spreadsheet_namespace,
                        )
                    )
                    for item in shared_root.findall(
                        "main:si",
                        spreadsheet_namespace,
                    )
                ]

            workbook_root = ElementTree.fromstring(
                workbook.read("xl/workbook.xml")
            )
            relationship_root = ElementTree.fromstring(
                workbook.read("xl/_rels/workbook.xml.rels")
            )
            relationship_targets = {
                relationship.get("Id"): relationship.get("Target")
                for relationship in relationship_root.findall(
                    "package:Relationship",
                    package_namespace,
                )
            }

            worksheets: list[
                tuple[str, list[tuple[int, dict[int, Any]]]]
            ] = []
            for sheet in workbook_root.findall(
                "main:sheets/main:sheet",
                spreadsheet_namespace,
            ):
                relationship_id = sheet.get(relationship_attribute)
                target = relationship_targets.get(relationship_id)
                if target is None:
                    continue
                target = target.replace("\\", "/")
                if target.startswith("/"):
                    worksheet_name = target.lstrip("/")
                elif target.startswith("xl/"):
                    worksheet_name = target
                else:
                    worksheet_name = f"xl/{target}"

                worksheet_root = ElementTree.fromstring(
                    workbook.read(worksheet_name)
                )
                rows: list[tuple[int, dict[int, Any]]] = []
                for row in worksheet_root.findall(
                    "main:sheetData/main:row",
                    spreadsheet_namespace,
                ):
                    row_number = int(row.get("r", "0"))
                    values: dict[int, Any] = {}
                    for cell in row.findall(
                        "main:c",
                        spreadsheet_namespace,
                    ):
                        cell_reference = cell.get("r")
                        if cell_reference is None:
                            continue
                        values[_xlsx_column_number(cell_reference)] = (
                            _xlsx_cell_value(cell, shared_strings)
                        )
                    rows.append((row_number, values))
                worksheets.append((sheet.get("name", "Worksheet"), rows))
    except (BadZipFile, ElementTree.ParseError, KeyError) as error:
        raise ValueError(
            f"Could not read XLSX workbook {workbook_path}: {error}"
        ) from error

    if not worksheets:
        raise ValueError(
            f"Factor-universe workbook has no worksheets: {workbook_path}"
        )
    return worksheets


def _normalized_header(value: Any) -> str:
    """Normalize a workbook label for case-insensitive matching."""
    return " ".join(str(value).strip().lower().split())


def _find_workbook_cell(
    rows: list[tuple[int, dict[int, Any]]],
    accepted_values: set[str],
) -> tuple[int, int] | None:
    """Locate the first cell whose normalized value matches a label."""
    for row_number, values in rows:
        for column_number, value in values.items():
            if _normalized_header(value) in accepted_values:
                return row_number, column_number
    return None


def _metadata_value_below(
    rows: list[tuple[int, dict[int, Any]]],
    label: str,
) -> Any:
    """Return the value directly below one Bloomberg export header."""
    location = _find_workbook_cell(rows, {_normalized_header(label)})
    if location is None:
        raise ValueError(
            f"Factor-universe workbook is missing the {label!r} header."
        )

    row_number, column_number = location
    rows_by_number = dict(rows)
    value = rows_by_number.get(row_number + 1, {}).get(column_number)
    if value is None:
        raise ValueError(
            f"Factor-universe workbook has no value below {label!r}."
        )
    return value


def _parse_workbook_date(value: Any) -> date:
    """Parse a Bloomberg export date stored as text or an Excel serial."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return (date(1899, 12, 30) + timedelta(days=int(value)))

    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(
            f"Could not parse factor-universe as-of date {value!r}."
        )
    return pd.Timestamp(parsed).date()


def _canonical_universe_name(value: Any) -> str:
    """Return a comparison key for names such as B3000 and B3000 Index."""
    tokens = re.findall(r"[A-Z0-9]+", str(value).upper())
    if tokens and tokens[-1] == "INDEX":
        tokens.pop()
    return "".join(tokens)


def load_factor_universe_workbook(
    workbook_path: str | Path,
) -> FactorUniverseSnapshot:
    """Load positive benchmark weights and FIGIs from a Bloomberg export."""
    resolved_path = Path(workbook_path).expanduser()
    worksheets = _read_xlsx_worksheets(resolved_path)

    selected_rows: list[tuple[int, dict[int, Any]]] | None = None
    figi_location: tuple[int, int] | None = None
    weight_location: tuple[int, int] | None = None
    for _, rows in worksheets:
        candidate_figi = _find_workbook_cell(rows, {"figi"})
        candidate_weight = _find_workbook_cell(
            rows,
            {"bmrk", "benchmark weight", "benchmark % weight"},
        )
        if candidate_figi is not None and candidate_weight is not None:
            selected_rows = rows
            figi_location = candidate_figi
            weight_location = candidate_weight
            break

    if (
        selected_rows is None
        or figi_location is None
        or weight_location is None
    ):
        raise ValueError(
            "Factor-universe workbook must contain FIGI and Bmrk columns."
        )

    as_of_date = _parse_workbook_date(
        _metadata_value_below(selected_rows, "As Of Date")
    )
    benchmark_name = str(
        _metadata_value_below(selected_rows, "Benchmark Name")
    ).strip()
    first_data_row = max(figi_location[0], weight_location[0]) + 1
    figi_column = figi_location[1]
    weight_column = weight_location[1]

    observed_figis: set[str] = set()
    member_weights: dict[str, float] = {}
    for row_number, values in selected_rows:
        if row_number < first_data_row:
            continue
        figi = str(values.get(figi_column, "")).strip().upper()
        if not FIGI_PATTERN.fullmatch(figi):
            continue
        if figi in observed_figis:
            raise ValueError(
                f"Duplicate FIGI {figi} in factor-universe workbook."
            )
        observed_figis.add(figi)

        index_weight = _as_float(values.get(weight_column))
        if not math.isnan(index_weight) and index_weight > 0:
            member_weights[figi] = index_weight

    if not member_weights:
        raise ValueError(
            "Factor-universe workbook contains no positive benchmark "
            "weights with valid FIGIs."
        )

    members = pd.Series(
        member_weights,
        index=pd.Index(member_weights, name="ticker"),
        dtype="float64",
        name="index_weight",
    ).sort_index()
    total_weight = float(members.sum())
    if 0.99 <= total_weight <= 1.01:
        members = members.mul(100.0)
        total_weight = float(members.sum())
    if not 99.0 <= total_weight <= 101.0:
        raise ValueError(
            "Benchmark weights selected from the factor-universe workbook "
            f"sum to {total_weight:.6f}, not approximately 100."
        )

    return FactorUniverseSnapshot(
        as_of_date=as_of_date,
        benchmark_name=benchmark_name,
        members=members,
    )


def import_factor_universe_workbook(
    workbook_path: str | Path,
    index_ticker: str = DEFAULT_FACTOR_UNIVERSE,
    database: str | Path = BLOOMBERG_DATABASE,
    expected_as_of_date: date | None = None,
) -> FactorUniverseSnapshot:
    """Replace one cached universe date with a FIGI workbook snapshot."""
    snapshot = load_factor_universe_workbook(workbook_path)
    if (
        expected_as_of_date is not None
        and snapshot.as_of_date != expected_as_of_date
    ):
        raise ValueError(
            f"Workbook as-of date {snapshot.as_of_date} does not match "
            f"portfolio date {expected_as_of_date}."
        )
    expected_name = _canonical_universe_name(index_ticker)
    workbook_name = _canonical_universe_name(snapshot.benchmark_name)
    if expected_name != workbook_name:
        raise ValueError(
            f"Workbook benchmark {snapshot.benchmark_name!r} does not "
            f"match --factor-universe {index_ticker!r}."
        )

    replace_index_members(
        index_ticker=index_ticker,
        as_of_date=snapshot.as_of_date,
        members=snapshot.members,
        database=database,
    )
    return snapshot


def initialize_price_database(
    database: str | Path = PRICE_DATABASE,
) -> Path:
    """Create Bloomberg cache tables when they do not exist."""
    database_path = Path(database)
    database_path.parent.mkdir(parents=True, exist_ok=True)

    with closing(sqlite3.connect(database_path)) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS daily_prices (
                ticker TEXT NOT NULL,
                date TEXT NOT NULL,
                close REAL,
                total_return_index REAL,
                volume REAL,
                short_interest REAL,
                equity_float REAL,
                analyst_sentiment REAL,
                historical_fields_version INTEGER NOT NULL DEFAULT 0,
                retrieved_at TEXT NOT NULL,
                PRIMARY KEY (ticker, date)
            )
            """
        )
        existing_columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(daily_prices)"
            ).fetchall()
        }
        new_columns = {
            "short_interest": "REAL",
            "equity_float": "REAL",
            "analyst_sentiment": "REAL",
            "historical_fields_version": (
                "INTEGER NOT NULL DEFAULT 0"
            ),
        }
        for column, data_type in new_columns.items():
            if column not in existing_columns:
                connection.execute(
                    f"ALTER TABLE daily_prices "
                    f"ADD COLUMN {column} {data_type}"
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
            CREATE TABLE IF NOT EXISTS index_memberships (
                index_ticker TEXT NOT NULL,
                as_of_date TEXT NOT NULL,
                member_ticker TEXT NOT NULL,
                index_weight REAL,
                retrieved_at TEXT NOT NULL,
                PRIMARY KEY (index_ticker, as_of_date, member_ticker)
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_index_memberships_date
            ON index_memberships (index_ticker, as_of_date)
            """
        )
        connection.commit()

    return database_path


def load_index_members(
    index_ticker: str,
    as_of_date: date,
    database: str | Path = BLOOMBERG_DATABASE,
) -> pd.Series:
    """Return cached members and weights for one index date."""
    database_path = initialize_price_database(database)
    with closing(sqlite3.connect(database_path)) as connection:
        frame = pd.read_sql_query(
            """
            SELECT member_ticker, index_weight
            FROM index_memberships
            WHERE index_ticker = ? AND as_of_date = ?
            ORDER BY member_ticker
            """,
            connection,
            params=[index_ticker, as_of_date.isoformat()],
        )

    if frame.empty:
        return pd.Series(dtype="float64", name="index_weight")
    return frame.set_index("member_ticker")["index_weight"].rename_axis(
        "ticker"
    )


def upsert_index_members(
    index_ticker: str,
    as_of_date: date,
    members: pd.Series,
    database: str | Path = BLOOMBERG_DATABASE,
) -> None:
    """Store one dated index-membership snapshot."""
    if members.empty:
        return

    retrieved_at = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )
    rows = [
        (
            index_ticker,
            as_of_date.isoformat(),
            str(ticker),
            None if pd.isna(weight) else float(weight),
            retrieved_at,
        )
        for ticker, weight in members.items()
    ]
    database_path = initialize_price_database(database)
    with closing(sqlite3.connect(database_path)) as connection:
        connection.executemany(
            """
            INSERT INTO index_memberships (
                index_ticker,
                as_of_date,
                member_ticker,
                index_weight,
                retrieved_at
            )
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (index_ticker, as_of_date, member_ticker) DO UPDATE SET
                index_weight = excluded.index_weight,
                retrieved_at = excluded.retrieved_at
            """,
            rows,
        )
        connection.commit()


def replace_index_members(
    index_ticker: str,
    as_of_date: date,
    members: pd.Series,
    database: str | Path = BLOOMBERG_DATABASE,
) -> None:
    """Atomically replace one dated index-membership snapshot."""
    if members.empty:
        raise ValueError("A replacement membership snapshot cannot be empty.")
    if members.index.has_duplicates:
        raise ValueError("Membership identifiers must be unique.")

    retrieved_at = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )
    rows = [
        (
            index_ticker,
            as_of_date.isoformat(),
            str(identifier),
            None if pd.isna(weight) else float(weight),
            retrieved_at,
        )
        for identifier, weight in members.items()
    ]
    database_path = initialize_price_database(database)
    with closing(sqlite3.connect(database_path)) as connection:
        connection.execute(
            """
            DELETE FROM index_memberships
            WHERE index_ticker = ? AND as_of_date = ?
            """,
            [index_ticker, as_of_date.isoformat()],
        )
        connection.executemany(
            """
            INSERT INTO index_memberships (
                index_ticker,
                as_of_date,
                member_ticker,
                index_weight,
                retrieved_at
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            rows,
        )
        connection.commit()


def get_index_members(
    index_ticker: str,
    as_of_date: date,
    max_retries: int = DEFAULT_MAX_RETRIES,
    database: str | Path = BLOOMBERG_DATABASE,
    allow_download: bool = True,
) -> pd.Series:
    """Return a cached or downloaded point-in-time index membership."""
    if max_retries < 0:
        raise ValueError("max_retries cannot be negative.")

    members = load_index_members(index_ticker, as_of_date, database)
    if not members.empty:
        return members
    if not allow_download:
        raise LookupError(
            f"No {index_ticker} membership is cached for {as_of_date}. "
            "Run without --cache-only to download it first."
        )

    records: list[dict[str, Any]] = []
    for attempt in range(max_retries + 1):
        try:
            records = _index_members_request(index_ticker, as_of_date)
            break
        except (RuntimeError, TimeoutError):
            if attempt == max_retries:
                raise
            time.sleep(RETRY_BACKOFF_SECONDS * (2**attempt))

    members = _parse_index_members(records)
    if members.empty:
        raise RuntimeError(
            f"Bloomberg returned no {index_ticker} members for "
            f"{as_of_date} using {INDEX_MEMBERS_FIELD}."
        )
    upsert_index_members(index_ticker, as_of_date, members, database)
    return members


def _download_price_batch(
    tickers: list[str],
    start_date: date,
    end_date: date,
    max_retries: int,
) -> pd.DataFrame:
    """Download one Bloomberg historical-price batch."""
    records: list[dict[str, Any]] = []

    for attempt in range(max_retries + 1):
        try:
            records = _historical_data_request(
                tickers=tickers,
                fields=HISTORICAL_FIELDS,
                start_date=start_date,
                end_date=end_date,
            )
            break
        except (RuntimeError, TimeoutError):
            if attempt == max_retries:
                raise
            time.sleep(RETRY_BACKOFF_SECONDS * (2**attempt))

    history_records: list[dict[str, Any]] = []
    for record in records:
        close = _as_float(record.get("PX_LAST"))
        if math.isnan(close) or "date" not in record:
            continue
        total_return_index = _as_float(
            record.get("TOT_RETURN_INDEX_GROSS_DVDS")
        )
        history_records.append(
            {
                "date": pd.Timestamp(record["date"]).tz_localize(None),
                "ticker": record["ticker"],
                "close": close,
                "total_return_index": (
                    close
                    if math.isnan(total_return_index)
                    else total_return_index
                ),
                "volume": _as_float(record.get("PX_VOLUME")),
                "short_interest": _as_float(
                    record.get("SHORT_INT")
                ),
                "equity_float": _as_float(record.get("EQY_FLOAT")),
                "analyst_sentiment": _as_float(
                    record.get("EQY_REC_CONS")
                ),
                "historical_fields_version": HISTORICAL_FIELDS_VERSION,
            }
        )

    return pd.DataFrame.from_records(
        history_records,
        columns=[
            "date",
            "ticker",
            "close",
            "total_return_index",
            "volume",
            "short_interest",
            "equity_float",
            "analyst_sentiment",
            "historical_fields_version",
        ],
    )


def _upsert_price_history(
    history: pd.DataFrame,
    database: str | Path = PRICE_DATABASE,
) -> None:
    """Insert or update Bloomberg price observations."""
    if history.empty:
        return

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
                if pd.isna(row.total_return_index)
                else float(row.total_return_index)
            ),
            None if pd.isna(row.volume) else float(row.volume),
            (
                None
                if pd.isna(row.short_interest)
                else float(row.short_interest)
            ),
            (
                None
                if pd.isna(row.equity_float)
                else float(row.equity_float)
            ),
            (
                None
                if pd.isna(row.analyst_sentiment)
                else float(row.analyst_sentiment)
            ),
            int(row.historical_fields_version),
            retrieved_at,
        )
        for row in history.itertuples(index=False)
    ]

    database_path = initialize_price_database(database)
    with closing(sqlite3.connect(database_path)) as connection:
        connection.executemany(
            """
            INSERT INTO daily_prices (
                ticker,
                date,
                close,
                total_return_index,
                volume,
                short_interest,
                equity_float,
                analyst_sentiment,
                historical_fields_version,
                retrieved_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (ticker, date) DO UPDATE SET
                close = excluded.close,
                total_return_index = excluded.total_return_index,
                volume = excluded.volume,
                short_interest = excluded.short_interest,
                equity_float = excluded.equity_float,
                analyst_sentiment = excluded.analyst_sentiment,
                historical_fields_version = (
                    excluded.historical_fields_version
                ),
                retrieved_at = excluded.retrieved_at
            """,
            rows,
        )
        connection.commit()


def _cached_date_bounds(
    tickers: pd.Index,
    database: str | Path = PRICE_DATABASE,
) -> dict[str, tuple[date, date, int]]:
    """Return cached date bounds and field version by ticker."""
    database_path = initialize_price_database(database)
    bounds: dict[str, tuple[date, date, int]] = {}

    with closing(sqlite3.connect(database_path)) as connection:
        ticker_list = tickers.astype(str).unique().tolist()
        for start in range(0, len(ticker_list), 900):
            batch = ticker_list[start : start + 900]
            placeholders = ", ".join("?" for _ in batch)
            rows = connection.execute(
                f"""
                SELECT
                    ticker,
                    MIN(date),
                    MAX(date),
                    MAX(historical_fields_version)
                FROM daily_prices
                WHERE ticker IN ({placeholders})
                GROUP BY ticker
                """,
                batch,
            ).fetchall()
            for (
                ticker,
                minimum_date,
                maximum_date,
                fields_version,
            ) in rows:
                bounds[ticker] = (
                    date.fromisoformat(minimum_date),
                    date.fromisoformat(maximum_date),
                    int(fields_version),
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
    """Download only leading or trailing ranges absent from SQLite."""
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

        minimum_date, maximum_date, fields_version = bounds
        if fields_version < HISTORICAL_FIELDS_VERSION:
            requests_by_range.setdefault(
                (start_date, end_date),
                [],
            ).append(ticker)
            continue
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
            history = _download_price_batch(
                tickers=batch,
                start_date=request_start,
                end_date=request_end,
                max_retries=max_retries,
            )
            _upsert_price_history(history, database)


def get_price_history(
    tickers: pd.Index,
    start_date: date,
    end_date: date,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_retries: int = DEFAULT_MAX_RETRIES,
    database: str | Path = PRICE_DATABASE,
    allow_download: bool = True,
) -> pd.DataFrame:
    """Return an inclusive price range from the Bloomberg SQLite cache."""
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
            frames.append(
                pd.read_sql_query(
                    f"""
                    SELECT
                        date,
                        ticker,
                        close,
                        total_return_index,
                        volume,
                        short_interest,
                        equity_float,
                        analyst_sentiment
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
            )

    if not frames:
        return pd.DataFrame(
            columns=[
                "date",
                "ticker",
                "close",
                "total_return_index",
                "volume",
                "short_interest",
                "equity_float",
                "analyst_sentiment",
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
    """Return Bloomberg closing prices for one portfolio date."""
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
    prices = prices.reindex(unique_tickers).rename("price")
    missing_tickers = prices.index[prices.isna()]

    if len(missing_tickers) > 0:
        preview = ", ".join(missing_tickers[:10])
        if len(missing_tickers) > 10:
            preview = f"{preview}, ..."
        warnings.warn(
            "SQLite contains no Bloomberg closing price for "
            f"{len(missing_tickers)} ticker(s) on {as_of_date}: {preview}",
            RuntimeWarning,
            stacklevel=2,
        )

    return prices


def load_historical_market_caps(
    tickers: pd.Index,
    as_of_date: date,
    database: str | Path = BLOOMBERG_DATABASE,
) -> pd.Series:
    """Return cached Bloomberg market caps for one historical date."""
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
            frames.append(
                pd.read_sql_query(
                    f"""
                    SELECT ticker, market_cap
                    FROM historical_market_caps
                    WHERE ticker IN ({placeholders})
                        AND as_of_date = ?
                    """,
                    connection,
                    params=[*batch, as_of_date.isoformat()],
                )
            )

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
    return pd.to_numeric(
        market_caps.reindex(unique_tickers),
        errors="coerce",
    ).rename("market_cap")


def upsert_historical_market_caps(
    market_caps: pd.Series,
    as_of_date: date,
    database: str | Path = BLOOMBERG_DATABASE,
) -> None:
    """Store successful Bloomberg market-cap observations."""
    successful = pd.to_numeric(market_caps, errors="coerce").dropna()
    successful = successful.loc[successful.gt(0)]
    if successful.empty:
        return

    retrieved_at = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )
    rows = [
        (
            str(ticker),
            as_of_date.isoformat(),
            float(market_cap),
            retrieved_at,
        )
        for ticker, market_cap in successful.items()
    ]
    database_path = initialize_price_database(database)
    with closing(sqlite3.connect(database_path)) as connection:
        connection.executemany(
            """
            INSERT INTO historical_market_caps (
                ticker,
                as_of_date,
                market_cap,
                retrieved_at
            )
            VALUES (?, ?, ?, ?)
            ON CONFLICT (ticker, as_of_date) DO UPDATE SET
                market_cap = excluded.market_cap,
                retrieved_at = excluded.retrieved_at
            """,
            rows,
        )
        connection.commit()


def _download_market_cap_batch(
    tickers: list[str],
    as_of_date: date,
    max_retries: int,
) -> pd.Series:
    """Download historical Bloomberg market caps in absolute dollars."""
    records: list[dict[str, Any]] = []
    for attempt in range(max_retries + 1):
        try:
            records = _historical_data_request(
                tickers=tickers,
                fields=[MARKET_CAP_FIELD],
                start_date=as_of_date,
                end_date=as_of_date,
            )
            break
        except (RuntimeError, TimeoutError):
            if attempt == max_retries:
                raise
            time.sleep(RETRY_BACKOFF_SECONDS * (2**attempt))

    market_caps = pd.Series(
        index=pd.Index(tickers, name="ticker"),
        dtype="float64",
        name="market_cap",
    )
    for record in records:
        value_in_millions = _as_float(record.get(MARKET_CAP_FIELD))
        if not math.isnan(value_in_millions) and value_in_millions > 0:
            market_caps.loc[record["ticker"]] = (
                value_in_millions * MARKET_CAP_MULTIPLIER
            )
    return market_caps


def get_historical_market_caps(
    tickers: pd.Index,
    as_of_date: date,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_retries: int = DEFAULT_MAX_RETRIES,
    database: str | Path = BLOOMBERG_DATABASE,
    allow_download: bool = True,
) -> pd.Series:
    """Return cached or downloaded historical Bloomberg market caps."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if max_retries < 0:
        raise ValueError("max_retries cannot be negative.")

    unique_tickers = pd.Index(
        tickers.astype(str).unique(),
        name="ticker",
    )
    market_caps = load_historical_market_caps(
        tickers=unique_tickers,
        as_of_date=as_of_date,
        database=database,
    )
    missing_tickers = market_caps.index[market_caps.isna()]
    if allow_download:
        for start in range(0, len(missing_tickers), batch_size):
            batch = missing_tickers[start : start + batch_size].tolist()
            downloaded = _download_market_cap_batch(
                tickers=batch,
                as_of_date=as_of_date,
                max_retries=max_retries,
            )
            market_caps.update(downloaded)
            upsert_historical_market_caps(
                market_caps=downloaded,
                as_of_date=as_of_date,
                database=database,
            )

    missing_tickers = market_caps.index[market_caps.isna()]
    if len(missing_tickers) == len(market_caps):
        if not allow_download:
            raise RuntimeError(
                "No historical market caps are cached for "
                f"{as_of_date.isoformat()}. Run without --cache-only to "
                "download them first."
            )
        raise RuntimeError(
            "Bloomberg did not return historical market caps for any ticker."
        )
    if len(missing_tickers) > 0:
        preview = ", ".join(missing_tickers[:10])
        if len(missing_tickers) > 10:
            preview = f"{preview}, ..."
        source_message = (
            "The local cache does not contain historical market caps for "
            if not allow_download
            else "Bloomberg did not return historical market caps for "
        )
        warnings.warn(
            f"{source_message}{len(missing_tickers)} ticker(s): {preview}",
            RuntimeWarning,
            stacklevel=2,
        )
    return market_caps


def load_security_metadata(
    tickers: pd.Index,
    database: str | Path = BLOOMBERG_DATABASE,
) -> pd.DataFrame:
    """Return cached Bloomberg sector and price-to-book metadata."""
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
            frames.append(
                pd.read_sql_query(
                    f"""
                    SELECT ticker, sector, price_to_book, retrieved_at
                    FROM security_metadata
                    WHERE ticker IN ({placeholders})
                    """,
                    connection,
                    params=batch,
                )
            )

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
    database: str | Path = BLOOMBERG_DATABASE,
) -> None:
    """Store successful Bloomberg security metadata."""
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
            retrieved_at,
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


def _download_metadata_batch(
    tickers: list[str],
    max_retries: int,
) -> pd.DataFrame:
    """Download Bloomberg GICS sector and price-to-book metadata."""
    records: list[dict[str, Any]] = []
    for attempt in range(max_retries + 1):
        try:
            records = _reference_data_request(
                tickers=tickers,
                fields=METADATA_FIELDS,
            )
            break
        except (RuntimeError, TimeoutError):
            if attempt == max_retries:
                raise
            time.sleep(RETRY_BACKOFF_SECONDS * (2**attempt))

    retrieved_at = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )
    metadata_records = [
        {
            "ticker": record["ticker"],
            "sector": record.get("GICS_SECTOR_NAME"),
            "price_to_book": _as_float(record.get("PX_TO_BOOK_RATIO")),
            "retrieved_at": retrieved_at,
        }
        for record in records
    ]
    return pd.DataFrame.from_records(
        metadata_records,
        columns=["ticker", "sector", "price_to_book", "retrieved_at"],
    )


def get_security_metadata(
    tickers: pd.Index,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_retries: int = DEFAULT_MAX_RETRIES,
    database: str | Path = BLOOMBERG_DATABASE,
    allow_download: bool = True,
) -> pd.DataFrame:
    """Return cached or downloaded Bloomberg security metadata."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if max_retries < 0:
        raise ValueError("max_retries cannot be negative.")

    unique_tickers = pd.Index(
        tickers.astype(str).unique(),
        name="ticker",
    )
    metadata = load_security_metadata(unique_tickers, database)
    cached = metadata[["sector", "price_to_book"]].notna().all(axis=1)
    missing_tickers = metadata.index[~cached]

    if allow_download:
        for start in range(0, len(missing_tickers), batch_size):
            batch = missing_tickers[start : start + batch_size].tolist()
            downloaded = _download_metadata_batch(
                tickers=batch,
                max_retries=max_retries,
            )
            upsert_security_metadata(downloaded, database)

    return load_security_metadata(unique_tickers, database)


def _parse_iso_date(value: str) -> date:
    """Parse a YYYY-MM-DD command-line date."""
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Dates must use YYYY-MM-DD format."
        ) from error


def _resolve_cache_portfolios(
    csv_files: list[Path],
) -> list[Path]:
    """Resolve explicit portfolios or discover every portfolio in port_data."""
    from construct_port import (
        PORTFOLIO_DATA_DIR,
        PORTFOLIO_FILENAME_PATTERN,
        resolve_portfolio_path,
    )

    if csv_files:
        return [resolve_portfolio_path(csv_file) for csv_file in csv_files]

    portfolio_paths = sorted(
        portfolio_path
        for portfolio_path in PORTFOLIO_DATA_DIR.glob("*.csv")
        if PORTFOLIO_FILENAME_PATTERN.fullmatch(portfolio_path.name)
    )
    if not portfolio_paths:
        raise FileNotFoundError(
            "No full, long-only, or short-only dated portfolio CSVs "
            f"found in {PORTFOLIO_DATA_DIR}."
        )
    return portfolio_paths


def main() -> None:
    """Preload Bloomberg portfolio data into the SQLite cache."""
    parser = argparse.ArgumentParser(
        description=(
            "Download Bloomberg history, market caps, and metadata into "
            "the local SQLite cache before running portfolio reports."
        )
    )
    parser.add_argument(
        "portfolio_csv",
        nargs="*",
        type=Path,
        help=(
            "Portfolio CSVs to include. If omitted, every supported "
            "dated portfolio CSV in port_data is used."
        ),
    )
    parser.add_argument(
        "--start-date",
        type=_parse_iso_date,
        help=(
            "First Bloomberg history date in YYYY-MM-DD format. Defaults "
            "to enough calendar history for --lookback-days."
        ),
    )
    parser.add_argument(
        "--end-date",
        type=_parse_iso_date,
        help=(
            "Last Bloomberg history date in YYYY-MM-DD format. Defaults "
            "to the latest selected portfolio date."
        ),
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=DEFAULT_CACHE_LOOKBACK_DAYS,
        help=(
            "Trading-day history used to infer --start-date "
            "(default: 252)."
        ),
    )
    parser.add_argument(
        "--benchmark",
        default=DEFAULT_BENCHMARK,
        help="Benchmark ticker to cache (default: VTHR).",
    )
    parser.add_argument(
        "--factor-universe",
        default=DEFAULT_FACTOR_UNIVERSE,
        help=(
            "Bloomberg index whose members define the custom-factor "
            f"normalization universe (default: {DEFAULT_FACTOR_UNIVERSE})."
        ),
    )
    parser.add_argument(
        "--factor-universe-workbook",
        action="append",
        type=Path,
        default=[],
        help=(
            "Bloomberg PORT XLSX export containing As Of Date, Benchmark "
            "Name, Bmrk weight, and FIGI columns. Repeats are allowed. "
            "Each workbook replaces the cached universe for its date."
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
        "--database",
        type=Path,
        default=BLOOMBERG_DATABASE,
        help=(
            "SQLite output path (default: port_data/bloomberg_data.db)."
        ),
    )
    args = parser.parse_args()

    if args.lookback_days <= 1:
        parser.error("--lookback-days must be greater than one.")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive.")
    if args.max_retries < 0:
        parser.error("--max-retries cannot be negative.")

    imported_snapshots: list[FactorUniverseSnapshot] = []
    for workbook_path in args.factor_universe_workbook:
        try:
            imported_snapshots.append(
                import_factor_universe_workbook(
                    workbook_path=workbook_path,
                    index_ticker=args.factor_universe,
                    database=args.database,
                )
            )
        except (FileNotFoundError, ValueError) as error:
            parser.error(str(error))

    from construct_port import get_portfolio_date, load_shares

    portfolio_paths = _resolve_cache_portfolios(args.portfolio_csv)
    imported_dates = {
        snapshot.as_of_date for snapshot in imported_snapshots
    }
    if imported_dates and not args.portfolio_csv:
        portfolio_paths = [
            portfolio_path
            for portfolio_path in portfolio_paths
            if get_portfolio_date(portfolio_path) in imported_dates
        ]
        if not portfolio_paths:
            parser.error(
                "No portfolio CSV matches a factor-universe workbook date."
            )
    portfolios = [
        (
            portfolio_path,
            get_portfolio_date(portfolio_path),
            load_shares(portfolio_path),
        )
        for portfolio_path in portfolio_paths
    ]
    end_date = args.end_date or max(
        portfolio_date for _, portfolio_date, _ in portfolios
    )
    selected_portfolios = [
        portfolio
        for portfolio in portfolios
        if portfolio[1] <= end_date
    ]
    if not selected_portfolios:
        parser.error("No selected portfolio exists on or before --end-date.")

    first_portfolio_date = min(
        portfolio_date
        for _, portfolio_date, _ in selected_portfolios
    )
    calendar_days = math.ceil(
        args.lookback_days * HISTORY_CALENDAR_MULTIPLIER
    )
    start_date = args.start_date or (
        first_portfolio_date - timedelta(days=calendar_days)
    )
    if start_date > end_date:
        parser.error("--start-date cannot be after --end-date.")

    portfolio_tickers = pd.Index([], dtype="object", name="ticker")
    portfolio_tickers_by_date: dict[date, pd.Index] = {}
    for _, portfolio_date, shares in selected_portfolios:
        portfolio_tickers = portfolio_tickers.union(shares.index)
        existing_tickers = portfolio_tickers_by_date.get(
            portfolio_date,
            pd.Index([], dtype="object", name="ticker"),
        )
        portfolio_tickers_by_date[portfolio_date] = (
            existing_tickers.union(shares.index)
        )

    factor_tickers = portfolio_tickers.copy()
    factor_tickers_by_date: dict[date, pd.Index] = {}
    for portfolio_date, dated_portfolio_tickers in sorted(
        portfolio_tickers_by_date.items()
    ):
        universe_members = get_index_members(
            index_ticker=args.factor_universe,
            as_of_date=portfolio_date,
            max_retries=args.max_retries,
            database=args.database,
        )
        dated_factor_tickers = universe_members.index.union(
            dated_portfolio_tickers
        )
        factor_tickers_by_date[portfolio_date] = dated_factor_tickers
        factor_tickers = factor_tickers.union(dated_factor_tickers)

    history_tickers = factor_tickers.union(
        pd.Index([args.benchmark], name="ticker")
    )

    database_path = initialize_price_database(args.database)
    ensure_price_history(
        tickers=history_tickers,
        start_date=start_date,
        end_date=end_date,
        batch_size=args.batch_size,
        max_retries=args.max_retries,
        database=database_path,
    )
    for portfolio_date, dated_factor_tickers in sorted(
        factor_tickers_by_date.items()
    ):
        get_historical_market_caps(
            tickers=dated_factor_tickers,
            as_of_date=portfolio_date,
            batch_size=args.batch_size,
            max_retries=args.max_retries,
            database=database_path,
        )
    get_security_metadata(
        tickers=factor_tickers,
        batch_size=args.batch_size,
        max_retries=args.max_retries,
        database=database_path,
    )

    portfolio_dates = {
        portfolio_date for _, portfolio_date, _ in selected_portfolios
    }
    print("Bloomberg cache updated")
    print(f"Database       : {database_path.resolve()}")
    print(f"Portfolio files: {len(selected_portfolios):,}")
    print(f"Portfolio dates: {len(portfolio_dates):,}")
    print(f"Position tickers: {len(portfolio_tickers):,}")
    print(f"Factor universe: {args.factor_universe}")
    print(f"Factor identifiers: {len(factor_tickers):,}")
    for snapshot in imported_snapshots:
        print(
            "Imported universe: "
            f"{snapshot.as_of_date} ({len(snapshot.members):,} FIGIs)"
        )
    print(f"Benchmark      : {args.benchmark}")
    print(f"History range  : {start_date} through {end_date}")


if __name__ == "__main__":
    main()
