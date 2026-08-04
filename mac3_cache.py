"""Import Bloomberg MAC3 Risk Model Files and calculate factor risk."""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import sqlite3
import ssl
import urllib.request
import zipfile
from contextlib import closing
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from bloomberg_cache import BLOOMBERG_DATABASE


DEFAULT_HORIZON = "quarterly"
HTTP_TIMEOUT_SECONDS = 120


@dataclass(frozen=True)
class Mac3RiskReport:
    """Store portfolio risk calculated from Bloomberg MAC3 model files."""

    as_of_date: date
    model: str
    horizon: str
    capital: float
    systematic_variance: float
    systematic_volatility: float
    position_coverage: int
    total_positions: int
    covered_gross_weight: float
    missing_tickers: tuple[str, ...]
    factor_risk: pd.DataFrame


def initialize_mac3_database(
    database: str | Path = BLOOMBERG_DATABASE,
) -> Path:
    """Create MAC3 tables in the shared Bloomberg SQLite database."""
    database_path = Path(database).resolve()
    database_path.parent.mkdir(parents=True, exist_ok=True)

    with closing(sqlite3.connect(database_path)) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS mac3_factor_exposures (
                as_of_date TEXT NOT NULL,
                model TEXT NOT NULL,
                horizon TEXT NOT NULL,
                security_id TEXT NOT NULL,
                ticker TEXT NOT NULL,
                factor_id TEXT NOT NULL,
                factor_name TEXT NOT NULL,
                factor_type TEXT,
                exposure REAL NOT NULL,
                PRIMARY KEY (
                    as_of_date,
                    model,
                    horizon,
                    security_id,
                    factor_id
                )
            );

            CREATE INDEX IF NOT EXISTS idx_mac3_exposure_lookup
            ON mac3_factor_exposures (
                as_of_date,
                model,
                horizon,
                ticker
            );

            CREATE TABLE IF NOT EXISTS mac3_factor_covariance (
                as_of_date TEXT NOT NULL,
                model TEXT NOT NULL,
                horizon TEXT NOT NULL,
                row_factor_id TEXT NOT NULL,
                column_factor_id TEXT NOT NULL,
                covariance REAL NOT NULL,
                PRIMARY KEY (
                    as_of_date,
                    model,
                    horizon,
                    row_factor_id,
                    column_factor_id
                )
            );

            CREATE TABLE IF NOT EXISTS mac3_factor_returns (
                return_date TEXT NOT NULL,
                model TEXT NOT NULL,
                horizon TEXT NOT NULL,
                factor_id TEXT NOT NULL,
                factor_name TEXT NOT NULL,
                factor_return REAL NOT NULL,
                PRIMARY KEY (
                    return_date,
                    model,
                    horizon,
                    factor_id
                )
            );

            CREATE TABLE IF NOT EXISTS mac3_model_snapshots (
                as_of_date TEXT NOT NULL,
                model TEXT NOT NULL,
                horizon TEXT NOT NULL,
                covariance_scale REAL NOT NULL,
                factor_return_scale REAL NOT NULL,
                imported_at TEXT NOT NULL,
                PRIMARY KEY (as_of_date, model, horizon)
            );
            """
        )
        connection.commit()

    return database_path


def _canonical_column(value: object) -> str:
    """Return a comparison-safe column label."""
    return "_".join(str(value).strip().lower().split())


def _resolve_column(
    frame: pd.DataFrame,
    explicit: str | None,
    aliases: Iterable[str],
    label: str,
    required: bool = True,
) -> object | None:
    """Resolve an explicit or commonly named input column."""
    lookup = {_canonical_column(column): column for column in frame.columns}
    if explicit is not None:
        key = _canonical_column(explicit)
        if key not in lookup:
            raise ValueError(
                f"{label} column {explicit!r} is not present. Available "
                f"columns: {list(frame.columns)}"
            )
        return lookup[key]

    for alias in aliases:
        if _canonical_column(alias) in lookup:
            return lookup[_canonical_column(alias)]

    if required:
        raise ValueError(
            f"Could not identify the {label} column. Pass its name "
            "explicitly on the command line."
        )
    return None


def _response_headers(header_environment: list[str]) -> dict[str, str]:
    """Build HTTP headers from HEADER=ENVIRONMENT_VARIABLE entries."""
    headers = {
        "Accept": (
            "text/csv, application/json, application/zip, "
            "application/octet-stream"
        ),
        "User-Agent": "risk-framework-mac3/1.0",
    }
    for entry in header_environment:
        if "=" not in entry:
            raise ValueError(
                "API headers must use HEADER=ENVIRONMENT_VARIABLE format."
            )
        header, environment_name = entry.split("=", maxsplit=1)
        value = os.environ.get(environment_name)
        if value is None:
            raise RuntimeError(
                f"Environment variable {environment_name!r} is not set."
            )
        headers[header.strip()] = value
    return headers


def _read_tabular_payload(
    payload: bytes,
    source_name: str,
    zip_member: str | None = None,
) -> pd.DataFrame:
    """Read tabular JSON, CSV, or a ZIP containing a CSV model file."""
    if payload[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            members = [
                name
                for name in archive.namelist()
                if name.lower().endswith((".csv", ".txt"))
            ]
            selected = zip_member
            if selected is None:
                if len(members) != 1:
                    raise ValueError(
                        f"{source_name} contains {len(members)} CSV files. "
                        "Specify the desired ZIP member."
                    )
                selected = members[0]
            if selected not in archive.namelist():
                raise ValueError(
                    f"ZIP member {selected!r} is not present in "
                    f"{source_name}."
                )
            payload = archive.read(selected)

    stripped = payload.lstrip()
    if stripped.startswith((b"[", b"{")):
        try:
            decoded = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"{source_name} contains invalid JSON.") from error
        if isinstance(decoded, list):
            return pd.DataFrame.from_records(decoded)
        if isinstance(decoded, dict):
            for key in ("data", "results", "records"):
                if isinstance(decoded.get(key), list):
                    return pd.DataFrame.from_records(decoded[key])
            if decoded and all(
                isinstance(value, list) for value in decoded.values()
            ):
                return pd.DataFrame(decoded)
        raise ValueError(
            f"{source_name} JSON is not a record list or column mapping."
        )

    try:
        return pd.read_csv(io.BytesIO(payload))
    except (pd.errors.ParserError, UnicodeDecodeError) as error:
        raise ValueError(
            f"{source_name} is not a readable CSV or ZIP of CSV data."
        ) from error


def read_mac3_resource(
    source: str | Path,
    header_environment: list[str] | None = None,
    client_certificate: str | Path | None = None,
    client_key: str | Path | None = None,
    zip_member: str | None = None,
) -> pd.DataFrame:
    """Read one local or Bloomberg-issued HTTPS MAC3 resource URL."""
    source_text = str(source)
    if source_text.lower().startswith(("https://", "http://")):
        headers = _response_headers(header_environment or [])
        request = urllib.request.Request(source_text, headers=headers)
        context = ssl.create_default_context()
        if client_certificate is not None:
            context.load_cert_chain(
                certfile=str(client_certificate),
                keyfile=None if client_key is None else str(client_key),
            )
        with urllib.request.urlopen(
            request,
            timeout=HTTP_TIMEOUT_SECONDS,
            context=context,
        ) as response:
            payload = response.read()
        return _read_tabular_payload(payload, source_text, zip_member)

    source_path = Path(source).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"MAC3 resource does not exist: {source_path}")
    return _read_tabular_payload(
        source_path.read_bytes(),
        str(source_path),
        zip_member,
    )


def _infer_ticker(security_id: object) -> str:
    """Infer a portfolio ticker from a Bloomberg security description."""
    value = str(security_id).strip()
    suffixes = (" US Equity", " Equity")
    for suffix in suffixes:
        if value.endswith(suffix):
            return value[: -len(suffix)].strip()
    return value


def normalize_factor_exposures(
    frame: pd.DataFrame,
    security_column: str | None = None,
    ticker_column: str | None = None,
    factor_column: str | None = None,
    exposure_column: str | None = None,
    factor_name_column: str | None = None,
    factor_type_column: str | None = None,
    exclude_columns: Iterable[str] = (),
) -> pd.DataFrame:
    """Normalize long- or wide-form factor exposures."""
    security = _resolve_column(
        frame,
        security_column,
        ("security_id", "security", "identifier", "id_bb_global", "figi"),
        "security identifier",
    )
    ticker = _resolve_column(
        frame,
        ticker_column,
        ("ticker",),
        "ticker",
        required=False,
    )
    factor = _resolve_column(
        frame,
        factor_column,
        ("factor_id", "factor"),
        "factor identifier",
        required=False,
    )
    exposure = _resolve_column(
        frame,
        exposure_column,
        ("exposure", "factor_exposure"),
        "factor exposure",
        required=False,
    )

    if (factor is None) != (exposure is None):
        raise ValueError(
            "Long-form exposures require both factor and exposure columns."
        )

    if factor is not None and exposure is not None:
        factor_name = _resolve_column(
            frame,
            factor_name_column,
            ("factor_name", "name"),
            "factor name",
            required=False,
        )
        factor_type = _resolve_column(
            frame,
            factor_type_column,
            ("factor_type", "type", "category"),
            "factor type",
            required=False,
        )
        normalized = pd.DataFrame(
            {
                "security_id": frame[security].astype(str).str.strip(),
                "factor_id": frame[factor].astype(str).str.strip(),
                "factor_name": (
                    frame[factor_name].astype(str).str.strip()
                    if factor_name is not None
                    else frame[factor].astype(str).str.strip()
                ),
                "factor_type": (
                    frame[factor_type].astype("string").str.strip()
                    if factor_type is not None
                    else pd.Series(pd.NA, index=frame.index, dtype="string")
                ),
                "exposure": pd.to_numeric(frame[exposure], errors="coerce"),
            }
        )
        ticker_values = (
            frame[ticker].astype(str).str.strip()
            if ticker is not None
            else normalized["security_id"].map(_infer_ticker)
        )
        normalized.insert(1, "ticker", ticker_values)
    else:
        excluded = {
            _canonical_column(value)
            for value in (
                *exclude_columns,
                security,
                ticker,
            )
            if value is not None
        }
        factor_columns = [
            column
            for column in frame.columns
            if _canonical_column(column) not in excluded
            and pd.to_numeric(frame[column], errors="coerce").notna().any()
        ]
        if not factor_columns:
            raise ValueError("The wide exposure file has no numeric factor columns.")
        identifiers = pd.DataFrame(
            {
                "security_id": frame[security].astype(str).str.strip(),
                "ticker": (
                    frame[ticker].astype(str).str.strip()
                    if ticker is not None
                    else frame[security].map(_infer_ticker)
                ),
            }
        )
        wide = pd.concat([identifiers, frame[factor_columns]], axis="columns")
        normalized = wide.melt(
            id_vars=["security_id", "ticker"],
            var_name="factor_id",
            value_name="exposure",
        )
        normalized["factor_name"] = normalized["factor_id"].astype(str)
        normalized["factor_type"] = pd.NA
        normalized["exposure"] = pd.to_numeric(
            normalized["exposure"], errors="coerce"
        )

    normalized = normalized.dropna(subset=["exposure"])
    normalized = normalized[
        [
            "security_id",
            "ticker",
            "factor_id",
            "factor_name",
            "factor_type",
            "exposure",
        ]
    ]
    if normalized.empty:
        raise ValueError("The exposure resource contains no usable values.")
    return normalized.drop_duplicates(
        ["security_id", "factor_id"], keep="last"
    )


def normalize_factor_covariance(
    frame: pd.DataFrame,
    row_factor_column: str | None = None,
    column_factor_column: str | None = None,
    covariance_column: str | None = None,
) -> pd.DataFrame:
    """Normalize long- or square-matrix factor covariance data."""
    column_factor = _resolve_column(
        frame,
        column_factor_column,
        ("column_factor_id", "factor_id_2", "column_factor"),
        "column factor",
        required=False,
    )
    covariance = _resolve_column(
        frame,
        covariance_column,
        ("covariance", "value"),
        "covariance",
        required=False,
    )

    if (column_factor is None) != (covariance is None):
        raise ValueError(
            "Long-form covariance requires column-factor and covariance columns."
        )

    if column_factor is not None and covariance is not None:
        row_factor = _resolve_column(
            frame,
            row_factor_column,
            ("row_factor_id", "factor_id_1", "row_factor"),
            "row factor",
        )
        normalized = pd.DataFrame(
            {
                "row_factor_id": frame[row_factor].astype(str).str.strip(),
                "column_factor_id": frame[column_factor]
                .astype(str)
                .str.strip(),
                "covariance": pd.to_numeric(
                    frame[covariance], errors="coerce"
                ),
            }
        )
    else:
        row_factor = _resolve_column(
            frame,
            row_factor_column,
            (
                "row_factor_id",
                "factor_id",
                "factor",
                "unnamed:_0",
            ),
            "row factor",
            required=False,
        )
        if row_factor is None:
            row_factor = frame.columns[0]
        value_columns = [
            column
            for column in frame.columns
            if column != row_factor
            and pd.to_numeric(frame[column], errors="coerce").notna().any()
        ]
        wide = frame[[row_factor, *value_columns]].rename(
            columns={row_factor: "row_factor_id"}
        )
        normalized = wide.melt(
            id_vars="row_factor_id",
            var_name="column_factor_id",
            value_name="covariance",
        )
        normalized["covariance"] = pd.to_numeric(
            normalized["covariance"], errors="coerce"
        )

    normalized = normalized.dropna(subset=["covariance"])
    if normalized.empty:
        raise ValueError("The covariance resource contains no usable values.")
    return normalized.drop_duplicates(
        ["row_factor_id", "column_factor_id"], keep="last"
    )


def normalize_factor_returns(
    frame: pd.DataFrame,
    date_column: str | None = None,
    factor_column: str | None = None,
    return_column: str | None = None,
    factor_name_column: str | None = None,
) -> pd.DataFrame:
    """Normalize optional long- or wide-form factor returns."""
    return_date = _resolve_column(
        frame,
        date_column,
        ("return_date", "date", "as_of_date"),
        "factor-return date",
    )
    factor = _resolve_column(
        frame,
        factor_column,
        ("factor_id", "factor"),
        "factor identifier",
        required=False,
    )
    factor_return = _resolve_column(
        frame,
        return_column,
        ("factor_return", "return"),
        "factor return",
        required=False,
    )
    if (factor is None) != (factor_return is None):
        raise ValueError(
            "Long-form factor returns require factor and return columns."
        )

    if factor is not None and factor_return is not None:
        factor_name = _resolve_column(
            frame,
            factor_name_column,
            ("factor_name", "name"),
            "factor name",
            required=False,
        )
        normalized = pd.DataFrame(
            {
                "return_date": pd.to_datetime(frame[return_date]).dt.date,
                "factor_id": frame[factor].astype(str).str.strip(),
                "factor_name": (
                    frame[factor_name].astype(str).str.strip()
                    if factor_name is not None
                    else frame[factor].astype(str).str.strip()
                ),
                "factor_return": pd.to_numeric(
                    frame[factor_return], errors="coerce"
                ),
            }
        )
    else:
        factor_columns = [
            column
            for column in frame.columns
            if column != return_date
            and pd.to_numeric(frame[column], errors="coerce").notna().any()
        ]
        normalized = frame[[return_date, *factor_columns]].melt(
            id_vars=return_date,
            var_name="factor_id",
            value_name="factor_return",
        )
        normalized = normalized.rename(columns={return_date: "return_date"})
        normalized["return_date"] = pd.to_datetime(
            normalized["return_date"]
        ).dt.date
        normalized["factor_name"] = normalized["factor_id"].astype(str)
        normalized["factor_return"] = pd.to_numeric(
            normalized["factor_return"], errors="coerce"
        )

    return normalized.dropna(subset=["return_date", "factor_return"])[
        ["return_date", "factor_id", "factor_name", "factor_return"]
    ].drop_duplicates(["return_date", "factor_id"], keep="last")


def store_mac3_model_data(
    as_of_date: date,
    model: str,
    horizon: str,
    exposures: pd.DataFrame,
    covariance: pd.DataFrame,
    factor_returns: pd.DataFrame | None = None,
    database: str | Path = BLOOMBERG_DATABASE,
    covariance_scale: float = 1.0,
    factor_return_scale: float = 1.0,
) -> None:
    """Atomically replace one dated MAC3 model snapshot in SQLite.

    Scale factors convert the units documented in the Bloomberg delivery
    contract to decimal return and decimal-return-squared units.
    """
    if not math.isfinite(covariance_scale) or covariance_scale <= 0:
        raise ValueError("covariance_scale must be finite and positive.")
    if not math.isfinite(factor_return_scale) or factor_return_scale <= 0:
        raise ValueError("factor_return_scale must be finite and positive.")

    database_path = initialize_mac3_database(database)
    key = (as_of_date.isoformat(), model, horizon)
    exposure_rows = [
        (
            *key,
            str(row.security_id),
            str(row.ticker),
            str(row.factor_id),
            str(row.factor_name),
            None if pd.isna(row.factor_type) else str(row.factor_type),
            float(row.exposure),
        )
        for row in exposures.itertuples(index=False)
    ]
    covariance_rows = [
        (
            *key,
            str(row.row_factor_id),
            str(row.column_factor_id),
            float(row.covariance) * covariance_scale,
        )
        for row in covariance.itertuples(index=False)
    ]

    with closing(sqlite3.connect(database_path)) as connection:
        with connection:
            connection.execute(
                """
                DELETE FROM mac3_factor_exposures
                WHERE as_of_date = ? AND model = ? AND horizon = ?
                """,
                key,
            )
            connection.execute(
                """
                DELETE FROM mac3_factor_covariance
                WHERE as_of_date = ? AND model = ? AND horizon = ?
                """,
                key,
            )
            connection.executemany(
                """
                INSERT INTO mac3_factor_exposures (
                    as_of_date, model, horizon, security_id, ticker,
                    factor_id, factor_name, factor_type, exposure
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                exposure_rows,
            )
            connection.executemany(
                """
                INSERT INTO mac3_factor_covariance (
                    as_of_date, model, horizon, row_factor_id,
                    column_factor_id, covariance
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                covariance_rows,
            )
            connection.execute(
                """
                INSERT INTO mac3_model_snapshots (
                    as_of_date, model, horizon, covariance_scale,
                    factor_return_scale, imported_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT (as_of_date, model, horizon) DO UPDATE SET
                    covariance_scale = excluded.covariance_scale,
                    factor_return_scale = excluded.factor_return_scale,
                    imported_at = excluded.imported_at
                """,
                (
                    *key,
                    covariance_scale,
                    factor_return_scale,
                    datetime.now().astimezone().isoformat(
                        timespec="seconds"
                    ),
                ),
            )
            if factor_returns is not None and not factor_returns.empty:
                return_dates = sorted(
                    {
                        value.isoformat()
                        for value in factor_returns["return_date"]
                    }
                )
                placeholders = ", ".join("?" for _ in return_dates)
                connection.execute(
                    f"""
                    DELETE FROM mac3_factor_returns
                    WHERE model = ? AND horizon = ?
                    AND return_date IN ({placeholders})
                    """,
                    (model, horizon, *return_dates),
                )
                connection.executemany(
                    """
                    INSERT INTO mac3_factor_returns (
                        return_date, model, horizon, factor_id,
                        factor_name, factor_return
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            row.return_date.isoformat(),
                            model,
                            horizon,
                            str(row.factor_id),
                            str(row.factor_name),
                            float(row.factor_return) * factor_return_scale,
                        )
                        for row in factor_returns.itertuples(index=False)
                    ],
                )


def load_mac3_model_data(
    as_of_date: date,
    model: str,
    horizon: str,
    database: str | Path = BLOOMBERG_DATABASE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load one exposure and factor-covariance snapshot from SQLite."""
    database_path = initialize_mac3_database(database)
    params = (as_of_date.isoformat(), model, horizon)
    with closing(sqlite3.connect(database_path)) as connection:
        exposures = pd.read_sql_query(
            """
            SELECT security_id, ticker, factor_id, factor_name,
                   factor_type, exposure
            FROM mac3_factor_exposures
            WHERE as_of_date = ? AND model = ? AND horizon = ?
            """,
            connection,
            params=params,
        )
        covariance = pd.read_sql_query(
            """
            SELECT row_factor_id, column_factor_id, covariance
            FROM mac3_factor_covariance
            WHERE as_of_date = ? AND model = ? AND horizon = ?
            """,
            connection,
            params=params,
        )
    return exposures, covariance


def calculate_mac3_factor_risk(
    weights: pd.Series,
    capital: float,
    as_of_date: date,
    model: str,
    horizon: str = DEFAULT_HORIZON,
    database: str | Path = BLOOMBERG_DATABASE,
) -> Mac3RiskReport:
    """Calculate exact portfolio factor risk from delivered MAC3 matrices.

    The factor exposure vector is ``b = X.T @ w``. Systematic variance is
    ``b.T @ F @ b``, where ``F`` is Bloomberg's horizon-specific VCV matrix.
    No locally estimated proxy factor or covariance matrix is used.
    """
    exposures, covariance_long = load_mac3_model_data(
        as_of_date=as_of_date,
        model=model,
        horizon=horizon,
        database=database,
    )
    if exposures.empty or covariance_long.empty:
        raise LookupError(
            "No MAC3 exposure/VCV snapshot is cached for "
            f"{as_of_date.isoformat()}, model={model!r}, "
            f"horizon={horizon!r}. Run mac3_cache.py first."
        )

    normalized_weights = pd.to_numeric(weights, errors="coerce").dropna()
    normalized_weights.index = normalized_weights.index.astype(str)
    factor_metadata = exposures[
        ["factor_id", "factor_name", "factor_type"]
    ].drop_duplicates("factor_id", keep="last").set_index("factor_id")
    exposure_matrix = exposures.pivot_table(
        index="ticker",
        columns="factor_id",
        values="exposure",
        aggfunc="last",
    )
    covered_tickers = normalized_weights.index.intersection(
        exposure_matrix.index
    )
    missing_tickers = tuple(
        normalized_weights.index.difference(exposure_matrix.index).tolist()
    )
    if covered_tickers.empty:
        raise ValueError(
            "None of the portfolio tickers match the cached MAC3 exposure "
            "identifiers. Supply a ticker column or security mapping when "
            "importing the exposure file."
        )

    aligned_exposures = exposure_matrix.reindex(covered_tickers).fillna(0.0)
    aligned_weights = normalized_weights.reindex(covered_tickers).fillna(0.0)
    portfolio_factor_exposure = aligned_exposures.T.dot(aligned_weights)

    covariance = covariance_long.pivot_table(
        index="row_factor_id",
        columns="column_factor_id",
        values="covariance",
        aggfunc="last",
    )
    factor_ids = portfolio_factor_exposure.index.intersection(
        covariance.index
    ).intersection(covariance.columns)
    if factor_ids.empty:
        raise ValueError(
            "The MAC3 factor IDs in the exposure and VCV resources do not "
            "overlap. Check the model, horizon, and file column mapping."
        )
    covariance = covariance.reindex(
        index=factor_ids, columns=factor_ids
    ).astype("float64")
    covariance = covariance.combine_first(covariance.T)
    if covariance.isna().any().any():
        missing_cells = int(covariance.isna().sum().sum())
        raise ValueError(
            f"The MAC3 VCV matrix is incomplete ({missing_cells} missing "
            "cells after applying symmetry)."
        )
    covariance = (covariance + covariance.T) / 2.0

    factor_exposure = portfolio_factor_exposure.reindex(factor_ids)
    exposure_vector = factor_exposure.to_numpy(dtype="float64")
    covariance_values = covariance.to_numpy(dtype="float64")
    marginal_variance = covariance_values @ exposure_vector
    variance_contribution = exposure_vector * marginal_variance
    systematic_variance = float(variance_contribution.sum())
    if systematic_variance < -1e-12:
        raise ValueError(
            "The delivered MAC3 covariance matrix produces negative "
            "portfolio variance. Verify its units and column mapping."
        )
    systematic_variance = max(systematic_variance, 0.0)
    systematic_volatility = math.sqrt(systematic_variance)
    component_volatility = (
        variance_contribution / systematic_volatility
        if systematic_volatility > 0
        else np.zeros_like(variance_contribution)
    )
    percentage_contribution = (
        variance_contribution / systematic_variance
        if systematic_variance > 0
        else np.zeros_like(variance_contribution)
    )
    factor_risk = pd.DataFrame(
        {
            "factor_name": factor_metadata["factor_name"].reindex(factor_ids),
            "factor_type": factor_metadata["factor_type"].reindex(factor_ids),
            "exposure": factor_exposure,
            "marginal_variance": marginal_variance,
            "variance_contribution": variance_contribution,
            "component_volatility": component_volatility,
            "percentage_contribution": percentage_contribution,
        },
        index=pd.Index(factor_ids, name="factor_id"),
    )
    factor_risk["absolute_variance_contribution"] = factor_risk[
        "variance_contribution"
    ].abs()
    factor_risk = factor_risk.sort_values(
        "absolute_variance_contribution", ascending=False
    ).drop(columns="absolute_variance_contribution")

    return Mac3RiskReport(
        as_of_date=as_of_date,
        model=model,
        horizon=horizon,
        capital=float(capital),
        systematic_variance=systematic_variance,
        systematic_volatility=systematic_volatility,
        position_coverage=len(covered_tickers),
        total_positions=len(normalized_weights),
        covered_gross_weight=float(aligned_weights.abs().sum()),
        missing_tickers=missing_tickers,
        factor_risk=factor_risk,
    )


def format_mac3_risk_report(report: Mac3RiskReport) -> str:
    """Return every delivered MAC3 factor exposure and risk contribution."""
    lines = [
        "BLOOMBERG MAC3 FACTOR RISK",
        "=" * 104,
        f"As of                    : {report.as_of_date.isoformat()}",
        f"Model                    : {report.model}",
        f"Horizon                  : {report.horizon}",
        (
            "Position coverage        : "
            f"{report.position_coverage:,}/{report.total_positions:,}"
        ),
        (
            "Covered gross weight      : "
            f"{report.covered_gross_weight:.2%}"
        ),
        (
            "Systematic volatility     : "
            f"{report.systematic_volatility:.2%} (MAC3 horizon units)"
        ),
        "",
        "ALL FACTOR RISKS",
        "-" * 104,
        report.factor_risk.to_string(
            formatters={
                "exposure": lambda value: f"{value:+.5f}",
                "marginal_variance": lambda value: f"{value:+.8f}",
                "variance_contribution": lambda value: f"{value:+.8f}",
                "component_volatility": lambda value: f"{value:+.4%}",
                "percentage_contribution": lambda value: f"{value:+.2%}",
            }
        ),
    ]
    if report.missing_tickers:
        lines.extend(
            [
                "",
                "Unmatched portfolio tickers: "
                + ", ".join(report.missing_tickers),
            ]
        )
    return "\n".join(lines)


def write_mac3_factor_risk(
    report: Mac3RiskReport,
    output_directory: str | Path,
) -> Path:
    """Write all MAC3 factor exposures and risk contributions to CSV."""
    output_path = Path(output_directory).resolve() / (
        f"mac3_factor_risk_{report.as_of_date:%Y%m%d}.csv"
    )
    report.factor_risk.to_csv(output_path, index=True)
    return output_path


def _parse_date(value: str) -> date:
    """Parse an ISO command-line date."""
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Dates must use YYYY-MM-DD format."
        ) from error


def _csv_list(value: str) -> list[str]:
    """Parse a comma-separated command-line value."""
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    """Download/import Bloomberg-issued MAC3 resources into SQLite."""
    parser = argparse.ArgumentParser(
        description=(
            "Import Bloomberg MAC3 Risk Model Files from local files or "
            "Bloomberg-issued HTTPS resource URLs."
        )
    )
    parser.add_argument("--as-of-date", type=_parse_date, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--horizon", default=DEFAULT_HORIZON)
    parser.add_argument("--exposures-source", required=True)
    parser.add_argument("--covariance-source", required=True)
    parser.add_argument("--factor-returns-source")
    parser.add_argument(
        "--api-header-env",
        action="append",
        default=[],
        metavar="HEADER=ENV_VAR",
        help=(
            "HTTP header populated from an environment variable; repeat "
            "for every header required by Bloomberg's entitlement contract."
        ),
    )
    parser.add_argument("--client-certificate", type=Path)
    parser.add_argument("--client-key", type=Path)
    parser.add_argument("--exposures-zip-member")
    parser.add_argument("--covariance-zip-member")
    parser.add_argument("--factor-returns-zip-member")
    parser.add_argument("--exposure-security-column")
    parser.add_argument("--exposure-ticker-column")
    parser.add_argument("--exposure-factor-column")
    parser.add_argument("--exposure-value-column")
    parser.add_argument("--exposure-factor-name-column")
    parser.add_argument("--exposure-factor-type-column")
    parser.add_argument(
        "--exposure-exclude-columns",
        type=_csv_list,
        default=[],
    )
    parser.add_argument("--covariance-row-factor-column")
    parser.add_argument("--covariance-column-factor-column")
    parser.add_argument("--covariance-value-column")
    parser.add_argument("--return-date-column")
    parser.add_argument("--return-factor-column")
    parser.add_argument("--return-value-column")
    parser.add_argument("--return-factor-name-column")
    parser.add_argument(
        "--covariance-scale",
        type=float,
        default=1.0,
        help=(
            "Multiplier converting delivered VCV cells to decimal-return "
            "squared units (default: 1)."
        ),
    )
    parser.add_argument(
        "--factor-return-scale",
        type=float,
        default=1.0,
        help=(
            "Multiplier converting delivered factor returns to decimal "
            "returns (default: 1)."
        ),
    )
    parser.add_argument("--database", type=Path, default=BLOOMBERG_DATABASE)
    args = parser.parse_args()

    resource_options = {
        "header_environment": args.api_header_env,
        "client_certificate": args.client_certificate,
        "client_key": args.client_key,
    }
    exposure_frame = read_mac3_resource(
        args.exposures_source,
        zip_member=args.exposures_zip_member,
        **resource_options,
    )
    covariance_frame = read_mac3_resource(
        args.covariance_source,
        zip_member=args.covariance_zip_member,
        **resource_options,
    )
    exposures = normalize_factor_exposures(
        exposure_frame,
        security_column=args.exposure_security_column,
        ticker_column=args.exposure_ticker_column,
        factor_column=args.exposure_factor_column,
        exposure_column=args.exposure_value_column,
        factor_name_column=args.exposure_factor_name_column,
        factor_type_column=args.exposure_factor_type_column,
        exclude_columns=args.exposure_exclude_columns,
    )
    covariance = normalize_factor_covariance(
        covariance_frame,
        row_factor_column=args.covariance_row_factor_column,
        column_factor_column=args.covariance_column_factor_column,
        covariance_column=args.covariance_value_column,
    )
    factor_returns = None
    if args.factor_returns_source:
        return_frame = read_mac3_resource(
            args.factor_returns_source,
            zip_member=args.factor_returns_zip_member,
            **resource_options,
        )
        factor_returns = normalize_factor_returns(
            return_frame,
            date_column=args.return_date_column,
            factor_column=args.return_factor_column,
            return_column=args.return_value_column,
            factor_name_column=args.return_factor_name_column,
        )

    store_mac3_model_data(
        as_of_date=args.as_of_date,
        model=args.model,
        horizon=args.horizon,
        exposures=exposures,
        covariance=covariance,
        factor_returns=factor_returns,
        database=args.database,
        covariance_scale=args.covariance_scale,
        factor_return_scale=args.factor_return_scale,
    )
    print(
        f"Stored {len(exposures):,} exposures and "
        f"{len(covariance):,} VCV cells in {args.database.resolve()}"
    )
    if factor_returns is not None:
        print(f"Stored {len(factor_returns):,} factor-return observations.")


if __name__ == "__main__":
    main()
