from datetime import date
from pathlib import Path

import pandas as pd

from mac3_cache import (
    calculate_mac3_factor_risk,
    normalize_factor_covariance,
    normalize_factor_exposures,
    normalize_factor_returns,
    store_mac3_model_data,
)


database = Path("tmp/mac3_validation.db")
exposures = normalize_factor_exposures(
    pd.DataFrame(
        {
            "Security": ["AAPL US Equity", "MSFT US Equity"],
            "Ticker": ["AAPL", "MSFT"],
            "Market": [1.0, 1.0],
            "Size": [0.5, -1.0],
        }
    )
)
covariance = normalize_factor_covariance(
    pd.DataFrame(
        {
            "factor": ["Market", "Size"],
            "Market": [0.04, 0.01],
            "Size": [0.01, 0.09],
        }
    )
)
returns = normalize_factor_returns(
    pd.DataFrame(
        {
            "date": ["2026-01-30"],
            "Market": [0.01],
            "Size": [-0.002],
        }
    )
)
store_mac3_model_data(
    date(2026, 1, 30),
    "TEST",
    "quarterly",
    exposures,
    covariance,
    returns,
    database,
)
report = calculate_mac3_factor_risk(
    pd.Series({"AAPL": 0.6, "MSFT": -0.4}),
    100.0,
    date(2026, 1, 30),
    "TEST",
    "quarterly",
    database,
)
assert report.position_coverage == 2
assert abs(
    report.factor_risk["variance_contribution"].sum()
    - report.systematic_variance
) < 1e-12
assert abs(
    report.factor_risk["percentage_contribution"].sum() - 1.0
) < 1e-12
print(report.systematic_volatility)
print(
    report.factor_risk[
        [
            "exposure",
            "variance_contribution",
            "percentage_contribution",
        ]
    ]
)
