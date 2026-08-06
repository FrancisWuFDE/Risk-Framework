# Risk Framework

This project reads dated long/short equity portfolios, caches Bloomberg data
in SQLite, and produces a daily portfolio summary and market-risk report.

The normal workflow is:

```text
Portfolio CSV files
        |
        v
bloomberg_cache.py --------> port_data/bloomberg_data.db
mac3_cache.py -------------->          |
        |                              v
        +---------------------> port_summary.py
                                  |
                    +-------------+-------------+
                    |                           |
                    v                           v
          Printed summary/report       Dated output CSVs
```

All market data used by the current branch comes from the Bloomberg Desktop
API. The old `yfinance_data.db` file is not used by these scripts.

## Project files

| File | Purpose |
| --- | --- |
| `bloomberg_cache.py` | Downloads Bloomberg data and stores it in SQLite. It can also be run as a standalone cache preloader. |
| `mac3_cache.py` | Imports entitled Bloomberg MAC3 Risk Model Files from local files or API resource URLs, and calculates exact factor-model risk from the delivered exposures and VCV matrix. |
| `construct_port.py` | Validates a portfolio CSV and returns shares and dated Bloomberg closing prices as pandas Series. |
| `port_summary.py` | Produces the daily portfolio summary, preloads required data, and prints the risk report. |
| `risk_metrics.py` | Calculates exposures, normalized factor scores, beta, covariance risk, VaR, and stock-level risk contributions. |
| `price_cache.py` | Compatibility module that re-exports the Bloomberg cache functions. |
| `port_data/` | Holds input portfolios, the SQLite database, and generated output CSVs. |

## Requirements

- Windows with Bloomberg Terminal installed.
- An active Bloomberg Terminal login on the same computer running Python.
- Python with `pandas`, `numpy`, and Bloomberg's `blpapi` package.
- Bloomberg Desktop API permission for the requested securities and fields.

Create and activate a virtual environment in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install pandas numpy
python -m pip install --index-url=https://blpapi.bloomberg.com/repository/releases/python/simple/ blpapi
```

Bloomberg publishes the supported package command and API documentation on its
[API Library page](https://professional.bloomberg.com/support/api-library/).

If PowerShell prevents environment activation for the current session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

The Bloomberg connection uses the Desktop API in DAPI mode at
`localhost:8194`. Keep Bloomberg Terminal open and logged in while downloading
data.

## Portfolio input format

Portfolio files normally belong in `port_data/` and must use one of these
exact naming conventions:

```text
US_live_port_YYYYMMDD.csv
long_positions_YYYYMMDD.csv
short_positions_YYYYMMDD.csv
```

For example, these files represent the three supported portfolio types for the
same date:

```text
US_live_port_20260130.csv
long_positions_20260130.csv
short_positions_20260130.csv
```

The date must be a real calendar date. Each CSV requires unique `ticker` and
`shares` columns. Column names are stripped and converted to lowercase when
loaded.

```csv
ticker,shares
AAPL,1250
MSFT,800
NVDA,-500
```

- Positive shares are long positions.
- Negative shares are short positions.
- Blank or duplicate tickers are rejected.
- Tickers without a Bloomberg yellow key are converted to
  `<ticker> US Equity`.
- An identifier that already ends in a recognized yellow key, such as
  `SPX Index`, is sent to Bloomberg unchanged.

`port_summary.py` also requires the same portfolio type for the immediately
preceding NYSE trading session. For example, running
`long_positions_20260202.csv` requires `long_positions_20260130.csv`; it never
uses the full or short-only file as the prior portfolio.

## Download Bloomberg data first

The recommended workflow is to preload the database before running the
portfolio report. This separates Bloomberg download problems from calculation
problems and makes subsequent report runs much faster.

### Cache every portfolio in `port_data`

With Bloomberg Terminal open and the virtual environment active, run:

```powershell
python .\bloomberg_cache.py
```

When no portfolio arguments are supplied, the script discovers every valid
full, long-only, and short-only dated portfolio CSV in `port_data`. It then:

1. Loads the dated B3000 membership from SQLite, an imported PORT workbook,
   or Bloomberg's index-membership field.
2. Unions the B3000 members with all position tickers.
3. Adds the VTHR benchmark.
4. Downloads the required historical data in sequential batches of 50.
5. Downloads historical market caps for every selected portfolio date.
6. Downloads GICS sector and price-to-book metadata.
7. Stores the results in `port_data/bloomberg_data.db`.

The first B3000-backed preload is materially larger than a portfolio-only
download because factor descriptors must be available across the reference
universe. Later runs reuse the SQLite history and download only new ranges.

### Import the B3000 universe from a PORT workbook

Use a Bloomberg PORT export when `INDX_MWEIGHT_HIST` returns `#N/A Review` or
`WORKFLOW_REVIEW_NEEDED`. The workbook must contain:

- `As Of Date` and `Benchmark Name` metadata
- a `Bmrk` weight column
- a `FIGI` column

Only securities with a positive benchmark weight are imported. Portfolio-only
rows in a combined portfolio-plus-benchmark export are excluded from the
normalization universe. The imported snapshot replaces any previously cached
membership for the same universe and date.

To import the workbook and download all required history, market caps, and
metadata for its FIGIs, run:

```powershell
python .\bloomberg_cache.py `
    .\port_data\US_live_port_20260129.csv `
    --factor-universe-workbook `
    "C:\path\to\port_weights_20260129.xlsx" `
    --factor-universe "B3000 Index" `
    --batch-size 50
```

The downloader sends each FIGI to Bloomberg as `/bbgid/<FIGI>`. Portfolio
tickers are still downloaded under their normal Bloomberg equity identifiers,
so positions outside B3000 are scored with the B3000 normalization parameters.

Repeat `--factor-universe-workbook` to import several dated exports in one
run. When workbook arguments are supplied without portfolio arguments, the
cache script automatically selects portfolio CSVs whose dates match the
workbook dates.

By default, history ends on the latest selected portfolio date. The start date
is inferred from the earliest selected portfolio date using a 252-trading-day
lookback and a 2.25 calendar-day multiplier.

### Cache specific portfolios

```powershell
python .\bloomberg_cache.py `
    .\port_data\US_live_port_20260129.csv `
    .\port_data\US_live_port_20260130.csv
```

### Cache an explicit history range

```powershell
python .\bloomberg_cache.py `
    .\port_data\US_live_port_20260129.csv `
    .\port_data\US_live_port_20260130.csv `
    --start-date 2025-01-01 `
    --end-date 2026-01-30 `
    --benchmark VTHR `
    --factor-universe "B3000 Index" `
    --batch-size 50 `
    --max-retries 3
```

When preloading for `port_summary.py`, include both the report-date portfolio
and its previous-trading-day portfolio so tickers that were completely exited
are cached too. Running the cache command without portfolio arguments includes
all available portfolio files automatically.

Dates passed on the command line use `YYYY-MM-DD`, while dates inside portfolio
filenames use `YYYYMMDD`.

Run this to see every cache option:

```powershell
python .\bloomberg_cache.py --help
```

### Incremental cache behavior

The cache does not redownload a ticker's complete history on every run:

- A new ticker receives the full requested date range.
- An existing ticker receives only a missing leading or trailing range.
- Adding new historical fields triggers a one-time versioned backfill.
- Successful observations are inserted or updated by ticker and date.
- The current implementation does not scan for isolated holes inside an
  otherwise cached date range.

## Bloomberg fields and SQLite tables

The default database is `port_data/bloomberg_data.db`.

### `daily_prices`

One row per ticker and trading date:

| SQLite column | Bloomberg field | Use |
| --- | --- | --- |
| `close` | `PX_LAST` | Position valuation and dollar volume. |
| `total_return_index` | `TOT_RETURN_INDEX_GROSS_DVDS` | Returns, beta, volatility, covariance, relative strength, reversal, and liquidity returns. |
| `volume` | `PX_VOLUME` | Amihud liquidity. |
| `short_interest` | `SHORT_INT` | Short-interest numerator. |
| `equity_float` | `EQY_FLOAT` | Short-interest denominator. |
| `analyst_sentiment` | `EQY_REC_CONS` | Historical analyst-consensus level used to calculate recommendation changes. |

### `index_memberships`

One row per index, portfolio date, and member identifier. B3000 membership can
be imported from a PORT workbook using FIGIs or downloaded through Bloomberg's
`INDX_MWEIGHT_HIST` bulk field with an `END_DATE_OVERRIDE` equal to the
portfolio date. The cached membership defines the cross-sectional
normalization universe for custom factors.

If Bloomberg does not return a total-return index for an otherwise valid price
row, the downloader currently falls back to `PX_LAST` for that row.

### `historical_market_caps`

Stores `CUR_MKT_CAP` by ticker and portfolio date. Bloomberg returns this field
in millions, so the downloader multiplies it by 1,000,000 before storage.

### `security_metadata`

Stores the current Bloomberg snapshot of:

- `GICS_SECTOR_NAME`
- `PX_TO_BOOK_RATIO`

This table is not historically versioned. Therefore sector and price-to-book
are current cached metadata, while prices, total returns, market caps, short
interest, and analyst consensus are requested through the historical-data
interface.

### MAC3 Risk Model Files tables

The optional MAC3 integration adds four separate tables to the same database:

- `mac3_factor_exposures`
- `mac3_factor_covariance`
- `mac3_factor_returns`
- `mac3_model_snapshots`

These tables contain Bloomberg-delivered model values. They do not replace the
locally derived factor scores in `risk_metrics.py`; `port_summary.py` prints a
separate MAC3 section when `--mac3-model` is supplied.

## Import Bloomberg MAC3 factors

MAC3 Risk Model Files is a separately entitled Bloomberg product. Bloomberg's
[MAC3 product page](https://professional.bloomberg.com/products/risk/mac3/)
states that the service provides factor exposures, the variance-covariance
matrix, and factor returns in CSV format or through an API. It is not the
ordinary Desktop API `//blp/refdata` service used by `bloomberg_cache.py`.

Bloomberg does not publish the entitled API URL, authentication contract, or
customer-specific model identifiers on the public product page. Obtain those
details and a sample response from Bloomberg support before running the API
import. `mac3_cache.py` deliberately accepts the complete Bloomberg-issued
resource URLs instead of inventing an endpoint.

### Import from Bloomberg-issued API URLs

Put each secret header value in an environment variable. The command below is
an example; use the exact header name and resource URLs in your Bloomberg
contract:

```powershell
$env:MAC3_AUTHORIZATION = "Bearer YOUR_ISSUED_TOKEN"

python .\mac3_cache.py `
    --as-of-date 2026-01-30 `
    --model "YOUR_BLOOMBERG_MODEL_ID" `
    --horizon quarterly `
    --exposures-source "YOUR_BLOOMBERG_EXPOSURES_URL" `
    --covariance-source "YOUR_BLOOMBERG_VCV_URL" `
    --factor-returns-source "YOUR_BLOOMBERG_FACTOR_RETURNS_URL" `
    --api-header-env "Authorization=MAC3_AUTHORIZATION"
```

Repeat `--api-header-env` when the contract requires more than one header. For
mutual TLS, also pass `--client-certificate` and, when applicable,
`--client-key`. The importer accepts tabular JSON, CSV, or ZIP payloads. Use
the corresponding `--*-zip-member` option when a ZIP contains multiple CSVs.

### Import downloaded Risk Model Files

The same command accepts local paths and does not require API headers:

```powershell
python .\mac3_cache.py `
    --as-of-date 2026-01-30 `
    --model "YOUR_BLOOMBERG_MODEL_ID" `
    --horizon quarterly `
    --exposures-source .\mac3_data\exposures.csv `
    --covariance-source .\mac3_data\vcv.csv `
    --factor-returns-source .\mac3_data\factor_returns.csv
```

Long-form and wide matrix files are supported. Common column names are found
automatically. If Bloomberg's entitled schema uses different names, pass the
mapping explicitly; for example:

```powershell
python .\mac3_cache.py `
    --as-of-date 2026-01-30 `
    --model "YOUR_BLOOMBERG_MODEL_ID" `
    --exposures-source .\mac3_data\exposures.csv `
    --covariance-source .\mac3_data\vcv.csv `
    --exposure-security-column SECURITY_ID `
    --exposure-ticker-column TICKER `
    --exposure-factor-column FACTOR_ID `
    --exposure-value-column EXPOSURE `
    --covariance-row-factor-column FACTOR_1 `
    --covariance-column-factor-column FACTOR_2 `
    --covariance-value-column COVARIANCE
```

The database stores covariance in decimal-return-squared units and factor
returns in decimal-return units. The default input multipliers are `1`. If the
Bloomberg delivery contract defines VCV cells in percent-squared units, add
`--covariance-scale 0.0001`; if it defines factor returns in percentage-point
units, add `--factor-return-scale 0.01`. The applied scales and import time are
recorded in `mac3_model_snapshots`.

The ticker column must match the portfolio CSV ticker values. If the model
file contains only FIGIs or another identifier, include Bloomberg's ticker
mapping in the exported file before import or specify the file's ticker
column. The report refuses to silently treat unmatched securities as covered.

## Run the complete report

After preloading Bloomberg data, run:

```powershell
python .\port_summary.py .\port_data\US_live_port_20260130.csv
```

You can also import the matching universe workbook and download missing data
during the report run:

```powershell
python .\port_summary.py .\port_data\US_live_port_20260129.csv `
    --factor-universe-workbook `
    "C:\path\to\port_weights_20260129.xlsx"
```

The workbook's as-of date must equal the portfolio date. Add `--cache-only`
only after the FIGI universe data has already been preloaded.

After importing a matching MAC3 snapshot, add the model and horizon:

```powershell
python .\port_summary.py US_live_port_20260130.csv `
    --mac3-model "YOUR_BLOOMBERG_MODEL_ID" `
    --mac3-horizon quarterly
```

A bare filename is also resolved inside `port_data`:

```powershell
python .\port_summary.py US_live_port_20260130.csv
```

Run the long-only or short-only report with the corresponding split file:

```powershell
python .\port_summary.py long_positions_20260130.csv
python .\port_summary.py short_positions_20260130.csv
```

Use `divide_port.py` to generate both split files from a full portfolio:

```powershell
python .\divide_port.py US_live_port_20260130.csv
```

Useful options:

```powershell
python .\port_summary.py US_live_port_20260130.csv `
    --benchmark VTHR `
    --factor-universe "B3000 Index" `
    --risk-lookback-days 252 `
    --adv-lookback-days 20 `
    --liquidation-participation-rate 0.10 `
    --minimum-observations 60 `
    --downside-threshold 0.0 `
    --var-confidence 0.95 `
    --batch-size 50 `
    --max-retries 3
```

Use `--capital` to override the default gross exposure used as the weight
denominator:

```powershell
python .\port_summary.py US_live_port_20260130.csv `
    --capital 300000000
```

To save both standard output and errors to a log:

```powershell
python .\port_summary.py US_live_port_20260130.csv 2>&1 |
    Tee-Object -FilePath port_summary_output.log
```

Even without a separate preload command, `port_summary.py` first downloads any
missing historical fields and metadata, then calls the report calculations
with further price downloads disabled. The calculations therefore use the
SQLite snapshot established at the beginning of that run.

To bypass Bloomberg completely and use only the existing SQLite cache:

```powershell
python .\port_summary.py US_live_port_20260130.csv --cache-only
```

In cache-only mode, the script does not start a Bloomberg session or request
prices, volume, market caps, sectors, price-to-book, short interest, float, or
analyst consensus. Required missing data produces an error; partial coverage
continues to appear as warnings or blank position-level values. Run once
without `--cache-only` whenever the database needs to be updated.

## Portfolio summary calculations

Let shares be `q_i`, closing price be `P_i`, and capital be `C`.

### Number of stocks

The number of unique tickers in the current portfolio. It is not the number of
shares.

### Ticker coverage

A ticker is covered when it has a current price, market capitalization, and a
complete share-ADV and dollar-ADV lookback window. Positions with incomplete
coverage remain in the dated position-level CSVs.

### Capital

Unless `--capital` is supplied:

```text
C = sum_i |q_i P_i|
```

This is current gross absolute market exposure.

### Weighted average market cap

```text
absolute weight_i = |q_i P_i| / C
weighted average market cap = sum_i absolute weight_i * market_cap_i
```

Stocks with missing market cap are excluded from the sum, and the remaining
weights are not rescaled.

### Daily return

The current implementation follows the reinvested-gross-capital assumption:

```text
daily return = current gross exposure / previous gross exposure - 1
```

This is not a conventional holdings-based close-to-close P&L return. Changes
in portfolio composition and position sizing can affect it.

### Daily turnover

Each day's absolute weights are independently normalized by that day's gross
exposure:

```text
turnover_t = 1/2 * sum_i |w_i,t - w_i,t-1|
```

A ticker absent from either portfolio receives a zero weight for that day.

### Market value

```text
market value = sum_i q_i P_i
```

This is signed net market value. Market-value change is the current signed
market value minus the previous signed market value.

### Position differences

The program unions current and previous tickers and calculates:

```text
share difference_i = current shares_i - previous shares_i
```

Missing positions are treated as zero. Nonzero differences are written to:

```text
port_data/position_diff_YYYYMMDD.csv
port_data/position_diff_long_positions_YYYYMMDD.csv
port_data/position_diff_short_positions_YYYYMMDD.csv
```

### Average daily volume by position

The default ADV window is 20 trading days and can be changed with
`--adv-lookback-days`. For each current position:

```text
share ADV_i = mean(volume_i,t) over the trailing window
dollar ADV_i = mean(close_i,t * volume_i,t) over the trailing window
position shares / ADV_i = |shares_i| / share ADV_i
position value / dollar ADV_i = |shares_i * current price_i| / dollar ADV_i
percent of daily market volume_i = 100 * |shares_i| / share ADV_i
estimated liquidation days_i = (|shares_i| / share ADV_i)
                               / daily participation rate
```

The default liquidation participation rate is `0.10`, meaning the estimate
assumes the portfolio can trade up to 10% of each stock's normal daily share
volume per day. For example, a position equal to 25% of ADV has an estimated
liquidation time of `0.25 / 0.10 = 2.5` trading days, or three whole trading
days when rounded up. Change the assumption with
`--liquidation-participation-rate`; supply it as a decimal, such as `0.05` for
5% or `0.20` for 20%.

The report requires a complete lookback window for each ADV value. Every
portfolio ticker is included in the output; positions without enough valid
Bloomberg `PX_VOLUME` and `PX_LAST` observations have blank ADV fields. The
complete position table is written to:

```text
port_data/adv_by_position_YYYYMMDD.csv
port_data/adv_by_position_long_positions_YYYYMMDD.csv
port_data/adv_by_position_short_positions_YYYYMMDD.csv
```

The CSV contains shares, current price, signed position market value, share
ADV, dollar ADV, both position-to-ADV participation ratios, the explicit
percentage of average daily market volume, the assumed participation cap, and
estimated fractional and whole trading days to liquidate. Short-position
ratios use the absolute position size.

Here, “percent of the market” means percent of the stock's trailing average
daily reported trading volume—not percent of shares outstanding and not a
guarantee that this volume is immediately executable. The estimate assumes
volume and trading capacity remain constant. It does not model bid-ask spread,
market impact, intraday volume patterns, halts, borrow constraints, or changing
market conditions.

## Risk calculations

Risk calculations use signed weights:

```text
signed weight_i = q_i P_i / C
```

If capital is the default gross exposure, the absolute signed weights sum to
one, subject to missing prices.

### Position exposure

- Long exposure: sum of positive position values.
- Short exposure: sum of negative position values.
- Gross exposure: long exposure plus the absolute short exposure.
- Net exposure: long exposure plus short exposure.

### Beta

Each stock beta is estimated against VTHR by default:

```text
beta_i = covariance(stock return_i, benchmark return)
         / variance(benchmark return)
```

Portfolio beta exposure is:

```text
sum_i signed weight_i * beta_i
```

The default lookback is 252 trading days, with at least 60 paired observations
required for each beta.

### Sector exposure

GICS sector exposures are formed by grouping signed weights. The report shows
long, short, net, and gross exposure for each sector. Missing sectors are
reported as `Unclassified`.

### Raw style-factor definitions

| Factor | Raw stock-level definition |
| --- | --- |
| Size | `log(market cap)` |
| Value | `1 / price-to-book` for positive P/B values |
| Relative strength | Stock total-return growth divided by benchmark total-return growth minus one, measured from 63 to 5 trading days ago |
| One-day reversal | Negative of the latest completed close-to-close total return; a recent loser has a positive raw score |
| Short-term reversal | Negative exponentially weighted sum of the five returns immediately preceding the one-day-reversal return, using a 3-day half-life |
| Short interest | `SHORT_INT / EQY_FLOAT`, using the latest historical values on or before the report date |
| Sentiment | `EQY_REC_CONS_t - EQY_REC_CONS_t-20`; positive means analyst consensus improved |
| Volatility | Standard deviation of available daily total returns times `sqrt(252)` |
| Downside risk | Mean of `min(daily total return - tau, 0)^2` over the risk lookback; tau defaults to zero |
| Liquidity | 63-day Amihud illiquidity: average of `abs(return) / (close * volume)` |

The factor named `liquidity` is numerically an illiquidity measure. A larger raw
value means a larger price response per dollar traded and therefore lower
liquidity.

Downside-risk tau is a minimum acceptable daily return, not a benchmark. The
default `--downside-threshold 0.0` measures absolute downside by treating only
negative daily returns as shortfalls. A nonzero value must be supplied as a
daily decimal return. For example, do not pass an annual 4% target directly;
convert it to the corresponding daily threshold first.

### Factor normalization and portfolio exposure

For every factor, the raw stock values are:

1. Evaluated across the point-in-time B3000 membership.
2. Winsorized using the B3000 1st and 99th percentiles.
3. Converted using the winsorized B3000 mean and population standard
   deviation.

Portfolio factor exposure is then:

```text
factor exposure_f = sum_i signed weight_i * z_score_i,f
```

The default normalization universe is `B3000 Index`; change it with
`--factor-universe`. Portfolio securities outside B3000 are not allowed to
change the normalization parameters. When their raw data is available, they
are scored using the B3000 winsorization bounds, mean, and standard deviation.
Missing factor scores are excluded, and covered weights are not renormalized.
The report's `ticker_coverage` column should therefore be considered alongside
each exposure.

These are B3000-normalized custom exposures, not Bloomberg MAC3 exposures.
They use comparable z-score units but do not reproduce MAC3's proprietary
descriptor combinations, estimation weights, or orthogonalization.

Beta and sector exposures are not z-score normalized.

### Covariance volatility

The model calculates daily total returns, selects positions meeting the
minimum-observation requirement, subtracts each return series' mean, and fills
remaining missing returns with zero. It then estimates the ordinary sample
asset covariance matrix and annualizes it by multiplying by 252.

```text
annual variance = w' Cov_annual w
annual volatility = sqrt(annual variance)
daily volatility = annual volatility / sqrt(252)
```

This is an asset covariance model, not Bloomberg MAC3. With more stocks than
independent return observations, its covariance matrix can be rank-deficient.

### Bloomberg MAC3 factor risk

When requested, the separate MAC3 report uses the signed portfolio weight
vector `w`, Bloomberg's security-by-factor exposure matrix `X`, and the
delivered horizon-specific factor VCV matrix `F`:

```text
portfolio factor exposure = b = X' w
systematic variance = b' F b
factor variance contribution_k = b_k * (F b)_k
factor component volatility_k = factor variance contribution_k
                                / systematic volatility
```

Every common factor in the exposure and VCV files is printed and written to
`port_data/mac3_factor_risk_YYYYMMDD.csv`. No local regression, z-score, EWMA,
PCA approximation, or Yahoo Finance data is used in this MAC3 calculation.
The VCV is used in the units delivered for the selected MAC3 horizon.

### Parametric VaR

```text
one-day VaR = normal quantile * daily volatility * capital
```

The default confidence level is 95%. The reported number is a positive loss
magnitude and assumes normally distributed returns.

### Stock-level risk contribution

For each covered stock:

```text
marginal contribution_i = (Cov_annual * w)_i / portfolio volatility
component contribution_i = w_i * marginal contribution_i
percentage contribution_i = component contribution_i
                            / portfolio volatility
beta contribution_i = w_i * beta_i
```

The complete table is written to:

```text
port_data/risk_contribution_YYYYMMDD.csv
port_data/risk_contribution_long_positions_YYYYMMDD.csv
port_data/risk_contribution_short_positions_YYYYMMDD.csv
```

## Run individual scripts

Display shares and Bloomberg prices for one portfolio:

```powershell
python .\construct_port.py US_live_port_20260130.csv
```

Run the risk report without the portfolio-summary section:

```powershell
python .\risk_metrics.py US_live_port_20260130.csv
```

Show all options for any executable script:

```powershell
python .\bloomberg_cache.py --help
python .\mac3_cache.py --help
python .\port_summary.py --help
python .\risk_metrics.py --help
```

## Troubleshooting

### `No module named blpapi`

Install `blpapi` into the active virtual environment using Bloomberg's package
index. Confirm which Python is active with:

```powershell
python -c "import sys; print(sys.executable)"
```

### `Could not start a Bloomberg Desktop API session`

Confirm that Bloomberg Terminal is open and logged in on the same Windows
session. The script connects to `localhost:8194`; it does not connect to a
Terminal running on a different computer.

### Low ticker or factor coverage

Common causes include:

- The ticker needs a different Bloomberg identifier or yellow key.
- Bloomberg does not provide the requested field for that security.
- The user lacks entitlement to the field.
- The stock lacks enough return history.
- Short interest, float, analyst consensus, or price-to-book is unavailable.

Use `FLDS <GO>` in Bloomberg Terminal to confirm a field's historical
availability and your entitlement.

### Previous portfolio error

`port_summary.py` requires the exact previous NYSE trading-day file. Add the
missing file to `port_data`; the program intentionally does not substitute an
older portfolio.

### Permission denied when writing an output CSV

Close the existing output file in Excel and rerun the command. Excel commonly
locks open CSV files on Windows.

### Inspect whether data is cached

The SQLite database can be opened with any SQLite browser. The program will
also safely create or migrate its required tables the next time the cache
script runs.

### No matching MAC3 snapshot

The MAC3 report requires an exact match on portfolio date, model identifier,
and horizon. Run `mac3_cache.py` for that combination. If positions remain
unmatched, verify that the imported exposure ticker column uses the same
ticker strings as the portfolio CSV.

## Important modeling limitations

- Portfolio daily return is the change in gross exposure, not a holdings-based
  close-to-close performance return.
- Factor z-scores are normalized within the current portfolio rather than a
  broad market estimation universe.
- Sector and price-to-book metadata are current snapshots, not point-in-time
  history.
- The covariance estimator is a plain sample covariance matrix, not EWMA,
  PCA-shrunk, or factor-model covariance.
- Transaction costs, borrow fees, financing, cash, and execution timing are
  not modeled.
- Bloomberg field availability and historical coverage depend on the Terminal
  subscription and security.

These limitations should be addressed before treating the output as a
production risk model or comparing it directly with Bloomberg PORT/MAC3.
They apply to the locally calculated report. The optional MAC3 section instead
uses Bloomberg's delivered exposures and VCV, subject to model-file coverage
and the units and methodology specified in the customer's Bloomberg contract.
