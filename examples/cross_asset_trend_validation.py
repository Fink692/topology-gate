"""Validate the fixed 100-day trend rule across independent ETF risk assets.

No parameters are selected in this script. It downloads a fixed, pre-declared
universe, holds each risk ETF while it is above its prior 100-session average,
and holds cash otherwise. The result is an aggregate diagnostic only.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
from free_etf_strategy_research import _metrics

RISK_ASSETS = ("SPY", "QQQ", "DIA", "IWM", "EFA", "EEM")
LOOKBACK = 100
START_DATE = dt.date(2007, 1, 1)


def _epoch(value: dt.date) -> int:
    return int(dt.datetime.combine(value, dt.time(), tzinfo=dt.timezone.utc).timestamp())


def _download(cache_dir: Path, ticker: str) -> dict[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    end_date = dt.datetime.now(dt.timezone.utc).date() + dt.timedelta(days=1)
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?period1={_epoch(START_DATE)}&period2={_epoch(end_date)}"
        "&interval=1d&events=div%2Csplits&includeAdjustedClose=true"
    )
    request = urllib.request.Request(url, headers={"User-Agent": "topology-gate-cross-asset/1.0"})
    with urllib.request.urlopen(request, timeout=45) as response:
        raw = response.read()
    path = cache_dir / f"{ticker.lower()}.json"
    path.write_bytes(raw)
    payload = json.loads(raw)
    result = payload["chart"]["result"][0]
    values: dict[str, float] = {}
    for timestamp, value in zip(result["timestamp"], result["indicators"]["adjclose"][0]["adjclose"]):
        if value is None:
            continue
        number = float(value)
        if math.isfinite(number) and number > 0.0:
            date = dt.datetime.fromtimestamp(int(timestamp), tz=dt.timezone.utc).date().isoformat()
            values[date] = number
    return {"ticker": ticker, "values": values, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _path(prices: np.ndarray[Any, Any], cost_bps: float) -> np.ndarray[Any, Any]:
    positions = np.zeros(prices.shape[0], dtype=float)
    for index in range(LOOKBACK, prices.shape[0] - 1):
        positions[index] = float(prices[index] > np.mean(prices[index - LOOKBACK : index]))
    returns = prices[1:] / prices[:-1] - 1.0
    net: list[float] = []
    for index, realized in enumerate(returns):
        turnover = abs(positions[index] - positions[index - 1]) if index else abs(positions[index])
        net.append(float(positions[index] * realized - cost_bps / 10_000.0 * turnover))
    return np.asarray(net)


def run(*, cache_dir: Path, output: Path | None) -> dict[str, Any]:
    records = [_download(cache_dir, ticker) for ticker in (*RISK_ASSETS, "GLD")]
    series = {record["ticker"]: record["values"] for record in records}
    dates = tuple(sorted(set.intersection(*(set(values) for values in series.values()))))
    prices = {ticker: np.asarray([series[ticker][date] for date in dates], dtype=float) for ticker in series}
    holdout = np.asarray([date > "2022-12-31" for date in dates[1:]], dtype=bool)
    results: dict[str, Any] = {}
    for ticker in RISK_ASSETS:
        results[ticker] = {}
        for cost_bps in (5.0, 15.0, 25.0):
            net = _path(prices[ticker], cost_bps)
            results[ticker][f"{int(cost_bps)}bps"] = {
                "all": _metrics(net),
                "holdout_2023_onward": _metrics(net[holdout]),
            }
    result: dict[str, Any] = {
        "kind": "cross_asset_fixed_trend_validation",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "source": {
            "provider": "Yahoo Finance chart endpoint",
            "risk_assets": list(RISK_ASSETS),
            "defensive_mode": "cash",
            "common_start": dates[0],
            "common_end": dates[-1],
            "rows": len(dates),
            "manifest": [{key: value for key, value in record.items() if key != "values"} for record in records],
            "point_in_time_verified": False,
        },
        "rule": {"lookback_sessions": LOOKBACK, "signal_timing": "close compared with prior closes; next-session return"},
        "results": results,
        "claim_status": "public-final-history cross-asset diagnostic only",
        "limitations": [
            "The universe and adjusted histories are not point-in-time verified.",
            "This is not a live-trading system or profitability guarantee.",
            "Cash is modeled at zero return and does not include interest or operational friction.",
        ],
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path(tempfile.gettempdir()) / "topology-gate-cross-asset")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(cache_dir=args.cache_dir, output=args.output)
    print(json.dumps(result["results"], indent=2))


if __name__ == "__main__":
    main()
