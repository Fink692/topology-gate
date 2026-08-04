"""Evaluate a small, unlevered SPY/GLD paper-trading model.

Rule at the close of day ``t``:

* hold SPY for the next session when today's SPY close is above its trailing
  100-session average (computed from prior closes);
* otherwise hold GLD;
* use no leverage and charge turnover costs at the position change.

The 100-session lookback and defensive asset are selected only on the training
window ending 2015-12-31. The tuning window ends 2022-12-31 and is reported
separately from the final public-history diagnostic holdout beginning
2023-01-01. This is a historical paper model, not a profitability guarantee
or an instruction to trade real money.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

import numpy as np
from free_etf_strategy_research import _load_prices, _metrics, _path

LOOKBACK_CANDIDATES = (50, 75, 100, 125, 150, 175, 200, 225, 252)
SAFE_ASSET_CANDIDATES: tuple[int | None, ...] = (None, 5)
TRAINING_END = "2015-12-31"
TUNING_END = "2022-12-31"


def _positions(prices: np.ndarray[Any, Any], lookback: int, safe_asset: int | None) -> np.ndarray[Any, Any]:
    positions = np.zeros_like(prices)
    for index in range(lookback, prices.shape[0] - 1):
        risk_on = prices[index, 0] > np.mean(prices[index - lookback : index, 0])
        positions[index, 0] = float(risk_on)
        if not risk_on and safe_asset is not None:
            positions[index, safe_asset] = 1.0
    return positions


def _mask(dates: tuple[str, ...], start: str | None, end: str | None) -> np.ndarray[Any, Any]:
    return np.asarray(
        [
            (start is None or date >= start) and (end is None or date <= end)
            for date in dates[1:]
        ],
        dtype=bool,
    )


def run(*, cache_dir: Path, output: Path | None) -> dict[str, Any]:
    prices, dates, manifest = _load_prices(cache_dir)
    return_dates = dates[1:]
    training = _mask(dates, None, TRAINING_END)
    tuning = _mask(dates, "2016-01-01", TUNING_END)
    holdout = _mask(dates, "2023-01-01", None)

    candidates: list[dict[str, Any]] = []
    for lookback in LOOKBACK_CANDIDATES:
        for safe_asset in SAFE_ASSET_CANDIDATES:
            positions = _positions(prices, lookback, safe_asset)
            _, net = _path(prices, positions, cost_bps=5.0)
            candidates.append(
                {
                    "lookback": lookback,
                    "safe_asset": "GLD" if safe_asset == 5 else "cash",
                    "training": _metrics(net[training]),
                }
            )
    candidates.sort(
        key=lambda row: (
            row["training"]["net_sharpe"],
            row["training"]["annualized_return"],
            -row["training"]["max_drawdown"],
        ),
        reverse=True,
    )
    selected = candidates[0]
    safe_asset = 5 if selected["safe_asset"] == "GLD" else None
    positions = _positions(prices, int(selected["lookback"]), safe_asset)

    costs: dict[str, dict[str, Any]] = {}
    for cost_bps in (5.0, 15.0, 25.0):
        _, net = _path(prices, positions, cost_bps=cost_bps)
        costs[str(int(cost_bps)) + "bps"] = {
            "training": _metrics(net[training]),
            "tuning": _metrics(net[tuning]),
            "holdout": _metrics(net[holdout]),
            "annual_by_year": {
                year: _metrics(net[np.asarray([date[:4] == year for date in return_dates])])
                for year in sorted({date[:4] for date in return_dates})
            },
        }

    last_close = float(prices[-1, 0])
    trailing_average = float(np.mean(prices[-int(selected["lookback"]) - 1 : -1, 0]))
    risk_on = last_close > trailing_average
    result: dict[str, Any] = {
        "kind": "unlevered_spy_gld_trend_filter_paper_model",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "source": {
            "provider": "Yahoo Finance chart endpoint cached adjusted history",
            "tickers": ["SPY", "TLT", "EFA", "EEM", "IWM", "GLD"],
            "manifest": manifest,
            "point_in_time_verified": False,
        },
        "rule": {
            "risk_asset": "SPY",
            "defensive_asset": selected["safe_asset"],
            "lookback_sessions": selected["lookback"],
            "signal_timing": "close_t compared with trailing prior closes; position held for t+1",
            "leverage": 1.0,
            "selection_training_end": TRAINING_END,
            "tuning_end": TUNING_END,
            "holdout_start": "2023-01-01",
        },
        "selection": {"candidate_count": len(candidates), "selected": selected, "all_candidates": candidates},
        "cost_scenarios": costs,
        "current_paper_signal": {
            "last_observation_date": dates[-1],
            "last_spy_close": last_close,
            "trailing_average": trailing_average,
            "risk_on": risk_on,
            "next_session_asset": "SPY" if risk_on else selected["safe_asset"],
        },
        "claim_status": "public-final-history paper-model diagnostic only",
        "limitations": [
            "Positive historical returns do not establish future profitability.",
            "The public adjusted-price history is not point-in-time and does not prove delisting, universe, or execution coverage.",
            "The model is not connected to a broker and places no live orders.",
            "A loss is possible, including during gaps, fast reversals, and data outages.",
        ],
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("reports"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(cache_dir=args.cache_dir, output=args.output)
    print(json.dumps({"selected": result["selection"]["selected"], "cost_scenarios": result["cost_scenarios"], "current_paper_signal": result["current_paper_signal"]}, indent=2))


if __name__ == "__main__":
    main()
