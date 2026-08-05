"""Run strict annual walk-forward validation for the SPY/GLD trend family.

For each calendar year, the lookback and defensive-asset choice are selected
using only observations strictly before that year. The selected rule is then
evaluated on that year once. This is stronger evidence than selecting once and
reading a single favorable final window, but it still uses final adjusted
public history and is not a live-profit guarantee.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

import numpy as np
from free_etf_strategy_research import _load_prices, _metrics, _path
from trend_filter_paper_model import (
    LOOKBACK_CANDIDATES,
    SAFE_ASSET_CANDIDATES,
    _positions,
)


def _year_mask(dates: tuple[str, ...], year: str) -> np.ndarray[Any, Any]:
    return np.asarray([date[:4] == year for date in dates[1:]], dtype=bool)


def _select_for_year(
    prices: np.ndarray[Any, Any],
    dates: tuple[str, ...],
    year: str,
    cost_bps: float,
) -> tuple[int, int | None, np.ndarray[Any, Any], dict[str, float]]:
    test_start = f"{year}-01-01"
    train_mask = np.asarray([date < test_start for date in dates[1:]], dtype=bool)
    best: tuple[tuple[float, float, float], int, int | None, np.ndarray[Any, Any], dict[str, float]] | None = None
    for lookback in LOOKBACK_CANDIDATES:
        for safe_asset in SAFE_ASSET_CANDIDATES:
            _, net = _path(prices, _positions(prices, lookback, safe_asset), cost_bps=cost_bps)
            metrics = _metrics(net[train_mask])
            key = (
                metrics["net_sharpe"],
                metrics["annualized_return"],
                -metrics["max_drawdown"],
            )
            candidate = (key, lookback, safe_asset, net, metrics)
            if best is None or key > best[0]:
                best = candidate
    if best is None:
        raise RuntimeError(f"no walk-forward candidate for {year}")
    return best[1], best[2], best[3], best[4]


def run(*, cache_dir: Path, output: Path | None) -> dict[str, Any]:
    prices, dates, manifest = _load_prices(cache_dir)
    years = [str(year) for year in range(2012, int(dates[-1][:4]) + 1)]
    scenarios: dict[str, Any] = {}
    for cost_bps in (5.0, 15.0, 25.0):
        annual_rows: list[dict[str, Any]] = []
        walk_returns: list[np.ndarray[Any, Any]] = []
        for year in years:
            test_mask = _year_mask(dates, year)
            if not np.any(test_mask):
                continue
            lookback, safe_asset, net, training_metrics = _select_for_year(
                prices, dates, year, cost_bps
            )
            test_metrics = _metrics(net[test_mask])
            walk_returns.append(net[test_mask])
            annual_rows.append(
                {
                    "year": year,
                    "lookback": lookback,
                    "safe_asset": "GLD" if safe_asset == 5 else "cash",
                    "training": training_metrics,
                    "test": test_metrics,
                }
            )
        walk = np.concatenate(walk_returns)
        scenarios[f"{int(cost_bps)}bps"] = {
            "walk_forward": _metrics(walk),
            "annual": annual_rows,
        }

    result: dict[str, Any] = {
        "kind": "strict_annual_walkforward_trend_filter",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "source": {
            "provider": "Yahoo Finance chart endpoint cached adjusted history",
            "manifest": manifest,
            "point_in_time_verified": False,
        },
        "protocol": {
            "evaluation_years": years,
            "parameter_selection_rule": "maximize prior-years net Sharpe, then annualized return, then lower drawdown",
            "future_year_used_for_selection": False,
            "candidate_lookbacks": list(LOOKBACK_CANDIDATES),
            "candidate_defensive_assets": ["cash", "GLD"],
        },
        "scenarios": scenarios,
        "claim_status": "public-final-history strict walk-forward diagnostic only",
        "limitations": [
            "Final adjusted public history is not point-in-time evidence.",
            "The walk-forward has negative years, including 2022.",
            "Historical positive returns do not guarantee future profit or execution quality.",
            "No live orders are placed.",
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
    print(json.dumps({key: value["walk_forward"] for key, value in result["scenarios"].items()}, indent=2))


if __name__ == "__main__":
    main()
