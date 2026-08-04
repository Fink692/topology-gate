"""Search a small, pre-declared ETF strategy family without touching holdout data.

This is a research diagnostic on cached Yahoo Finance final adjusted history.
It is not a point-in-time or survivorship-free market study and it is not a
promise of future profitability. The search selects one low-turnover strategy
using only the tuning window, then reports an untouched holdout window and a
stress-cost version.

The cache is the one created by ``public_market_diagnostic.py``. Results should
be written outside the repository unless they are intentionally retained as a
diagnostic receipt.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

TICKERS = ("SPY", "TLT", "EFA", "EEM", "IWM", "GLD")
DEFAULT_CACHE = Path("reports")
DEFAULT_COST_BPS = 5.0
STRESS_COST_BPS = 15.0


def _load_prices(cache_dir: Path) -> tuple[np.ndarray[Any, Any], tuple[str, ...], list[dict[str, Any]]]:
    series: dict[str, dict[str, float]] = {}
    manifest: list[dict[str, Any]] = []
    for ticker in TICKERS:
        path = cache_dir / f"{ticker.lower()}.json"
        if not path.exists():
            raise RuntimeError(f"missing cached public payload: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        result = payload["chart"]["result"][0]
        dates = result["timestamp"]
        values = result["indicators"]["adjclose"][0]["adjclose"]
        if len(dates) != len(values):
            raise RuntimeError(f"{ticker}: timestamp/value length mismatch")
        rows: dict[str, float] = {}
        for timestamp, value in zip(dates, values):
            if value is None:
                continue
            date = dt.datetime.fromtimestamp(int(timestamp), tz=dt.timezone.utc).date().isoformat()
            number = float(value)
            if math.isfinite(number) and number > 0.0:
                rows[date] = number
        series[ticker] = rows
        manifest.append({"ticker": ticker, "source_file": path.name, "bytes": path.stat().st_size})
    common_dates = sorted(set.intersection(*(set(series[ticker]) for ticker in TICKERS)))
    prices = np.asarray([[series[ticker][date] for ticker in TICKERS] for date in common_dates])
    if prices.shape[0] < 500:
        raise RuntimeError("fewer than 500 common observations")
    return prices, tuple(common_dates), manifest


def _weights(
    prices: np.ndarray[Any, Any],
    *,
    lookback: int,
    skip: int,
    top_k: int,
    volatility_window: int,
    volatility_target: float,
    rebalance_days: int,
    momentum_threshold: float,
) -> np.ndarray[Any, Any]:
    returns = prices[1:] / prices[:-1] - 1.0
    positions = np.zeros_like(prices)
    start = max(lookback, volatility_window + 1, skip + 1)
    previous = np.zeros(prices.shape[1], dtype=float)
    for t in range(start, prices.shape[0] - 1):
        current = previous.copy()
        if (t - start) % rebalance_days == 0:
            momentum = prices[t - skip] / prices[t - lookback] - 1.0
            volatility = np.std(returns[t - volatility_window : t], axis=0, ddof=1)
            eligible = np.flatnonzero(momentum > momentum_threshold)
            if eligible.size:
                ranked = eligible[np.argsort(momentum[eligible])[::-1]][:top_k]
                inverse_vol = 1.0 / np.maximum(volatility[ranked] * math.sqrt(252.0), 1.0e-6)
                raw = inverse_vol / np.sum(inverse_vol)
                portfolio_vol = float(np.sqrt(np.sum((raw * volatility[ranked]) ** 2)) * math.sqrt(252.0))
                scale = min(1.0, volatility_target / max(portfolio_vol, 1.0e-8))
                current = np.zeros(prices.shape[1], dtype=float)
                current[ranked] = raw * scale
            else:
                current = np.zeros(prices.shape[1], dtype=float)
        positions[t] = current
        previous = current
    return positions


def _fixed_weights(prices: np.ndarray[Any, Any], weights: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    return np.repeat(np.asarray(weights, dtype=float)[None, :], prices.shape[0], axis=0)


def _path(
    prices: np.ndarray[Any, Any],
    positions: np.ndarray[Any, Any],
    *,
    cost_bps: float,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    returns = prices[1:] / prices[:-1] - 1.0
    gross: list[float] = []
    net: list[float] = []
    for t in range(prices.shape[0] - 1):
        turnover = float(np.sum(np.abs(positions[t] - positions[t - 1]))) if t else float(np.sum(np.abs(positions[t])))
        gross_return = float(positions[t] @ returns[t])
        gross.append(gross_return)
        net.append(gross_return - cost_bps / 10_000.0 * turnover)
    return np.asarray(gross), np.asarray(net)


def _max_drawdown(returns: np.ndarray[Any, Any]) -> float:
    equity = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(equity)
    return float(np.max(1.0 - equity / np.maximum(peak, 1.0e-12)))


def _metrics(returns: np.ndarray[Any, Any]) -> dict[str, float]:
    if returns.size < 2:
        return {"rows": float(returns.size), "net_sharpe": 0.0, "annualized_return": 0.0, "max_drawdown": 0.0, "turnover_proxy": 0.0}
    annualized = float(np.prod(1.0 + returns) ** (252.0 / returns.size) - 1.0)
    std = float(np.std(returns, ddof=1))
    return {
        "rows": float(returns.size),
        "net_sharpe": 0.0 if std <= 1.0e-15 else float(np.mean(returns) / std * math.sqrt(252.0)),
        "annualized_return": annualized,
        "max_drawdown": _max_drawdown(returns),
        "mean_daily_return": float(np.mean(returns)),
    }


def _yearly_sharpes(returns: np.ndarray[Any, Any], dates: tuple[str, ...]) -> list[float]:
    years = sorted({date[:4] for date in dates})
    values: list[float] = []
    for year in years:
        selection = np.asarray([date[:4] == year for date in dates])
        values.append(_metrics(returns[selection])["net_sharpe"])
    return values


def _parameter_grid() -> list[dict[str, Any]]:
    grid: list[dict[str, Any]] = []
    for lookback in (21, 63, 126, 252):
        for skip in (0, 5, 21):
            for top_k in (1, 2, 3):
                for volatility_window in (21, 63):
                    for volatility_target in (0.10, 0.20, 0.30):
                        for rebalance_days in (1, 5, 21):
                            for momentum_threshold in (0.0, 0.02, 0.05):
                                grid.append({
                                    "lookback": lookback,
                                    "skip": skip,
                                    "top_k": top_k,
                                    "volatility_window": volatility_window,
                                    "volatility_target": volatility_target,
                                    "rebalance_days": rebalance_days,
                                    "momentum_threshold": momentum_threshold,
                                })
    return grid


def run(*, cache_dir: Path, output: Path | None) -> dict[str, Any]:
    prices, dates, manifest = _load_prices(cache_dir)
    returns_dates = dates[1:]
    pre_end = "2019-12-31"
    tuning_end = "2022-12-31"
    pre_mask = np.asarray([date <= pre_end for date in returns_dates])
    tuning_mask = np.asarray([(date > pre_end) and (date <= tuning_end) for date in returns_dates])
    holdout_mask = np.asarray([date > tuning_end for date in returns_dates])
    candidates: list[dict[str, Any]] = []
    for params in _parameter_grid():
        positions = _weights(prices, **params)
        _, net = _path(prices, positions, cost_bps=DEFAULT_COST_BPS)
        tuning = net[tuning_mask]
        yearly = _yearly_sharpes(net, returns_dates)
        score = float(np.median(yearly[-7:])) if yearly else -math.inf
        candidates.append({"params": params, "tuning": _metrics(tuning), "selection_score": score, "yearly_sharpes": yearly})
    candidates.sort(key=lambda row: (row["selection_score"], row["tuning"]["net_sharpe"], row["tuning"]["max_drawdown"]), reverse=True)
    selected = candidates[0]
    positions = _weights(prices, **selected["params"])
    gross, net = _path(prices, positions, cost_bps=DEFAULT_COST_BPS)
    _, stress_net = _path(prices, positions, cost_bps=STRESS_COST_BPS)

    baselines: dict[str, Any] = {}
    for name, weights in {
        "equal_weight": np.full(len(TICKERS), 1.0 / len(TICKERS)),
        "spy_tlt_60_40": np.asarray([0.60, 0.40, 0.0, 0.0, 0.0, 0.0]),
        "spy_only": np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    }.items():
        _, baseline_net = _path(prices, _fixed_weights(prices, weights), cost_bps=DEFAULT_COST_BPS)
        baselines[name] = {"pre": _metrics(baseline_net[pre_mask]), "tuning": _metrics(baseline_net[tuning_mask]), "holdout": _metrics(baseline_net[holdout_mask])}

    result: dict[str, Any] = {
        "kind": "free_etf_strategy_research",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "source": {"provider": "Yahoo Finance chart endpoint cached adjusted history", "tickers": list(TICKERS), "manifest": manifest, "point_in_time_verified": False},
        "selection": {"pre_end": pre_end, "tuning_end": tuning_end, "holdout_start": "2023-01-01", "candidate_count": len(candidates), "selected": selected, "top_10": candidates[:10]},
        "selected_strategy": {"pre": _metrics(net[pre_mask]), "tuning": _metrics(net[tuning_mask]), "holdout": _metrics(net[holdout_mask]), "holdout_stress_15bps": _metrics(stress_net[holdout_mask]), "gross_holdout": _metrics(gross[holdout_mask])},
        "baselines": baselines,
        "claim_status": "public-final-history research diagnostic only",
        "limitations": ["Parameters were selected on pre-holdout tuning data, but this is still a final-history public source.", "The ETF universe, corporate actions, delistings, and source revisions are not point-in-time verified.", "No profitability guarantee or live-trading recommendation is made."],
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(cache_dir=args.cache_dir, output=args.output)
    print(json.dumps(result["selected_strategy"], indent=2))


if __name__ == "__main__":
    main()
