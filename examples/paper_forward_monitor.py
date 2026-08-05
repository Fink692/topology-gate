"""Update the free-data cache and append one guarded paper signal.

The monitor is intentionally paper-only. It downloads final adjusted daily
history into a cache outside the repository, settles the previous signal when
the next daily observation exists, and appends the next signal to an external
JSONL ledger. It never places orders.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
from free_etf_strategy_research import TICKERS, _load_prices

SOURCE_URL = (
    "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    "?period1={start}&period2={end}&interval=1d&events=div%2Csplits"
    "&includeAdjustedClose=true"
)
START_DATE = dt.date(2007, 1, 1)


def _epoch(value: dt.date) -> int:
    return int(dt.datetime.combine(value, dt.time(), tzinfo=dt.timezone.utc).timestamp())


def _refresh_cache(cache_dir: Path) -> list[dict[str, Any]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    end_date = dt.datetime.now(dt.timezone.utc).date() + dt.timedelta(days=1)
    manifest: list[dict[str, Any]] = []
    for ticker in TICKERS:
        url = SOURCE_URL.format(ticker=ticker, start=_epoch(START_DATE), end=_epoch(end_date))
        request = urllib.request.Request(
            url, headers={"User-Agent": "topology-gate-paper-monitor/1.0"}
        )
        with urllib.request.urlopen(request, timeout=45) as response:
            raw = response.read()
        path = cache_dir / f"{ticker.lower()}.json"
        path.write_bytes(raw)
        manifest.append({"ticker": ticker, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
    return manifest


def _read_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _append_ledger(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def run(*, cache_dir: Path, ledger: Path, refresh: bool) -> dict[str, Any]:
    source_manifest = _refresh_cache(cache_dir) if refresh else []
    prices, dates, cached_manifest = _load_prices(cache_dir)
    if prices.shape[0] < 101:
        raise RuntimeError("paper monitor requires at least 101 complete observations")
    lookback = 100
    risk_on = bool(prices[-1, 0] > np.mean(prices[-lookback - 1 : -1, 0]))
    asset = "SPY" if risk_on else "GLD"
    ledger_rows = _read_ledger(ledger)
    settled: list[dict[str, Any]] = []
    settled_ids = {
        str(row.get("settlement_for")) for row in ledger_rows if row.get("kind") == "paper_settlement"
    }
    date_index = {date: index for index, date in enumerate(dates)}
    for row in ledger_rows:
        if row.get("kind") != "paper_signal":
            continue
        observation_date = str(row["signal_observation_date"])
        index = date_index.get(observation_date)
        if index is None or index + 1 >= len(dates) or observation_date in settled_ids:
            continue
        realized_asset = str(row["asset"])
        realized_index = TICKERS.index(realized_asset)
        realized = float(prices[index + 1, realized_index] / prices[index, realized_index] - 1.0)
        net = realized - 0.0005
        settlement = {
            "kind": "paper_settlement",
            "settlement_for": observation_date,
            "realization_date": dates[index + 1],
            "asset": realized_asset,
            "gross_return": realized,
            "assumed_cost_bps": 5.0,
            "net_return": net,
        }
        _append_ledger(ledger, settlement)
        settled.append(settlement)

    signal = {
        "kind": "paper_signal",
        "signal_observation_date": dates[-1],
        "asset": asset,
        "risk_on": risk_on,
        "last_spy_close": float(prices[-1, 0]),
        "trailing_spy_average": float(np.mean(prices[-lookback - 1 : -1, 0])),
        "assumed_cost_bps": 5.0,
        "paper_only": True,
        "live_execution": False,
    }
    existing_signal_dates = {
        str(row.get("signal_observation_date"))
        for row in ledger_rows
        if row.get("kind") == "paper_signal"
    }
    new_signal = dates[-1] not in existing_signal_dates
    if new_signal:
        _append_ledger(ledger, signal)

    ledger_after = _read_ledger(ledger)
    net_returns = np.asarray(
        [float(row["net_return"]) for row in ledger_after if row.get("kind") == "paper_settlement"],
        dtype=float,
    )
    cumulative = float(np.prod(1.0 + net_returns) - 1.0) if net_returns.size else 0.0
    return {
        "kind": "paper_forward_monitor_receipt",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "source": {"download_manifest": source_manifest, "cached_manifest": cached_manifest, "last_observation_date": dates[-1]},
        "signal": signal,
        "new_signal_written": new_signal,
        "settlements_written": settled,
        "settled_observation_count": int(net_returns.size),
        "cumulative_net_return": cumulative,
        "ledger": str(ledger),
        "status": "paper_only",
        "warning": "This is not a live trading system and positive historical or paper returns do not guarantee future profit.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path(tempfile.gettempdir()) / "topology-gate-paper-monitor")
    parser.add_argument("--ledger", type=Path, default=Path(tempfile.gettempdir()) / "topology-gate-paper-ledger.jsonl")
    parser.add_argument("--no-refresh", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(cache_dir=args.cache_dir, ledger=args.ledger, refresh=not args.no_refresh), indent=2))


if __name__ == "__main__":
    main()
