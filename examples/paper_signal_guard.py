"""Emit a guarded paper signal for the SPY/GLD trend model.

This command never places orders. It rejects stale or malformed input and
records explicit limits so a later broker integration cannot silently inherit
leverage or live execution.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

import numpy as np
from free_etf_strategy_research import _load_prices
from trend_filter_paper_model import _positions

STALE_AFTER_CALENDAR_DAYS = 3
MAX_GROSS_EXPOSURE = 1.0


def run(*, cache_dir: Path, output: Path | None) -> dict[str, Any]:
    prices, dates, manifest = _load_prices(cache_dir)
    last_date = dt.date.fromisoformat(dates[-1])
    today = dt.datetime.now(dt.timezone.utc).date()
    age_days = (today - last_date).days
    if age_days > STALE_AFTER_CALENDAR_DAYS:
        raise RuntimeError(
            f"paper signal rejected: source is {age_days} calendar days stale "
            f"(last observation {last_date.isoformat()})"
        )
    if not np.all(np.isfinite(prices[-101:])) or np.any(prices[-101:] <= 0.0):
        raise RuntimeError("paper signal rejected: latest price window is invalid")

    positions = _positions(prices, 100, 5)
    signal = positions[-2].copy()
    gross = float(np.sum(np.abs(signal)))
    if gross > MAX_GROSS_EXPOSURE + 1.0e-12:
        raise RuntimeError("paper signal rejected: gross exposure exceeds limit")
    assets = ("SPY", "TLT", "EFA", "EEM", "IWM", "GLD")
    target = {asset: float(weight) for asset, weight in zip(assets, signal) if weight > 0.0}
    result: dict[str, Any] = {
        "kind": "guarded_paper_signal",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "source": {"last_observation_date": dates[-1], "age_calendar_days": age_days, "manifest": manifest},
        "model": {"risk_asset": "SPY", "defensive_asset": "GLD", "lookback_sessions": 100},
        "signal": {"decision_for_next_session": target, "gross_exposure": gross, "signal_observation_date": dates[-2]},
        "risk_controls": {
            "paper_only": True,
            "live_execution": False,
            "manual_confirmation_required": True,
            "max_gross_exposure": MAX_GROSS_EXPOSURE,
            "shorting_allowed": False,
            "leverage_allowed": False,
            "stale_after_calendar_days": STALE_AFTER_CALENDAR_DAYS,
        },
        "status": "paper_signal_only",
        "warning": "A paper signal is not a promise of profit and is not a trade instruction.",
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(cache_dir=args.cache_dir, output=args.output), indent=2))


if __name__ == "__main__":
    main()
