"""Run the control-layer diagnostic against a private Quantiacs data session.

This adapter deliberately keeps the Quantiacs dependency outside the project
environment. Install the current ``qnt`` toolbox in a separate environment,
set the free-session ``API_KEY`` there, and run this script with ``src`` and
``examples`` on ``PYTHONPATH``.

The output is an aggregate engineering receipt. Do not commit the receipt or
share raw Quantiacs data or Quantiacs-specific derived analysis. This remains
``private-final-history diagnostic only`` and never opens the market gate.

Example (PowerShell)::

    $env:PYTHONPATH = 'src;examples'
    $env:API_KEY = '<your free Quantiacs API key>'
    python examples/quantiacs_private_diagnostic.py `
      --output "$env:TEMP\topology-gate-quantiacs-private.json"

The script reuses the checked-in baseline/control-layer implementation from
``public_market_diagnostic.py`` after replacing its price loader. It does not
reuse the public Yahoo source, and it never writes a raw-data cache here.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np
import public_market_diagnostic as diagnostic

from topology_gate.calibration import (
    CalibrationConfig,
    StationaryBlockBootstrap,
    calibrate_threshold,
)

DEFAULT_ASSETS = (
    "NAS:AAPL",
    "NAS:MSFT",
    "NAS:AMZN",
    "NAS:CSCO",
    "NAS:INTC",
    "NAS:QCOM",
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.floating):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _load_prices(
    *, min_date: str, max_date: str, assets: tuple[str, ...]
) -> tuple[np.ndarray[Any, Any], tuple[str, ...], dict[str, Any]]:
    try:
        import qnt.data as qndata  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "Quantiacs is not installed in this environment. Follow "
            "docs/quantiacs-private-track.md and install qnt outside this repo."
        ) from exc

    data = qndata.stocks.load_ndx_data(
        assets=list(assets),
        min_date=min_date,
        max_date=max_date,
        forward_order=True,
    )
    fields = {str(field) for field in data.coords["field"].values}
    required = {"close", "is_liquid"}
    missing = required - fields
    if missing:
        raise RuntimeError(f"Quantiacs data is missing required fields: {sorted(missing)}")

    close = np.asarray(data.sel(field="close").transpose("time", "asset"), dtype=float)
    dates = tuple(
        str(np.datetime_as_string(value, unit="D")) for value in data.coords["time"].values
    )
    if close.ndim != 2 or close.shape[1] != len(assets):
        raise RuntimeError("Quantiacs close panel has an unexpected shape")
    complete = np.all(np.isfinite(close) & (close > 0.0), axis=1)
    close = close[complete]
    dates = tuple(date for date, keep in zip(dates, complete) if keep)
    if close.shape[0] < 500:
        raise RuntimeError("Quantiacs panel has fewer than 500 complete observations")

    liquid = np.asarray(data.sel(field="is_liquid").transpose("time", "asset"), dtype=float)
    source = {
        "provider": "Quantiacs qnt.data.stocks.load_ndx_data",
        "dataset": "NASDAQ-100 historical stock data",
        "assets": list(assets),
        "requested_min_date": min_date,
        "requested_max_date": max_date,
        "returned_rows": int(close.shape[0]),
        "returned_start": dates[0],
        "returned_end": dates[-1],
        "fields": sorted(fields),
        "historical_membership_field": "is_liquid",
        "membership_finite_fraction": float(np.mean(np.isfinite(liquid))),
        "retrieved_at_utc": dt.datetime.now(dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "raw_data_written_to_repository": False,
    }
    return close, dates, source


def _calibrate(
    detector_factory: Callable[[float], Any],
    source: np.ndarray[Any, Any],
    *,
    family: str,
    candidates: tuple[float, ...],
    calibration_seed: int,
    evaluation_seed: int,
) -> tuple[Any | None, dict[str, Any]]:
    calibration_factory = StationaryBlockBootstrap(
        source, block_length=16, source_id="quantiacs-private:calibration-prefix:v1"
    )
    evaluation_factory = StationaryBlockBootstrap(
        source, block_length=16, source_id="quantiacs-private:calibration-prefix:v1"
    )
    try:
        result = calibrate_threshold(
            detector_factory,
            calibration_factory,
            evaluation_factory,
            detector_family_identity=family,
            candidate_thresholds=candidates,
            calibration_config=CalibrationConfig(
                trials=32,
                horizon=48,
                n_features=diagnostic.FEATURE_COUNT,
                seed=calibration_seed,
            ),
            evaluation_config=CalibrationConfig(
                trials=32,
                horizon=48,
                n_features=diagnostic.FEATURE_COUNT,
                seed=evaluation_seed,
            ),
            max_false_alarm_rate=0.20,
        )
    except (RuntimeError, ValueError, FloatingPointError) as exc:
        return None, {
            "approved": False,
            "diagnostic_threshold": candidates[-1],
            "error": f"{type(exc).__name__}: {exc}",
        }
    summary = result.to_dict()
    if not result.approved:
        return None, summary
    return result.to_certificate(), summary


def run(
    *,
    min_date: str,
    max_date: str,
    assets: tuple[str, ...],
    output: Path | None,
) -> dict[str, Any]:
    prices, price_dates, source = _load_prices(
        min_date=min_date, max_date=max_date, assets=assets
    )
    labels = tuple(asset.replace(":", "_") for asset in assets)
    diagnostic.TICKERS = labels
    diagnostic.FEATURE_NAMES = tuple(
        (
            *(f"ret_1d_{asset}" for asset in labels),
            *(f"ret_5d_{asset}" for asset in labels),
            *(f"vol_20d_{asset}" for asset in labels),
        )
    )
    diagnostic.FEATURE_COUNT = len(diagnostic.FEATURE_NAMES)
    diagnostic.RollingWindowRLS.n_features = diagnostic.FEATURE_COUNT

    raw_features, outcomes, realized_returns, row_dates = diagnostic.build_features(
        prices, price_dates
    )
    calibration_rows = max(256, int(len(raw_features) * 0.40))
    if calibration_rows >= len(raw_features) - 64:
        raise RuntimeError("Quantiacs history is too short for calibration and holdout splits")
    features, normalization = diagnostic.normalize_features(raw_features, calibration_rows)
    holdout_start = int(len(features) * 0.85)
    validation_start = calibration_rows
    calibration_source = features[:calibration_rows]
    pl_certificate, pl_calibration = _calibrate(
        diagnostic._pl_factory,
        calibration_source,
        family="persistent-laplacian-cusum-family:quantiacs-private:v1",
        candidates=(2.0, 8.0, 32.0, 128.0, 1024.0),
        calibration_seed=1103,
        evaluation_seed=2903,
    )
    cpd_certificate, cpd_calibration = _calibrate(
        diagnostic._cpd_factory,
        calibration_source,
        family="mean-covariance-cusum-family:quantiacs-private:v1",
        candidates=(2.0, 4.0, 8.0, 16.0, 32.0),
        calibration_seed=1307,
        evaluation_seed=3707,
    )
    pl_threshold = (
        pl_calibration.get("selected_threshold") if pl_certificate is not None else None
    )
    cpd_threshold = cpd_calibration.get("selected_threshold")
    if cpd_threshold is None:
        cpd_threshold = cpd_calibration.get("diagnostic_threshold")

    systems: dict[str, Any] = {}
    names = (
        "static-rls",
        "rolling-rls",
        "exponential-rls",
        "standard-cpd-rls",
        "certified-pl-rls",
    )
    for name in names:
        if name == "certified-pl-rls" and pl_threshold is None:
            systems[name] = {"status": "not_run", "reason": "PL calibration was not approved"}
            continue
        result = diagnostic._run_system(
            name,
            features,
            outcomes,
            realized_returns,
            pl_threshold=pl_threshold,
            pl_certificate=pl_certificate,
            cpd_threshold=cpd_threshold,
            cpd_certificate=cpd_certificate,
        )
        systems[name] = {
            "status": "ran",
            "detector_threshold": (
                cpd_threshold
                if name == "standard-cpd-rls"
                else pl_threshold
                if name == "certified-pl-rls"
                else None
            ),
            "validation": diagnostic._rolling_metrics(result, validation_start),
            "holdout": diagnostic._rolling_metrics(result, holdout_start),
            "all_rows": {key: value for key, value in result.metrics.items()},
            "accelerated_forgetting_count": int(result.metrics["accelerated_forgetting_count"]),
            "alarm_count": int(np.count_nonzero(result.alarms)),
            "detector_telemetry": diagnostic._detector_telemetry(result),
            "_result": result,
        }

    static_result = systems["static-rls"].pop("_result")
    pl_entry = systems.get("certified-pl-rls", {})
    pl_result = pl_entry.pop("_result", None)
    promotion = (
        diagnostic._promotion_audit(static_result, pl_result, validation_start, holdout_start)
        if pl_result is not None
        else {"status": "not_run", "reason": "PL calibration was not approved"}
    )
    for entry in systems.values():
        if isinstance(entry, dict):
            entry.pop("_result", None)

    summary = {
        "kind": "quantiacs_private_control_layer_diagnostic",
        "run_version": 1,
        "source": source,
        "data": {
            "price_rows": int(prices.shape[0]),
            "feature_rows": int(features.shape[0]),
            "price_start": price_dates[0],
            "price_end": price_dates[-1],
            "feature_start": row_dates[0],
            "feature_end": row_dates[-1],
            "feature_names": list(diagnostic.FEATURE_NAMES),
            "feature_count": diagnostic.FEATURE_COUNT,
            "target": "next-day equal-weight return divided by current mean 20-day volatility",
            "realized_return": "next-day equal-weight raw return",
            "label_delay": 1,
            "normalization_fit_rows": calibration_rows,
            "normalization": normalization,
        },
        "splits": {
            "calibration_rows": calibration_rows,
            "validation_start": validation_start,
            "holdout_start": holdout_start,
            "holdout_fraction": 0.15,
            "holdout_is_final_private_history_diagnostic": True,
        },
        "calibration": {
            "persistent_laplacian": pl_calibration,
            "mean_covariance": cpd_calibration,
        },
        "systems": systems,
        "promotion": promotion,
        "claim_status": "private-final-history diagnostic only",
        "vendor_gate_status": "not_evaluated",
        "limitations": [
            "This is a private final-history diagnostic, not a point-in-time source package.",
            "The documented is_liquid field is not a complete six-role universe and delisting ledger.",
            "The five-basis-point cost assumption is not Quantiacs execution evidence.",
            "No raw Quantiacs data or Quantiacs-specific derived tables are written here.",
        ],
    }
    safe = _json_safe(summary)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(safe, indent=2, sort_keys=True) + os.linesep, encoding="utf-8")
    return safe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-date", default="2007-01-01")
    parser.add_argument("--max-date", default="2026-08-04")
    parser.add_argument("--asset", action="append", dest="assets")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not os.environ.get("API_KEY", "").strip():
        raise SystemExit(
            "API_KEY is not set. Create a free Quantiacs session and set API_KEY "
            "in the isolated environment; no paid subscription is required."
        )
    assets = tuple(args.assets) if args.assets else DEFAULT_ASSETS
    result = run(
        min_date=args.min_date,
        max_date=args.max_date,
        assets=assets,
        output=args.output,
    )
    print(json.dumps({"claim_status": result["claim_status"], "systems": result["systems"]}, indent=2))


if __name__ == "__main__":
    main()
