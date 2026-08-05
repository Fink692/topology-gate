"""Run a no-cost sensitivity study for the public PL-RLS diagnostic.

This study keeps the public final-history source boundary unchanged.  It tests
the detector's declared drift parameter, because the base diagnostic showed a
zero PL-CUSUM score and therefore never changed the learner's neutral memory.
Each cell is calibrated independently so an approved cell cannot borrow a
certificate from another detector configuration.

The result is exploratory public-proxy evidence, not a market-performance
claim or a substitute for point-in-time source validation.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from public_market_diagnostic import (
    FEATURE_COUNT,
    _calibrate,
    _detector_telemetry,
    _promotion_audit,
    _rolling_metrics,
    build_features,
    load_prices,
    normalize_features,
)

from topology_gate import RLS, OnlineRunConfig, RLSConfig, run_recursive_rls
from topology_gate.persistent import (
    PersistentLaplacianBackend,
    PersistentLaplacianConfig,
)
from topology_gate.pl_cusum import PersistentCUSUMConfig, PersistentLaplacianCUSUM

SENSITIVITY_CELLS = (
    {"name": "default-scale-drift-0.00", "drift": 0.00},
    {"name": "default-scale-drift-0.25", "drift": 0.25},
    {"name": "default-scale-drift-0.50", "drift": 0.50},
    {
        "name": "fixed-scale-2.00-20.00",
        "drift": 0.50,
        "scale_s": 2.0,
        "scale_t": 20.0,
        "forgetting_sensitivity": 1.0,
    },
    {
        "name": "fixed-scale-sensitivity-0.10",
        "drift": 0.50,
        "scale_s": 2.0,
        "scale_t": 20.0,
        "forgetting_sensitivity": 0.10,
    },
    {
        "name": "fixed-scale-sensitivity-0.01",
        "drift": 0.50,
        "scale_s": 2.0,
        "scale_t": 20.0,
        "forgetting_sensitivity": 0.01,
    },
    {
        "name": "fixed-scale-sensitivity-0.0001",
        "drift": 0.50,
        "scale_s": 2.0,
        "scale_t": 20.0,
        "forgetting_sensitivity": 0.0001,
    },
    {
        "name": "fixed-scale-sensitivity-0.00001",
        "drift": 0.50,
        "scale_s": 2.0,
        "scale_t": 20.0,
        "forgetting_sensitivity": 0.00001,
    },
)


def _pl_factory(
    threshold: float,
    *,
    drift: float,
    scale_s: float | None = None,
    scale_t: float | None = None,
    forgetting_sensitivity: float = 1.0,
) -> PersistentLaplacianCUSUM:
    backend = PersistentLaplacianBackend(
        PersistentLaplacianConfig(
            max_vertices=4,
            max_simplices=100,
            q=0,
            scale_s=scale_s,
            scale_t=scale_t,
            n_eigenvalues=4,
        )
    )
    return PersistentLaplacianCUSUM(
        backend,
        PersistentCUSUMConfig(
            cloud_window=4,
            min_points=4,
            backend_eigenvalues=4,
            positive_spectrum_width=2,
            betti_dimensions=(0, 1),
            calibration_window=8,
            calibration_min_periods=8,
            drift=drift,
            forgetting_lambda_min=0.8,
            forgetting_lambda_max=0.99,
            forgetting_sensitivity=forgetting_sensitivity,
            threshold=threshold,
        ),
    )


def _run_static(
    features: np.ndarray[Any, Any],
    outcomes: np.ndarray[Any, Any],
    realized_returns: np.ndarray[Any, Any],
) -> Any:
    return run_recursive_rls(
        features,
        outcomes,
        realized_returns=realized_returns,
        learner=RLS(
            RLSConfig(
                n_features=FEATURE_COUNT,
                ridge=1.0,
                forgetting_factor=1.0,
                lambda_min=0.8,
                lambda_max=1.0,
            )
        ),
        config=OnlineRunConfig(
            label_delay=1,
            transaction_cost_bps=5.0,
            position_limit=1.0,
            position_scale=0.01,
            require_realized_returns=True,
        ),
    )


def _run_cell(
    features: np.ndarray[Any, Any],
    outcomes: np.ndarray[Any, Any],
    realized_returns: np.ndarray[Any, Any],
    *,
    drift: float,
    threshold: float,
    scale_s: float | None,
    scale_t: float | None,
    forgetting_sensitivity: float,
    certificate: Any,
) -> Any:
    detector = _pl_factory(
        threshold,
        drift=drift,
        scale_s=scale_s,
        scale_t=scale_t,
        forgetting_sensitivity=forgetting_sensitivity,
    )
    return run_recursive_rls(
        features,
        outcomes,
        realized_returns=realized_returns,
        learner=RLS(
            RLSConfig(
                n_features=FEATURE_COUNT,
                ridge=1.0,
                forgetting_factor=0.99,
                lambda_min=0.8,
                lambda_max=0.99,
            )
        ),
        detector=detector,
        market_states=features,
        config=OnlineRunConfig(
            label_delay=1,
            transaction_cost_bps=5.0,
            position_limit=1.0,
            position_scale=0.01,
            require_realized_returns=True,
        ),
        calibration=certificate,
    )


def _finite(value: Any) -> float | None:
    converted = float(value)
    return converted if math.isfinite(converted) else None


def run(*, cache_dir: Path, output: Path | None, refresh: bool) -> dict[str, Any]:
    prices, price_dates, _ = load_prices(cache_dir=cache_dir, refresh=refresh)
    raw_features, outcomes, realized_returns, _ = build_features(prices, price_dates)
    calibration_rows = max(256, int(len(raw_features) * 0.40))
    features, _ = normalize_features(raw_features, calibration_rows)
    validation_start = calibration_rows
    holdout_start = int(len(features) * 0.85)
    calibration_source = features[:calibration_rows]
    static = _run_static(features, outcomes, realized_returns)
    rows: list[dict[str, Any]] = []

    for index, cell in enumerate(SENSITIVITY_CELLS):
        drift = float(cell["drift"])
        scale_s = cell.get("scale_s")
        scale_t = cell.get("scale_t")
        forgetting_sensitivity = float(cell.get("forgetting_sensitivity", 1.0))

        def factory(
            threshold: float,
            *,
            _drift: float = drift,
            _scale_s: float | None = scale_s,
            _scale_t: float | None = scale_t,
            _forgetting_sensitivity: float = forgetting_sensitivity,
        ) -> Any:
            return _pl_factory(
                threshold,
                drift=_drift,
                scale_s=_scale_s,
                scale_t=_scale_t,
                forgetting_sensitivity=_forgetting_sensitivity,
            )

        certificate, calibration = _calibrate(
            factory,
            calibration_source,
            family=f"persistent-laplacian-cusum-family:public-sensitivity:{cell['name']}:v1",
            candidates=(2.0, 8.0, 32.0, 128.0, 1024.0),
            calibration_seed=5103 + index * 101,
            evaluation_seed=6103 + index * 101,
        )
        entry: dict[str, Any] = {
            "name": cell["name"],
            "drift": drift,
            "scale_s": scale_s,
            "scale_t": scale_t,
            "forgetting_sensitivity": forgetting_sensitivity,
                "calibration": calibration,
                "status": "not_run" if certificate is None else "ran",
                "selected_threshold": calibration.get("selected_threshold"),
                "static_holdout_net_sharpe": _finite(
                _rolling_metrics(static, holdout_start)["net_sharpe"]
            ),
        }
        if certificate is None:
            rows.append(entry)
            continue
        result = _run_cell(
            features,
            outcomes,
            realized_returns,
            drift=drift,
            threshold=float(calibration["selected_threshold"]),
            scale_s=scale_s,
            scale_t=scale_t,
            forgetting_sensitivity=forgetting_sensitivity,
            certificate=certificate,
        )
        entry.update(
            {
                "calibration_identity": certificate.identity,
                "all_rows": {key: value for key, value in result.metrics.items()},
                "validation": _rolling_metrics(result, validation_start),
                "holdout": _rolling_metrics(result, holdout_start),
                "alarm_count": int(np.count_nonzero(result.alarms)),
                "authorized_acceleration_count": int(
                    np.count_nonzero(result.acceleration_authorized)
                ),
                "detector_telemetry": _detector_telemetry(result),
                "promotion": _promotion_audit(
                    static, result, validation_start, holdout_start
                ),
            }
        )
        rows.append(entry)

    summary: dict[str, Any] = {
        "kind": "public_market_pl_rls_sensitivity_diagnostic",
        "run_version": 1,
        "created_at_utc": dt.datetime.now(dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "source": {
            "provider": "Yahoo Finance chart endpoint",
            "final_adjusted_history": True,
            "point_in_time_universe_verified": False,
            "delisting_history_verified": False,
            "cache_dir": "<external-cache>",
        },
        "data": {
            "price_rows": int(prices.shape[0]),
            "feature_rows": int(features.shape[0]),
            "price_start": price_dates[0],
            "price_end": price_dates[-1],
            "calibration_rows": calibration_rows,
            "validation_start": validation_start,
            "holdout_start": holdout_start,
        },
        "cells": rows,
        "claim_status": "public-final-history sensitivity diagnostic only",
        "limitations": [
            "Cells are independent finite calibrations, not market-validity certificates.",
            "The source is final adjusted history and cannot certify point-in-time revisions.",
            "The fixed ETF universe does not prove delisting or historical-membership control.",
            "No live trading, execution, borrow, capacity, or venue-quality claim is made.",
        ],
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + os.linesep,
            encoding="utf-8",
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/public-market-sensitivity.json"),
    )
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    result = run(cache_dir=args.cache_dir, output=args.output, refresh=args.refresh)
    print(
        json.dumps(
            {
                "kind": result["kind"],
                "cells": [
                    {
                        "name": item["name"],
                        "status": item["status"],
                        "selected_threshold": item.get("selected_threshold"),
                        "holdout_net_sharpe": item.get("holdout", {}).get("net_sharpe"),
                        "score_p50": item.get("detector_telemetry", {}).get("score_p50"),
                        "forgetting_factor_p50": item.get("detector_telemetry", {}).get(
                            "forgetting_factor_p50"
                        ),
                    }
                    for item in result["cells"]
                ],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
