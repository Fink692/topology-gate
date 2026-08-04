"""Evaluate calibration-only normalization of the PL forgetting score."""

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

SCALE_S = 2.0
SCALE_T = 20.0
SCORE_QUANTILE = 0.50


def _pl_factory(
    threshold: float,
    *,
    score_scale: float,
) -> PersistentLaplacianCUSUM:
    backend = PersistentLaplacianBackend(
        PersistentLaplacianConfig(
            max_vertices=4,
            max_simplices=100,
            q=0,
            scale_s=SCALE_S,
            scale_t=SCALE_T,
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
            drift=0.5,
            forgetting_lambda_min=0.8,
            forgetting_lambda_max=0.99,
            forgetting_sensitivity=1.0,
            forgetting_score_scale=score_scale,
            threshold=threshold,
        ),
    )


def _derive_score_scale(source: np.ndarray[Any, Any]) -> float:
    probe = _pl_factory(1.0e12, score_scale=1.0)
    observations = probe.detect(source).observations
    scores = np.asarray(
        [item.score for item in observations if item.ready and item.score > 0.0],
        dtype=float,
    )
    if scores.size == 0 or not np.all(np.isfinite(scores)):
        raise RuntimeError("calibration prefix produced no positive finite PL scores")
    scale = float(np.quantile(scores, SCORE_QUANTILE))
    if not math.isfinite(scale) or scale <= 0.0:
        raise RuntimeError("derived forgetting score scale is not positive")
    return scale


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


def run(*, cache_dir: Path, output: Path | None, refresh: bool) -> dict[str, Any]:
    prices, price_dates, _ = load_prices(cache_dir=cache_dir, refresh=refresh)
    raw_features, outcomes, realized_returns, _ = build_features(prices, price_dates)
    calibration_rows = max(256, int(len(raw_features) * 0.40))
    features, _ = normalize_features(raw_features, calibration_rows)
    validation_start = calibration_rows
    holdout_start = int(len(features) * 0.85)
    calibration_source = features[:calibration_rows]
    score_scale = _derive_score_scale(calibration_source)

    def factory(threshold: float) -> Any:
        return _pl_factory(threshold, score_scale=score_scale)

    certificate, calibration = _calibrate(
        factory,
        calibration_source,
        family="persistent-laplacian-cusum-family:public-score-normalized:v1",
        candidates=(2.0, 8.0, 32.0, 128.0, 1024.0),
        calibration_seed=7103,
        evaluation_seed=8103,
    )
    if certificate is None:
        raise RuntimeError("score-normalized PL calibration was not approved")
    static = _run_static(features, outcomes, realized_returns)
    adaptive = run_recursive_rls(
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
        detector=_pl_factory(
            float(calibration["selected_threshold"]), score_scale=score_scale
        ),
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
    static_holdout = _rolling_metrics(static, holdout_start)
    adaptive_holdout = _rolling_metrics(adaptive, holdout_start)
    summary: dict[str, Any] = {
        "kind": "public_market_pl_rls_score_normalization_diagnostic",
        "run_version": 1,
        "created_at_utc": dt.datetime.now(dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "source": {
            "provider": "Yahoo Finance chart endpoint",
            "final_adjusted_history": True,
            "point_in_time_universe_verified": False,
            "delisting_history_verified": False,
            "cache_dir": str(cache_dir),
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
        "normalization": {
            "scale_s": SCALE_S,
            "scale_t": SCALE_T,
            "score_quantile": SCORE_QUANTILE,
            "score_scale": score_scale,
            "fit_source": "calibration prefix only",
        },
        "calibration": calibration,
        "static_holdout": static_holdout,
        "adaptive_holdout": adaptive_holdout,
        "adaptive_all_rows": {key: value for key, value in adaptive.metrics.items()},
        "adaptive_telemetry": _detector_telemetry(adaptive),
        "promotion": _promotion_audit(
            static, adaptive, validation_start, holdout_start
        ),
        "claim_status": "public-final-history score-normalization diagnostic only",
        "limitations": [
            "The score scale is fit on the calibration prefix, not the holdout.",
            "The fixed filtration scales remain an exploratory public-proxy choice.",
            "The source is final adjusted history and cannot certify point-in-time revisions.",
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
        default=Path("reports/public-market-score-normalization.json"),
    )
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    result = run(cache_dir=args.cache_dir, output=args.output, refresh=args.refresh)
    print(
        json.dumps(
            {
                "kind": result["kind"],
                "score_scale": result["normalization"]["score_scale"],
                "selected_threshold": result["calibration"]["selected_threshold"],
                "static_holdout_net_sharpe": result["static_holdout"]["net_sharpe"],
                "adaptive_holdout_net_sharpe": result["adaptive_holdout"]["net_sharpe"],
                "adaptive_e_value": result["promotion"]["challenger_e_value"],
                "promoted": result["promotion"]["promoted_challenger_id"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
