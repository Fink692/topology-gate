"""Run the complete control-layer experiment on a declared synthetic process.

This receipt exercises the original experiment matrix and requested metrics:
post-shift loss/regret, detector delay and recovery, IC, net Sharpe, drawdown,
turnover, transaction costs, and a finite promotion-null crossing rate. The
synthetic return path is deliberately marked as diagnostic and is never
treated as market evidence.

Run from the repository root with ``PYTHONPATH=src;examples`` so the script
can reuse the frozen detector-calibration helpers in
``synthetic_walk_forward_suite.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from synthetic_walk_forward_suite import (
    CHANGE_POINTS,
    SEEDS,
    RollingWindowRLS,
    _calibrate_cpd,
    _calibrate_detector,
    _cpd_factory,
    _detector_factory,
)

from topology_gate import (
    RLS,
    OnlineRunConfig,
    PromotionGate,
    RLSConfig,
    generate_synthetic_regimes,
    run_recursive_rls,
)
from topology_gate.backtest import calculate_metrics
from topology_gate.calibration import (
    PromotionCalibrationConfig,
    calibrate_promotion_null,
)

N_STEPS = 256
TRANSACTION_COST_BPS = 5.0
POST_SHIFT_START = CHANGE_POINTS[-1]


def _run_system(
    name: str,
    dataset: Any,
    realized_returns: np.ndarray[Any, Any],
    *,
    pl_certificate: Any,
    cpd_certificate: Any,
) -> Any:
    detector = None
    certificate = None
    if name == "static-rls":
        learner = RLS(RLSConfig(2, ridge=1.0, forgetting_factor=1.0, lambda_min=0.8, lambda_max=1.0))
    elif name == "rolling-rls":
        learner = RollingWindowRLS(window=32)
    elif name == "exponential-rls":
        learner = RLS(RLSConfig(2, ridge=1.0, forgetting_factor=0.97, lambda_min=0.8, lambda_max=0.97))
    elif name == "standard-cpd-rls":
        learner = RLS(RLSConfig(2, ridge=1.0, forgetting_factor=0.99, lambda_min=0.8, lambda_max=0.99))
        detector = _cpd_factory(float(cpd_certificate.selected_threshold))
        certificate = cpd_certificate.to_certificate()
    elif name == "certified-pl-rls":
        learner = RLS(RLSConfig(2, ridge=1.0, forgetting_factor=0.99, lambda_min=0.8, lambda_max=0.99))
        detector = _detector_factory(float(pl_certificate.selected_threshold))
        certificate = pl_certificate.to_certificate()
    else:
        raise ValueError(f"unknown system: {name}")
    return run_recursive_rls(
        dataset.features.values,
        dataset.labels.values,
        realized_returns=realized_returns,
        learner=learner,
        detector=detector,
        market_states=dataset.features.values,
        config=OnlineRunConfig(
            label_delay=1,
            transaction_cost_bps=TRANSACTION_COST_BPS,
            position_scale=1.0,
            position_limit=1.0,
            require_realized_returns=True,
        ),
        shift_points=tuple(dataset.change_points),
        calibration=certificate,
    )


def _metrics(
    result: Any,
    dataset: Any,
    realized_returns: np.ndarray[Any, Any],
    expected_returns: np.ndarray[Any, Any],
    *,
    start: int = 0,
) -> dict[str, Any]:
    n = len(realized_returns)
    evaluated = np.arange(n) >= start
    metrics = calculate_metrics(
        result.positions,
        realized_returns,
        predictions=result.predictions,
        labels=dataset.labels.values,
        expected_returns=expected_returns,
        transaction_costs=result.transaction_costs,
        comparator_transaction_costs=np.zeros(n, dtype=float),
        evaluated=evaluated,
        optimal_position=dataset.optimal_position,
        change_points=tuple(dataset.change_points),
    )
    return {
        "rows": int(metrics.n_evaluated),
        "mse": float(np.mean((result.predictions[start:] - dataset.labels.values[start:]) ** 2)),
        "post_shift_one_sided_utility_regret": metrics.one_sided_utility_regret,
        "mean_step_utility_regret": metrics.mean_step_utility_regret,
        "detection_delay": metrics.detection_delay,
        "recovery": metrics.recovery,
        "information_coefficient": metrics.information_coefficient,
        "hit_rate": metrics.hit_rate,
        "net_return": metrics.net_return,
        "net_sharpe": metrics.sharpe,
        "max_drawdown": metrics.max_drawdown,
        "turnover": metrics.turnover,
        "average_turnover": metrics.average_turnover,
        "transaction_cost": metrics.total_transaction_cost,
        "accelerated_forgetting_count": int(np.count_nonzero(result.acceleration_authorized)),
        "false_alarm_count": int(np.count_nonzero(result.alarms[: CHANGE_POINTS[0]])),
    }


def _promotion_diagnostic(static: Any, challenger: Any) -> dict[str, Any]:
    gate = PromotionGate("static-rls", alpha=0.05, eta=0.5, score_bound=0.01)
    gate.register_challenger("certified-pl-rls")
    gate.seal_registration()
    observations = 0
    for index in range(128, 192):
        decision = gate.observe_utilities(
            "certified-pl-rls",
            float(challenger.net_returns[index]),
            float(static.net_returns[index]),
            metadata={"row": index, "phase": "synthetic-validation"},
        )
        observations += 1
        if decision.promoted:
            break
    state = gate.challenger_state("certified-pl-rls")
    return {
        "observations": observations,
        "promoted": gate.promoted_challenger_id is not None,
        "promoted_challenger_id": gate.promoted_challenger_id,
        "e_value": state.e_value,
        "threshold": state.threshold,
        "registration_sealed": gate.registration_sealed,
    }


def _promotion_null() -> dict[str, Any]:
    def score_factory(rng: np.random.Generator, horizon: int, challengers: int) -> np.ndarray[Any, Any]:
        return rng.choice((-0.25, 0.25), size=(horizon, challengers))

    result = calibrate_promotion_null(
        score_factory,
        config=PromotionCalibrationConfig(
            trials=256,
            horizon=256,
            challengers=2,
            alpha=0.05,
            eta=0.5,
            seed=20260804,
        ),
    )
    return {
        "trials": result.trials,
        "horizon": result.horizon,
        "challengers": result.challenger_count,
        "crossing_count": result.threshold_crossing_count,
        "crossing_rate": result.threshold_crossing_rate,
        "crossing_ci_95": [result.threshold_crossing_ci_low, result.threshold_crossing_ci_high],
        "note": "finite synthetic null; conditional-mean market validity is not established",
    }


def run(output: Path | None) -> dict[str, Any]:
    pl_calibration = _calibrate_detector()
    cpd_calibration = _calibrate_cpd()
    rows: list[dict[str, Any]] = []
    promotion_example: dict[str, Any] | None = None
    for seed in SEEDS:
        dataset = generate_synthetic_regimes(
            n_steps=N_STEPS,
            n_features=2,
            change_points=CHANGE_POINTS,
            seed=seed,
            label_delay=1,
            feature_noise=0.20,
            return_noise=0.0,
            signal_strength=1.0,
            return_magnitude=0.0,
        )
        rng = np.random.default_rng(seed + 9000)
        expected_returns = dataset.optimal_position * 0.002
        realized_returns = expected_returns + rng.normal(0.0, 0.0005, size=N_STEPS)
        results: dict[str, Any] = {}
        for name in (
            "static-rls",
            "rolling-rls",
            "exponential-rls",
            "standard-cpd-rls",
            "certified-pl-rls",
        ):
            result = _run_system(
                name,
                dataset,
                realized_returns,
                pl_certificate=pl_calibration,
                cpd_certificate=cpd_calibration,
            )
            results[name] = result
            rows.append(
                {
                    "seed": seed,
                    "system": name,
                    "full_stream": _metrics(result, dataset, realized_returns, expected_returns),
                    "post_shift": _metrics(
                        result,
                        dataset,
                        realized_returns,
                        expected_returns,
                        start=POST_SHIFT_START,
                    ),
                }
            )
        if promotion_example is None:
            promotion_example = _promotion_diagnostic(
                results["static-rls"], results["certified-pl-rls"]
            )
    summary = {
        "kind": "integrated_synthetic_control_layer_study",
        "steps": N_STEPS,
        "change_points": list(CHANGE_POINTS),
        "seeds": list(SEEDS),
        "label_delay": 1,
        "transaction_cost_bps": TRANSACTION_COST_BPS,
        "systems": [
            "static-rls",
            "rolling-rls",
            "exponential-rls",
            "standard-cpd-rls",
            "certified-pl-rls",
        ],
        "pl_calibration": {
            "selected_threshold": pl_calibration.selected_threshold,
            "identity": pl_calibration.identity,
        },
        "cpd_calibration": {
            "selected_threshold": cpd_calibration.selected_threshold,
            "identity": cpd_calibration.identity,
        },
        "rows": rows,
        "promotion": promotion_example,
        "promotion_null": _promotion_null(),
        "claim_status": "synthetic control-layer diagnostic only; not market evidence",
        "dynamic_regret_status": "post-shift one-sided utility regret against the declared synthetic oracle; not conventional dynamic regret",
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/integrated-synthetic-study.json"),
    )
    args = parser.parse_args()
    print(json.dumps(run(args.output), sort_keys=True))


if __name__ == "__main__":
    main()
