"""Run the control-layer experiment on a reproducible public market proxy.

This is a diagnostic, not the study's licensed point-in-time market evidence.
Yahoo Finance's chart endpoint supplies final adjusted ETF history.  The
script records URL and payload hashes, uses a fixed ETF universe, performs
all feature normalization on the calibration prefix, and keeps the vendor
data gate explicitly unevaluated.

The experiment is intentionally small enough to run from a clean checkout:

    python examples/public_market_diagnostic.py --output reports/public-market-diagnostic.json

Raw responses are cached outside the repository by default.  They should not
be treated as a durable data archive or as proof of delisting/survivorship
control.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Callable

import numpy as np

from topology_gate import (
    RLS,
    MeanCovarianceCUSUM,
    MeanCovarianceCUSUMConfig,
    OnlineRunConfig,
    PromotionGate,
    RLSConfig,
    run_recursive_rls,
)
from topology_gate.calibration import (
    CalibrationConfig,
    StationaryBlockBootstrap,
    calibrate_threshold,
)
from topology_gate.persistent import (
    PersistentLaplacianBackend,
    PersistentLaplacianConfig,
)
from topology_gate.pl_cusum import PersistentCUSUMConfig, PersistentLaplacianCUSUM

TICKERS = ("SPY", "TLT", "EFA", "EEM", "IWM", "GLD")
START_DATE = "2007-01-01"
END_DATE = "2026-08-04"
SOURCE_URL_TEMPLATE = (
    "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    "?period1={period1}&period2={period2}&interval=1d&events=div%2Csplits"
    "&includeAdjustedClose=true"
)
FEATURE_NAMES = (
    *(f"ret_1d_{ticker}" for ticker in TICKERS),
    *(f"ret_5d_{ticker}" for ticker in TICKERS),
    *(f"vol_20d_{ticker}" for ticker in TICKERS),
)
FEATURE_COUNT = len(FEATURE_NAMES)


class RollingWindowRLS:
    """Fixed-window ridge learner used only as the declared baseline."""

    lambda_min = 1.0
    lambda_max = 1.0
    n_features = FEATURE_COUNT

    def __init__(self, window: int = 32, ridge: float = 1.0) -> None:
        self.window = window
        self.ridge = ridge
        self._rows: list[np.ndarray[Any, Any]] = []
        self._labels: list[float] = []
        self._theta = np.zeros(self.n_features, dtype=float)

    def reset(self) -> None:
        self._rows = []
        self._labels = []
        self._theta = np.zeros(self.n_features, dtype=float)

    def predict(self, features: Any) -> float:
        return float(np.asarray(features, dtype=float) @ self._theta)

    def update(self, features: Any, target: Any, forgetting_factor: Any = None) -> None:
        del forgetting_factor
        row = np.asarray(features, dtype=float).reshape(-1)
        self._rows.append(np.array(row, copy=True))
        self._labels.append(float(target))
        self._rows = self._rows[-self.window :]
        self._labels = self._labels[-self.window :]
        matrix = np.vstack(self._rows)
        gram = matrix.T @ matrix + self.ridge * np.eye(self.n_features)
        self._theta = np.linalg.solve(gram, matrix.T @ np.asarray(self._labels))


def _epoch(date_text: str) -> int:
    return int(dt.datetime.strptime(date_text, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc).timestamp())


def _json_hash(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _json_safe(value: Any) -> Any:
    """Map numerical non-finites to JSON null without changing finite values."""

    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.floating):
        converted = float(value)
        return converted if math.isfinite(converted) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _download_payload(
    ticker: str,
    *,
    cache_dir: Path,
    refresh: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{ticker.lower()}.json"
    period1 = _epoch(START_DATE)
    period2 = _epoch((dt.date.fromisoformat(END_DATE) + dt.timedelta(days=1)).isoformat())
    url = SOURCE_URL_TEMPLATE.format(ticker=ticker, period1=period1, period2=period2)
    if path.exists() and not refresh:
        raw = path.read_bytes()
        cache_status = "cache"
    else:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "topology-gate-research/0.1 public diagnostic"},
        )
        with urllib.request.urlopen(request, timeout=45) as response:
            raw = response.read()
        path.write_bytes(raw)
        cache_status = "download"
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{ticker}: public response is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{ticker}: public response root is not an object")
    return payload, {
        "ticker": ticker,
        "url": url,
        "sha256": _json_hash(raw),
        "bytes": len(raw),
        "cache_status": cache_status,
    }


def _extract_adjusted_close(payload: dict[str, Any], ticker: str) -> dict[str, float]:
    try:
        result = payload["chart"]["result"][0]
        timestamps = result["timestamp"]
        adjusted = result["indicators"]["adjclose"][0]["adjclose"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"{ticker}: adjusted-close series is missing") from exc
    if not isinstance(timestamps, list) or not isinstance(adjusted, list):
        raise RuntimeError(f"{ticker}: adjusted-close series has an invalid shape")
    if len(timestamps) != len(adjusted):
        raise RuntimeError(f"{ticker}: timestamps and adjusted-close lengths differ")
    values: dict[str, float] = {}
    for timestamp, value in zip(timestamps, adjusted):
        if value is None:
            continue
        try:
            number = float(value)
            date_text = dt.datetime.fromtimestamp(
                int(timestamp), tz=dt.timezone.utc
            ).date().isoformat()
        except (TypeError, ValueError, OverflowError, OSError) as exc:
            raise RuntimeError(f"{ticker}: malformed adjusted-close observation") from exc
        if not math.isfinite(number) or number <= 0.0:
            raise RuntimeError(f"{ticker}: adjusted-close observation is not positive")
        values[date_text] = number
    if len(values) < 100:
        raise RuntimeError(f"{ticker}: too few adjusted-close observations")
    return values


def load_prices(
    *, cache_dir: Path, refresh: bool = False
) -> tuple[np.ndarray[Any, Any], tuple[str, ...], list[dict[str, Any]]]:
    series: dict[str, dict[str, float]] = {}
    manifest: list[dict[str, Any]] = []
    for ticker in TICKERS:
        payload, record = _download_payload(ticker, cache_dir=cache_dir, refresh=refresh)
        series[ticker] = _extract_adjusted_close(payload, ticker)
        manifest.append(record)
    common_dates = sorted(set.intersection(*(set(series[ticker]) for ticker in TICKERS)))
    if len(common_dates) < 500:
        raise RuntimeError("public source has fewer than 500 common trading dates")
    prices = np.asarray(
        [[series[ticker][date] for ticker in TICKERS] for date in common_dates],
        dtype=float,
    )
    if not np.all(np.isfinite(prices)) or np.any(prices <= 0.0):
        raise RuntimeError("public price panel is not finite and positive")
    return prices, tuple(common_dates), manifest


def build_features(
    prices: np.ndarray[Any, Any], dates: tuple[str, ...]
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any], tuple[str, ...]]:
    returns = prices[1:] / prices[:-1] - 1.0
    return_dates = dates[1:]
    rows: list[np.ndarray[Any, Any]] = []
    labels: list[float] = []
    realized: list[float] = []
    row_dates: list[str] = []
    for index in range(20, len(returns) - 1):
        one_day = returns[index]
        five_day = prices[index + 1] / prices[index - 3] - 1.0
        volatility = np.std(returns[index - 19 : index + 1], axis=0, ddof=1)
        next_return = float(np.mean(returns[index + 1]))
        scale = max(float(np.mean(volatility)), 1.0e-6)
        row = np.concatenate((one_day, five_day, volatility))
        rows.append(row)
        labels.append(next_return / scale)
        realized.append(next_return)
        row_dates.append(return_dates[index])
    features = np.asarray(rows, dtype=float)
    outcomes = np.asarray(labels, dtype=float)
    realized_returns = np.asarray(realized, dtype=float)
    if features.ndim != 2 or features.shape[1] != FEATURE_COUNT:
        raise RuntimeError("feature construction produced the wrong width")
    if not np.all(np.isfinite(features)) or not np.all(np.isfinite(outcomes)):
        raise RuntimeError("feature construction produced non-finite values")
    return features, outcomes, realized_returns, tuple(row_dates)


def normalize_features(
    features: np.ndarray[Any, Any], calibration_rows: int
) -> tuple[np.ndarray[Any, Any], dict[str, list[float]]]:
    center = np.mean(features[:calibration_rows], axis=0)
    scale = np.std(features[:calibration_rows], axis=0, ddof=1)
    scale = np.maximum(scale, 1.0e-8)
    normalized = (features - center) / scale
    if not np.all(np.isfinite(normalized)):
        raise RuntimeError("causal feature normalization produced non-finite values")
    return normalized, {"center": center.tolist(), "scale": scale.tolist()}


def _pl_factory(threshold: float) -> PersistentLaplacianCUSUM:
    backend = PersistentLaplacianBackend(
        PersistentLaplacianConfig(
            max_vertices=4,
            max_simplices=100,
            q=0,
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
            threshold=threshold,
        ),
    )


def _cpd_factory(threshold: float) -> MeanCovarianceCUSUM:
    return MeanCovarianceCUSUM(
        MeanCovarianceCUSUMConfig(
            n_features=FEATURE_COUNT,
            block_window=16,
            threshold=threshold,
            forgetting_lambda_min=0.8,
            forgetting_lambda_max=0.99,
            forgetting_sensitivity=1.0,
        )
    )


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
        source, block_length=16, source_id="yahoo-final-adjusted:calibration-prefix:v1"
    )
    evaluation_factory = StationaryBlockBootstrap(
        source, block_length=16, source_id="yahoo-final-adjusted:calibration-prefix:v1"
    )
    try:
        result = calibrate_threshold(
            detector_factory,
            calibration_factory,
            evaluation_factory,
            detector_family_identity=family,
            candidate_thresholds=candidates,
            calibration_config=CalibrationConfig(
                trials=32, horizon=48, n_features=FEATURE_COUNT, seed=calibration_seed
            ),
            evaluation_config=CalibrationConfig(
                trials=32, horizon=48, n_features=FEATURE_COUNT, seed=evaluation_seed
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


def _rolling_metrics(result: Any, start: int) -> dict[str, float]:
    net = result.net_returns[start:]
    predictions = result.predictions[start:]
    outcomes = result.outcomes[start:]
    volatility = float(np.std(net, ddof=1)) if len(net) > 1 else 0.0
    return {
        "rows": float(len(net)),
        "mse": float(np.mean((predictions - outcomes) ** 2)),
        "mean_net_return": float(np.mean(net)),
        "net_sharpe": 0.0 if volatility <= 1.0e-15 else float(np.mean(net) / volatility * math.sqrt(252.0)),
        "max_drawdown": float(_max_drawdown(net)),
        "turnover": float(np.mean(np.abs(np.diff(np.r_[0.0, result.positions[start:]])))),
    }


def _max_drawdown(returns: np.ndarray[Any, Any]) -> float:
    equity = 1.0 + np.cumsum(returns)
    peak = np.maximum.accumulate(equity)
    return float(np.max(np.where(peak > 0.0, (peak - equity) / peak, 0.0)))


def _detector_telemetry(result: Any) -> dict[str, float | int | None]:
    """Summarize the control signal so over-forgetting is observable."""

    scores = np.asarray(result.detector_scores, dtype=float)
    factors = np.asarray(result.forgetting_factors, dtype=float)
    authorized = np.asarray(result.acceleration_authorized, dtype=bool)
    finite_scores = scores[np.isfinite(scores)]
    finite_factors = factors[np.isfinite(factors)]

    def _quantile(values: np.ndarray[Any, Any], level: float) -> float | None:
        return None if values.size == 0 else float(np.quantile(values, level))

    return {
        "score_positive_fraction": float(np.mean(scores > 0.0)),
        "score_p50": _quantile(finite_scores, 0.50),
        "score_p90": _quantile(finite_scores, 0.90),
        "score_p99": _quantile(finite_scores, 0.99),
        "score_max": None if finite_scores.size == 0 else float(np.max(finite_scores)),
        "forgetting_factor_p01": _quantile(finite_factors, 0.01),
        "forgetting_factor_p50": _quantile(finite_factors, 0.50),
        "forgetting_factor_p99": _quantile(finite_factors, 0.99),
        "forgetting_factor_min": (
            None if finite_factors.size == 0 else float(np.min(finite_factors))
        ),
        "forgetting_factor_max": (
            None if finite_factors.size == 0 else float(np.max(finite_factors))
        ),
        "ready_or_authorized_rows": int(np.count_nonzero(authorized)),
        "ready_or_authorized_fraction": float(np.mean(authorized)),
    }


def _promotion_state_summary(gate: PromotionGate) -> dict[str, Any]:
    """Keep terminal gate facts and digest the verbose row-level audit."""

    raw = gate.state_dict()
    audit_records = raw.pop("audit_records")
    audit_digest = hashlib.sha256(
        json.dumps(
            audit_records,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    summary = {
        key: raw[key]
        for key in (
            "version",
            "schema",
            "incumbent_id",
            "global_alpha",
            "score_bound",
            "initial_wealth",
            "epoch",
            "status",
            "promoted_challenger_id",
            "allocated_alpha",
            "registration_sealed",
        )
    }
    compact_challengers: list[dict[str, Any]] = []
    for registered in raw["challengers"]:
        machine = registered["state"]
        process = machine["process"]
        compact_challengers.append(
            {
                "index": registered["index"],
                "challenger_id": machine["challenger_id"],
                "state": machine["state"],
                "epoch": process["epoch"],
                "alpha": process["alpha"],
                "wealth": process["wealth"],
                "observation_count": process["observation_count"],
                "ever_crossed": process["ever_crossed"],
                "first_crossing_observation": process["first_crossing_observation"],
            }
        )
    summary["challengers"] = compact_challengers
    summary["audit_record_count"] = len(audit_records)
    summary["audit_records_sha256"] = audit_digest
    return summary


def _run_system(
    name: str,
    features: np.ndarray[Any, Any],
    outcomes: np.ndarray[Any, Any],
    realized_returns: np.ndarray[Any, Any],
    *,
    pl_threshold: float | None,
    pl_certificate: Any | None,
    cpd_threshold: float | None,
    cpd_certificate: Any | None,
) -> Any:
    detector = None
    certificate = None
    if name == "static-rls":
        learner = RLS(RLSConfig(n_features=FEATURE_COUNT, ridge=1.0, forgetting_factor=1.0, lambda_min=0.8, lambda_max=1.0))
    elif name == "rolling-rls":
        learner = RollingWindowRLS(window=32)
    elif name == "exponential-rls":
        learner = RLS(RLSConfig(n_features=FEATURE_COUNT, ridge=1.0, forgetting_factor=0.97, lambda_min=0.8, lambda_max=0.97))
    elif name == "standard-cpd-rls":
        if cpd_threshold is None:
            raise RuntimeError("standard CPD threshold was not calibrated")
        learner = RLS(RLSConfig(n_features=FEATURE_COUNT, ridge=1.0, forgetting_factor=0.99, lambda_min=0.8, lambda_max=0.99))
        detector = _cpd_factory(cpd_threshold)
        certificate = cpd_certificate
    elif name == "certified-pl-rls":
        if pl_threshold is None:
            raise RuntimeError("persistent threshold was not calibrated")
        learner = RLS(RLSConfig(n_features=FEATURE_COUNT, ridge=1.0, forgetting_factor=0.99, lambda_min=0.8, lambda_max=0.99))
        detector = _pl_factory(pl_threshold)
        certificate = pl_certificate
    else:
        raise ValueError(f"unknown system: {name}")
    return run_recursive_rls(
        features,
        outcomes,
        realized_returns=realized_returns,
        learner=learner,
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


def _promotion_audit(static: Any, pl: Any, validation_start: int, holdout_start: int) -> dict[str, Any]:
    gate = PromotionGate("static-rls", alpha=0.05, eta=0.5, score_bound=0.02)
    gate.register_challenger("certified-pl-rls")
    gate.seal_registration()
    observations = 0
    first_crossing: int | None = None
    for index in range(validation_start, holdout_start):
        try:
            decision = gate.observe_utilities(
                "certified-pl-rls",
                float(pl.net_returns[index]),
                float(static.net_returns[index]),
                metadata={"row": index, "phase": "public-diagnostic-validation"},
            )
        except ValueError as exc:
            return {
                "status": "not_certified",
                "error": f"{type(exc).__name__}: {exc}",
                "observations": observations,
                "state": _promotion_state_summary(gate),
            }
        observations += 1
        if decision.promoted:
            first_crossing = index
            break
    challenger = gate.challenger_state("certified-pl-rls")
    return {
        "status": "diagnostic_only",
        "observations": observations,
        "validation_start": validation_start,
        "holdout_start": holdout_start,
        "first_crossing_row": first_crossing,
        "promoted_challenger_id": gate.promoted_challenger_id,
        "challenger_e_value": challenger.e_value,
        "challenger_threshold": challenger.threshold,
        "registration_sealed": gate.registration_sealed,
        "state": _promotion_state_summary(gate),
        "reason": "final-adjusted public history is not a point-in-time market evidence package",
    }


def run(*, cache_dir: Path, output: Path | None, refresh: bool) -> dict[str, Any]:
    prices, price_dates, source_manifest = load_prices(cache_dir=cache_dir, refresh=refresh)
    raw_features, outcomes, realized_returns, row_dates = build_features(prices, price_dates)
    calibration_rows = max(256, int(len(raw_features) * 0.40))
    if calibration_rows >= len(raw_features) - 64:
        raise RuntimeError("public history is too short for calibration and holdout splits")
    features, normalization = normalize_features(raw_features, calibration_rows)
    holdout_start = int(len(features) * 0.85)
    validation_start = calibration_rows
    calibration_source = features[:calibration_rows]
    pl_certificate, pl_calibration = _calibrate(
        _pl_factory,
        calibration_source,
        family="persistent-laplacian-cusum-family:public-diagnostic:v1",
        candidates=(2.0, 8.0, 32.0, 128.0, 1024.0),
        calibration_seed=1103,
        evaluation_seed=2903,
    )
    cpd_certificate, cpd_calibration = _calibrate(
        _cpd_factory,
        calibration_source,
        family="mean-covariance-cusum-family:public-diagnostic:v1",
        candidates=(2.0, 4.0, 8.0, 16.0, 32.0),
        calibration_seed=1307,
        evaluation_seed=3707,
    )
    pl_threshold = pl_calibration.get("selected_threshold") if pl_certificate is not None else None
    cpd_threshold = cpd_calibration.get("selected_threshold")
    if cpd_threshold is None:
        cpd_threshold = cpd_calibration.get("diagnostic_threshold")
    systems: dict[str, Any] = {}
    for name in ("static-rls", "rolling-rls", "exponential-rls", "standard-cpd-rls", "certified-pl-rls"):
        if name == "certified-pl-rls" and pl_threshold is None:
            systems[name] = {"status": "not_run", "reason": "PL calibration was not approved"}
            continue
        result = _run_system(
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
            "status": (
                "ran"
                if name not in {"standard-cpd-rls", "certified-pl-rls"}
                or (cpd_certificate if name == "standard-cpd-rls" else pl_certificate) is not None
                else "ran_unapproved_calibration_diagnostic"
            ),
            "detector_threshold": (
                cpd_threshold if name == "standard-cpd-rls" else pl_threshold
                if name == "certified-pl-rls"
                else None
            ),
            "all_rows": {key: value for key, value in result.metrics.items()},
            "validation": _rolling_metrics(result, validation_start),
            "holdout": _rolling_metrics(result, holdout_start),
            "calibration_identity": result.calibration_identity,
            "accelerated_forgetting_count": int(result.metrics["accelerated_forgetting_count"]),
            "alarm_count": int(np.count_nonzero(result.alarms)),
            "authorized_acceleration_count": int(np.count_nonzero(result.acceleration_authorized)),
            "detector_telemetry": _detector_telemetry(result),
        }
        systems[name]["_result"] = result
    static_result = systems["static-rls"].pop("_result")
    pl_entry = systems.get("certified-pl-rls", {})
    pl_result = pl_entry.pop("_result", None)
    promotion = (
        _promotion_audit(static_result, pl_result, validation_start, holdout_start)
        if pl_result is not None
        else {"status": "not_run", "reason": "PL calibration was not approved"}
    )
    for entry in systems.values():
        if isinstance(entry, dict):
            entry.pop("_result", None)
    summary: dict[str, Any] = {
        "kind": "public_market_control_layer_diagnostic",
        "run_version": 1,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "source": {
            "provider": "Yahoo Finance chart endpoint",
            "endpoint_template": SOURCE_URL_TEMPLATE,
            "start_date_requested": START_DATE,
            "end_date_requested": END_DATE,
            "tickers": list(TICKERS),
            "adjusted_close": True,
            "final_adjusted_history": True,
            "point_in_time_universe_verified": False,
            "delisting_history_verified": False,
            "manifest": source_manifest,
            "cache_dir": "<external-cache>",
        },
        "data": {
            "price_rows": int(prices.shape[0]),
            "feature_rows": int(features.shape[0]),
            "price_start": price_dates[0],
            "price_end": price_dates[-1],
            "feature_start": row_dates[0],
            "feature_end": row_dates[-1],
            "feature_names": list(FEATURE_NAMES),
            "feature_count": FEATURE_COUNT,
            "target": "next-day equal-weight ETF return divided by current mean 20-day volatility",
            "realized_return": "next-day equal-weight raw ETF return",
            "label_delay": 1,
            "normalization_fit_rows": calibration_rows,
            "normalization": normalization,
        },
        "splits": {
            "calibration_rows": calibration_rows,
            "validation_start": validation_start,
            "holdout_start": holdout_start,
            "holdout_fraction": 0.15,
            "holdout_is_final_public_history_diagnostic": True,
        },
        "calibration": {"persistent_laplacian": pl_calibration, "mean_covariance": cpd_calibration},
        "systems": systems,
        "promotion": promotion,
        "claim_status": "public-final-history diagnostic only",
        "vendor_gate_status": "not_evaluated",
        "limitations": [
            "The source is final adjusted history and cannot certify point-in-time revisions.",
            "The fixed ETF universe is not a historical security master and does not prove delisting control.",
            "Threshold calibration and e-process observations are diagnostic; they do not upgrade this source into licensed market evidence.",
            "No live trading, execution, borrow, capacity, or venue-quality claim is made.",
        ],
    }
    summary = _json_safe(summary)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + os.linesep, encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path(tempfile.gettempdir()) / "topology-gate-public-market-diagnostic")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    summary = run(cache_dir=args.cache_dir, output=args.output, refresh=args.refresh)
    print(json.dumps({
        "kind": summary["kind"],
        "price_rows": summary["data"]["price_rows"],
        "feature_rows": summary["data"]["feature_rows"],
        "price_range": [summary["data"]["price_start"], summary["data"]["price_end"]],
        "calibration_approved": {
            "persistent_laplacian": summary["calibration"]["persistent_laplacian"].get("approved", False),
            "mean_covariance": summary["calibration"]["mean_covariance"].get("approved", False),
        },
        "claim_status": summary["claim_status"],
        "output": str(args.output) if args.output is not None else None,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
