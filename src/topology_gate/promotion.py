"""Anytime-valid challenger promotion.

This module deliberately keeps the statistical contract small and explicit.
For a challenger utility ``u_c`` and incumbent utility ``u_i`` we use the
normalized, clipped score

    ``X_t = clip(u_c - u_i, -B, B) / B``.

Consequently ``X_t`` is always in ``[-1, 1]``.  The null tested by this module
is the *conditional* null ``E[X_t | F_{t-1}] <= 0``.  With a predictable
fraction ``eta_t`` in ``[0, 1]``,

    ``E_t = product_s (1 + eta_s X_s)``

is a nonnegative e-process under that null.  A threshold crossing is therefore
valid at an optional stopping time.  It is not a raw-return claim about an
unclipped utility difference, and it does not establish a claim for a different
score or a different null.

The public classes are intentionally usable without a framework:

* :class:`EProcess` tracks one e-process and its audit trail.
* :class:`PromotionStateMachine` adds the active/promoted state transition for
  one challenger.
* :class:`PromotionGate` controls several challengers with a geometric,
  preallocated alpha budget across challenger slots and epochs.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import (
    Any,
    Callable,
    Iterable,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

DEFAULT_SCORE_BOUND = 1.0
DEFAULT_ALPHA = 0.05
DEFAULT_ETA = 0.5
NULL_HYPOTHESIS = "E[bounded_score_t | F_(t-1)] <= 0"
MAX_PROMOTION_HISTORY = 100_000
MAX_PROMOTION_AUDIT_RECORDS = 200_000
_SECRET_METADATA_KEY = re.compile(
    r"(?:pass(word|wd)?|secret|token|api[_-]?key|credential|authorization|"
    r"private[_-]?key|access[_-]?token|refresh[_-]?token)",
    re.IGNORECASE,
)

Number = Union[int, float]
EtaRule = Union[Number, Callable[[Tuple[float, ...]], Number]]


class PromotionError(ValueError):
    """Base error for invalid promotion configuration or input."""


class InvalidEtaError(PromotionError):
    """Raised when a betting fraction is not predictable and safe."""


class PromotionClosedError(PromotionError):
    """Raised when a multi-challenger gate has already promoted a candidate."""


class PromotionStatus(str, Enum):
    """State of a single challenger or of the gate."""

    ACTIVE = "active"
    PROMOTED = "promoted"
    RETIRED = "retired"


# ``PromotionState`` is a convenient spelling for callers that use the state
# as an enum.  Keep it as an alias rather than a second, subtly different enum.
PromotionState = PromotionStatus


def _finite_number(value: Number, name: str) -> float:
    """Convert a scalar to a finite float, rejecting booleans and NaN."""

    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _sanitize_metadata(value: Any, *, key: str | None = None, depth: int = 0) -> Any:
    if depth > 16:
        raise PromotionError("promotion metadata is nested too deeply")
    if key is not None and _SECRET_METADATA_KEY.search(key):
        return "[REDACTED]"
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise PromotionError("promotion metadata contains a non-finite value")
        return value
    if isinstance(value, Mapping):
        if len(value) > 4096:
            raise PromotionError("promotion metadata exceeds the item limit")
        return {
            str(item_key): _sanitize_metadata(item, key=str(item_key), depth=depth + 1)
            for item_key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        if len(value) > 4096:
            raise PromotionError("promotion metadata exceeds the item limit")
        return [_sanitize_metadata(item, depth=depth + 1) for item in value]
    raise PromotionError(
        f"promotion metadata value {type(value).__name__} is not JSON-safe"
    )


def validate_score_bound(bound: Number) -> float:
    """Validate the positive absolute bound used for utility differences."""

    result = _finite_number(bound, "score bound")
    if result <= 0:
        raise ValueError("score bound must be greater than zero")
    return result


def validate_alpha(alpha: Number) -> float:
    """Validate a significance level in the open interval ``(0, 1)``."""

    result = _finite_number(alpha, "alpha")
    if not 0.0 < result < 1.0:
        raise ValueError("alpha must be strictly between zero and one")
    return result


def validate_eta(eta: Number) -> float:
    """Validate a betting fraction for a score in ``[-1, 1]``.

    Requiring ``0 <= eta <= 1`` makes ``1 + eta * score`` nonnegative for
    every admissible score and makes a positive conditional score the only
    source of conditional e-value growth under the stated null.
    """

    try:
        result = _finite_number(eta, "eta")
    except ValueError as exc:
        raise InvalidEtaError(str(exc)) from exc
    if not 0.0 <= result <= 1.0:
        raise InvalidEtaError("eta must be finite and in the interval [0, 1]")
    return result


def _validate_utility(value: Number, name: str) -> float:
    return _finite_number(value, name)


def clip_utility_difference(
    challenger_utility: Number,
    incumbent_utility: Number,
    *,
    bound: Number = DEFAULT_SCORE_BOUND,
) -> float:
    """Return the raw utility difference clipped to ``[-bound, bound]``.

    The clipped value is an intermediate quantity.  Promotion evidence uses
    :func:`bounded_utility_difference`, which normalizes it to ``[-1, 1]``.
    """

    bound_value = validate_score_bound(bound)
    challenger = _validate_utility(challenger_utility, "challenger utility")
    incumbent = _validate_utility(incumbent_utility, "incumbent utility")
    difference = challenger - incumbent
    if not math.isfinite(difference):
        # Finite utilities can have an overflowing difference.  Its sign is
        # still enough to determine which clipping endpoint applies.
        return bound_value if challenger > incumbent else -bound_value
    return max(-bound_value, min(bound_value, difference))


def bounded_utility_difference(
    challenger_utility: Number,
    incumbent_utility: Number,
    *,
    bound: Number = DEFAULT_SCORE_BOUND,
) -> float:
    """Return the normalized bounded utility-difference score.

    ``bound`` is the absolute difference scale.  Differences beyond the scale
    are clipped before normalization, so the result is always in ``[-1, 1]``.
    This function does not return or infer a raw-return improvement.
    """

    bound_value = validate_score_bound(bound)
    clipped = clip_utility_difference(
        challenger_utility,
        incumbent_utility,
        bound=bound_value,
    )
    score = clipped / bound_value
    # Protect the invariant from a possible last-bit rounding artifact when a
    # non-unit bound is used.
    return max(-1.0, min(1.0, score))


def bounded_score(
    challenger_utility: Number,
    incumbent_utility: Number,
    *,
    bound: Number = DEFAULT_SCORE_BOUND,
) -> float:
    """Alias for :func:`bounded_utility_difference`."""

    return bounded_utility_difference(
        challenger_utility,
        incumbent_utility,
        bound=bound,
    )


def _validate_score(score: Number) -> float:
    result = _finite_number(score, "score")
    if result < -1.0 or result > 1.0:
        raise ValueError("score must be in the interval [-1, 1]")
    # Keep exact endpoint values exact for deterministic factor calculations.
    if result == -1.0:
        return -1.0
    if result == 1.0:
        return 1.0
    return result


def _validate_history(history: Iterable[Number]) -> Tuple[float, ...]:
    try:
        values = tuple(_validate_score(value) for value in history)
    except TypeError as exc:
        raise ValueError("history must be an iterable of bounded scores") from exc
    return values


def predictable_betting_fraction(
    history: Sequence[Number],
    *,
    max_eta: Number = 1.0,
    scale: Number = 1.0,
) -> float:
    """Choose a deterministic, history-measurable betting fraction.

    Only scores strictly before the next observation belong in ``history``.
    The rule bets in the positive direction after a positive running mean and
    never bets negatively:

    ``eta_t = max_eta * clamp(mean(history) / scale, 0, 1)``.

    With no history the fraction is zero.  The result is in ``[0, 1]`` and is
    therefore safe for every score in ``[-1, 1]``.  A caller may instead pass a
    constant or history-based rule directly as ``eta`` to :class:`EProcess`.
    """

    values = _validate_history(history)
    maximum = validate_eta(max_eta)
    scale_value = _finite_number(scale, "betting scale")
    if scale_value <= 0:
        raise ValueError("betting scale must be greater than zero")
    if not values:
        return 0.0
    mean = math.fsum(values) / len(values)
    direction = max(0.0, min(1.0, mean / scale_value))
    return validate_eta(maximum * direction)


def resolve_betting_fraction(eta: EtaRule, history: Sequence[Number]) -> float:
    """Resolve a constant or history-measurable eta before an observation."""

    prior_scores = _validate_history(history)
    if callable(eta):
        try:
            candidate = eta(prior_scores)
        except Exception as exc:  # expose a stable configuration error
            raise InvalidEtaError("eta rule failed before observing the score") from exc
    else:
        candidate = eta
    return validate_eta(candidate)


def optional_stopping_threshold(
    alpha: Number,
    *,
    initial_wealth: Number = 1.0,
) -> float:
    """Return the e-value threshold ``initial_wealth / alpha``."""

    alpha_value = validate_alpha(alpha)
    wealth = _finite_number(initial_wealth, "initial wealth")
    if wealth <= 0:
        raise ValueError("initial wealth must be greater than zero")
    return wealth / alpha_value


def optional_stopping_threshold_reached(
    e_value: Number,
    alpha: Number,
    *,
    initial_wealth: Number = 1.0,
) -> bool:
    """Check whether an e-value crosses its optional-stopping threshold."""

    if isinstance(e_value, bool):
        raise ValueError("e-value must be a nonnegative real number")
    try:
        value = float(e_value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("e-value must be a nonnegative real number") from exc
    if math.isnan(value) or value < 0:
        raise ValueError("e-value must be a nonnegative real number")
    return value >= optional_stopping_threshold(alpha, initial_wealth=initial_wealth)


def check_optional_stopping_threshold(
    e_value: Number,
    alpha: Number,
    *,
    initial_wealth: Number = 1.0,
) -> bool:
    """Alias for :func:`optional_stopping_threshold_reached`."""

    return optional_stopping_threshold_reached(
        e_value,
        alpha,
        initial_wealth=initial_wealth,
    )


def geometric_alpha_allocation(
    global_alpha: Number,
    challenger_index: int,
    *,
    epoch: int = 0,
) -> float:
    """Allocate alpha to a challenger slot and epoch.

    Indices are one-based.  The allocation is

    ``global_alpha * 2**(-challenger_index) * 2**(-(epoch + 1))``.

    Summing over all positive challenger indices and all nonnegative epochs is
    at most ``global_alpha``.  This is a preallocation rule, not a claim that
    a later challenger may borrow unused evidence from an earlier one.
    """

    alpha_value = validate_alpha(global_alpha)
    if isinstance(challenger_index, bool) or not isinstance(challenger_index, int):
        raise ValueError("challenger index must be a positive integer")
    if challenger_index < 1:
        raise ValueError("challenger index must be a positive integer")
    if isinstance(epoch, bool) or not isinstance(epoch, int):
        raise ValueError("epoch must be a nonnegative integer")
    if epoch < 0:
        raise ValueError("epoch must be a nonnegative integer")
    return alpha_value * (2.0 ** (-challenger_index)) * (2.0 ** (-(epoch + 1)))


def alpha_for_challenger(
    global_alpha: Number,
    challenger_index: int,
    *,
    epoch: int = 0,
) -> float:
    """Alias for :func:`geometric_alpha_allocation`."""

    return geometric_alpha_allocation(
        global_alpha,
        challenger_index,
        epoch=epoch,
    )


@dataclass(frozen=True)
class AuditRecord:
    """Immutable record of an e-process or promotion event.

    ``unclipped_difference`` is retained only for reproducibility and is never
    used as the betting score.  ``score`` is the bounded quantity that carries
    the statistical meaning.
    """

    event: str
    epoch: int
    observation: Optional[int]
    challenger_id: Optional[str]
    score: Optional[float]
    unclipped_difference: Optional[float]
    eta: Optional[float]
    factor: Optional[float]
    wealth_before: float
    wealth_after: float
    alpha: float
    threshold: float
    state_before: str
    state_after: str
    threshold_crossed: bool
    reason: Optional[str] = None
    metadata: Optional[Mapping[str, Any]] = None

    def __post_init__(self) -> None:
        if not self.event or self.epoch < 0:
            raise PromotionError("audit record needs a non-empty event and non-negative epoch")
        if self.metadata is not None:
            sanitized = _sanitize_metadata(self.metadata)
            if not isinstance(sanitized, Mapping):  # pragma: no cover - invariant
                raise PromotionError("audit metadata must be a mapping")
            object.__setattr__(self, "metadata", dict(sanitized))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible record for an authenticated checkpoint."""

        return {
            "event": self.event,
            "epoch": self.epoch,
            "observation": self.observation,
            "challenger_id": self.challenger_id,
            "score": self.score,
            "unclipped_difference": self.unclipped_difference,
            "eta": self.eta,
            "factor": self.factor,
            "wealth_before": self.wealth_before,
            "wealth_after": self.wealth_after,
            "alpha": self.alpha,
            "threshold": self.threshold,
            "state_before": self.state_before,
            "state_after": self.state_after,
            "threshold_crossed": self.threshold_crossed,
            "reason": self.reason,
            "metadata": dict(self.metadata) if self.metadata is not None else None,
        }

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> "AuditRecord":
        if not isinstance(state, Mapping):
            raise PromotionError("audit record must be a mapping")
        return cls(
            event=str(state.get("event", "")),
            epoch=int(state.get("epoch", 0)),
            observation=(
                None if state.get("observation") is None else int(state["observation"])
            ),
            challenger_id=(
                None
                if state.get("challenger_id") is None
                else str(state["challenger_id"])
            ),
            score=(None if state.get("score") is None else _finite_number(state["score"], "audit.score")),
            unclipped_difference=(
                None
                if state.get("unclipped_difference") is None
                else _finite_number(state["unclipped_difference"], "audit.unclipped_difference")
            ),
            eta=(None if state.get("eta") is None else validate_eta(state["eta"])),
            factor=(
                None
                if state.get("factor") is None
                else _finite_number(state["factor"], "audit.factor")
            ),
            wealth_before=_finite_number(cast(Number, state.get("wealth_before")), "audit.wealth_before"),
            wealth_after=_finite_number(cast(Number, state.get("wealth_after")), "audit.wealth_after"),
            alpha=validate_alpha(cast(Number, state.get("alpha"))),
            threshold=_finite_number(cast(Number, state.get("threshold")), "audit.threshold"),
            state_before=str(state.get("state_before", "active")),
            state_after=str(state.get("state_after", "active")),
            threshold_crossed=bool(state.get("threshold_crossed", False)),
            reason=None if state.get("reason") is None else str(state["reason"]),
            metadata=(
                None
                if state.get("metadata") is None
                else dict(state["metadata"])
            ),
        )


@dataclass(frozen=True)
class EProcessUpdate:
    """Result of one e-process observation."""

    epoch: int
    observation: int
    score: float
    eta: float
    factor: float
    e_value_before: float
    e_value_after: float
    threshold: float
    threshold_crossed: bool
    first_crossing: bool
    audit_record: AuditRecord

    @property
    def e_value(self) -> float:
        return self.e_value_after

    @property
    def wealth(self) -> float:
        return self.e_value_after

    def __float__(self) -> float:
        """Allow a result to be used as its post-update e-value when useful."""

        return self.e_value_after


class EProcess:
    """Nonnegative product e-process for a bounded score sequence."""

    def __init__(
        self,
        alpha: Number = DEFAULT_ALPHA,
        eta: EtaRule = DEFAULT_ETA,
        *,
        score_bound: Number = DEFAULT_SCORE_BOUND,
        initial_wealth: Number = 1.0,
        epoch: int = 0,
        challenger_id: Optional[str] = None,
    ) -> None:
        self._alpha = validate_alpha(alpha)
        self._score_bound = validate_score_bound(score_bound)
        self._initial_wealth = _finite_number(initial_wealth, "initial wealth")
        if self._initial_wealth <= 0:
            raise ValueError("initial wealth must be greater than zero")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a nonnegative integer")
        if not callable(eta):
            validate_eta(eta)
        self._default_eta = eta
        self._epoch = epoch
        self._wealth = self._initial_wealth
        self._observation_count = 0
        self._history: list[float] = []
        self._audit: list[AuditRecord] = []
        self._ever_crossed = False
        self._first_crossing_observation: Optional[int] = None
        self._challenger_id = challenger_id

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def score_bound(self) -> float:
        return self._score_bound

    @property
    def initial_wealth(self) -> float:
        return self._initial_wealth

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def e_value(self) -> float:
        return self._wealth

    @property
    def wealth(self) -> float:
        return self._wealth

    @property
    def state(self) -> "EProcessSnapshot":
        """Read-only state snapshot for callers that prefer ``process.state``."""

        return self.snapshot()

    @property
    def threshold(self) -> float:
        return optional_stopping_threshold(
            self._alpha,
            initial_wealth=self._initial_wealth,
        )

    @property
    def observations(self) -> int:
        return self._observation_count

    @property
    def history(self) -> Tuple[float, ...]:
        return tuple(self._history)

    @property
    def audit_records(self) -> Tuple[AuditRecord, ...]:
        return tuple(self._audit)

    @property
    def ever_crossed(self) -> bool:
        """Whether this epoch has crossed its threshold at least once."""

        return self._ever_crossed

    @property
    def first_crossing_observation(self) -> Optional[int]:
        return self._first_crossing_observation

    def resolve_eta(self, eta: Optional[EtaRule] = None) -> float:
        """Resolve eta using only the history preceding the next score."""

        rule = self._default_eta if eta is None else eta
        if not callable(rule):
            return validate_eta(rule)
        return resolve_betting_fraction(rule, self.history)

    def threshold_reached(self) -> bool:
        """Check the current e-value against the current epoch threshold."""

        return optional_stopping_threshold_reached(
            self._wealth,
            self._alpha,
            initial_wealth=self._initial_wealth,
        )

    @property
    def threshold_crossed(self) -> bool:
        """Readable property alias for the current threshold check."""

        return self.threshold_reached()

    @property
    def audit_log(self) -> Tuple[AuditRecord, ...]:
        """Alias for the append-only audit record sequence."""

        return self.audit_records

    def check_optional_stopping(self) -> bool:
        """Alias for :meth:`threshold_reached`."""

        return self.threshold_reached()

    def would_cross(self, score: Number, eta: Optional[EtaRule] = None) -> bool:
        """Check the next threshold crossing without changing process state."""

        score_value = _validate_score(score)
        eta_value = self.resolve_eta(eta)
        factor = 1.0 + eta_value * score_value
        if self._wealth == math.inf:
            next_value = 0.0 if factor == 0.0 else math.inf
        else:
            next_value = self._wealth * factor
        return optional_stopping_threshold_reached(
            next_value,
            self._alpha,
            initial_wealth=self._initial_wealth,
        )

    def update(
        self,
        score: Number,
        *,
        eta: Optional[EtaRule] = None,
        unclipped_difference: Optional[Number] = None,
        state_before: Union[str, PromotionStatus] = PromotionStatus.ACTIVE,
        state_after: Union[str, PromotionStatus] = PromotionStatus.ACTIVE,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> EProcessUpdate:
        """Consume one bounded score and multiply the e-process.

        ``eta`` is resolved before ``score`` is added to history.  If it is a
        callable, it receives an immutable tuple of prior scores only.  The
        update factor is always ``1 + eta * score`` and is nonnegative by
        construction.
        """

        score_value = _validate_score(score)
        eta_value = self.resolve_eta(eta)
        if unclipped_difference is not None:
            raw_difference = _finite_number(
                unclipped_difference,
                "unclipped utility difference",
            )
        else:
            raw_difference = None
        factor = 1.0 + eta_value * score_value
        if factor < 0.0 or not math.isfinite(factor):
            raise AssertionError("bounded score and eta must yield a finite nonnegative factor")
        if len(self._audit) >= MAX_PROMOTION_AUDIT_RECORDS:
            raise PromotionError(
                "e-process audit history exceeds the configured resource limit "
                f"({MAX_PROMOTION_AUDIT_RECORDS})"
            )
        before = self._wealth
        if before == math.inf:
            raise FloatingPointError("e-process wealth is already non-finite")
        else:
            after = before * factor
        if not math.isfinite(after) or after < 0.0:
            raise FloatingPointError("e-process wealth overflowed or became negative")
        if self._observation_count >= MAX_PROMOTION_HISTORY:
            raise PromotionError(
                "e-process history exceeds the configured resource limit "
                f"({MAX_PROMOTION_HISTORY})"
            )
        observation = self._observation_count + 1
        threshold = self.threshold
        crossed = after >= threshold
        first_crossing = crossed and not self._ever_crossed

        self._wealth = after
        self._observation_count = observation
        self._history.append(score_value)
        if first_crossing:
            self._first_crossing_observation = observation
        if crossed:
            self._ever_crossed = True

        record = AuditRecord(
            event="observation",
            epoch=self._epoch,
            observation=observation,
            challenger_id=self._challenger_id,
            score=score_value,
            unclipped_difference=raw_difference,
            eta=eta_value,
            factor=factor,
            wealth_before=before,
            wealth_after=after,
            alpha=self._alpha,
            threshold=threshold,
            state_before=_state_value(state_before),
            state_after=_state_value(state_after),
            threshold_crossed=crossed,
            metadata=metadata,
        )
        if len(self._audit) >= MAX_PROMOTION_AUDIT_RECORDS:
            raise PromotionError(
                "e-process audit history exceeds the configured resource limit "
                f"({MAX_PROMOTION_AUDIT_RECORDS})"
            )
        self._audit.append(record)
        return EProcessUpdate(
            epoch=self._epoch,
            observation=observation,
            score=score_value,
            eta=eta_value,
            factor=factor,
            e_value_before=before,
            e_value_after=after,
            threshold=threshold,
            threshold_crossed=crossed,
            first_crossing=first_crossing,
            audit_record=record,
        )

    def update_utilities(
        self,
        challenger_utility: Number,
        incumbent_utility: Number,
        *,
        eta: Optional[EtaRule] = None,
        state_before: Union[str, PromotionStatus] = PromotionStatus.ACTIVE,
        state_after: Union[str, PromotionStatus] = PromotionStatus.ACTIVE,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> EProcessUpdate:
        """Convert utilities to a bounded score and consume that score."""

        challenger = _validate_utility(challenger_utility, "challenger utility")
        incumbent = _validate_utility(incumbent_utility, "incumbent utility")
        difference = challenger - incumbent
        audit_difference = difference if math.isfinite(difference) else None
        score = bounded_utility_difference(
            challenger,
            incumbent,
            bound=self._score_bound,
        )
        return self.update(
            score,
            eta=eta,
            unclipped_difference=audit_difference,
            state_before=state_before,
            state_after=state_after,
            metadata=metadata,
        )

    # A short spelling is useful for callers that treat the process as a
    # stream of utility observations.
    observe = update
    observe_utilities = update_utilities

    def reset(
        self,
        *,
        epoch: Optional[int] = None,
        alpha: Optional[Number] = None,
        reason: str = "epoch reset",
        state_before: Union[str, PromotionStatus] = PromotionStatus.ACTIVE,
        state_after: Union[str, PromotionStatus] = PromotionStatus.ACTIVE,
    ) -> AuditRecord:
        """Start a fresh epoch while retaining the prior audit trail.

        Resetting sets wealth to the original starting wealth and clears only
        the current epoch's predictability history.  The default epoch is the
        next integer; an explicit epoch must be strictly greater than the
        current one.  A caller that wants repeated testing to remain globally
        controlled should also allocate a fresh alpha, as :class:`PromotionGate`
        does.
        """

        if len(self._audit) >= MAX_PROMOTION_AUDIT_RECORDS:
            raise PromotionError(
                "e-process audit history exceeds the configured resource limit "
                f"({MAX_PROMOTION_AUDIT_RECORDS})"
            )

        if epoch is None:
            next_epoch = self._epoch + 1
        elif isinstance(epoch, bool) or not isinstance(epoch, int):
            raise ValueError("epoch must be a nonnegative integer")
        else:
            next_epoch = epoch
        if next_epoch <= self._epoch:
            raise ValueError("new epoch must be greater than the current epoch")
        if alpha is None:
            next_alpha = self._alpha
        else:
            next_alpha = validate_alpha(alpha)

        before = self._wealth
        old_alpha = self._alpha
        old_threshold = self.threshold
        self._epoch = next_epoch
        self._alpha = next_alpha
        self._wealth = self._initial_wealth
        self._observation_count = 0
        self._history.clear()
        self._ever_crossed = False
        self._first_crossing_observation = None
        record = AuditRecord(
            event="reset",
            epoch=self._epoch,
            observation=None,
            challenger_id=self._challenger_id,
            score=None,
            unclipped_difference=None,
            eta=None,
            factor=None,
            wealth_before=before,
            wealth_after=self._wealth,
            alpha=self._alpha,
            threshold=self.threshold,
            state_before=_state_value(state_before),
            state_after=_state_value(state_after),
            threshold_crossed=False,
            reason=(
                f"{reason}; previous_epoch={self._epoch - 1}; "
                f"previous_alpha={old_alpha}; previous_threshold={old_threshold}"
            ),
        )
        if len(self._audit) >= MAX_PROMOTION_AUDIT_RECORDS:
            raise PromotionError(
                "e-process audit history exceeds the configured resource limit "
                f"({MAX_PROMOTION_AUDIT_RECORDS})"
            )
        self._audit.append(record)
        return record

    def snapshot(self) -> "EProcessSnapshot":
        return EProcessSnapshot(
            epoch=self._epoch,
            e_value=self._wealth,
            alpha=self._alpha,
            threshold=self.threshold,
            observations=self._observation_count,
            history=self.history,
            ever_crossed=self._ever_crossed,
            first_crossing_observation=self._first_crossing_observation,
        )

    def state_dict(self) -> dict[str, Any]:
        """Return complete finite state for a checkpoint envelope."""

        if len(self._history) > MAX_PROMOTION_HISTORY or len(self._audit) > MAX_PROMOTION_AUDIT_RECORDS:
            raise PromotionError("promotion state exceeds its resource limit")
        return {
            "version": 1,
            "schema": "topology_gate.promotion.eprocess",
            "alpha": self._alpha,
            "score_bound": self._score_bound,
            "initial_wealth": self._initial_wealth,
            "epoch": self._epoch,
            "challenger_id": self._challenger_id,
            "eta": _eta_state(self._default_eta),
            "wealth": self._wealth,
            "observation_count": self._observation_count,
            "history": list(self._history),
            "ever_crossed": self._ever_crossed,
            "first_crossing_observation": self._first_crossing_observation,
            "audit_records": [record.to_dict() for record in self._audit],
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, Any],
        *,
        eta: Optional[EtaRule] = None,
    ) -> "EProcess":
        if not isinstance(state, Mapping) or state.get("version") != 1:
            raise PromotionError("unsupported e-process state version")
        if state.get("schema") != "topology_gate.promotion.eprocess":
            raise PromotionError("unsupported e-process state schema")
        rule = _eta_from_state(state.get("eta", {}), eta)
        candidate = cls(
            alpha=cast(Number, state.get("alpha")),
            eta=rule,
            score_bound=cast(Number, state.get("score_bound")),
            initial_wealth=cast(Number, state.get("initial_wealth")),
            epoch=cast(int, state.get("epoch")),
            challenger_id=state.get("challenger_id"),
        )
        history_raw = state.get("history", ())
        if isinstance(history_raw, (str, bytes, bytearray)):
            raise PromotionError("e-process history must be a sequence")
        history = tuple(_validate_score(value) for value in history_raw)
        if len(history) > MAX_PROMOTION_HISTORY:
            raise PromotionError("e-process history exceeds its resource limit")
        observations = state.get("observation_count")
        if not isinstance(observations, int) or isinstance(observations, bool):
            raise PromotionError("e-process observation_count must be an integer")
        if observations != len(history):
            raise PromotionError("e-process observation_count disagrees with history")
        wealth = _finite_number(cast(Number, state.get("wealth")), "e-process wealth")
        if wealth < 0.0:
            raise PromotionError("e-process wealth must be non-negative")
        first = state.get("first_crossing_observation")
        if first is not None:
            if not isinstance(first, int) or isinstance(first, bool) or not 0 < first <= observations:
                raise PromotionError("invalid first_crossing_observation")
        audit_raw = state.get("audit_records", ())
        if isinstance(audit_raw, (str, bytes, bytearray)):
            raise PromotionError("e-process audit_records must be a sequence")
        audit = [AuditRecord.from_dict(value) for value in audit_raw]
        if len(audit) > MAX_PROMOTION_AUDIT_RECORDS:
            raise PromotionError("e-process audit history exceeds its resource limit")
        current_epoch = int(cast(int, state.get("epoch")))
        if any(record.epoch > current_epoch for record in audit):
            raise PromotionError("e-process audit epoch exceeds state epoch")
        candidate._wealth = wealth
        candidate._observation_count = observations
        candidate._history = list(history)
        candidate._ever_crossed = bool(state.get("ever_crossed", False))
        candidate._first_crossing_observation = first
        candidate._audit = audit
        return candidate

    def load_state_dict(
        self,
        state: Mapping[str, Any],
        *,
        eta: Optional[EtaRule] = None,
    ) -> "EProcess":
        candidate = type(self).from_state_dict(
            state,
            eta=self._default_eta if eta is None else eta,
        )
        self.__dict__.update(candidate.__dict__)
        return self


@dataclass(frozen=True)
class EProcessSnapshot:
    """Read-only current state of an :class:`EProcess`."""

    epoch: int
    e_value: float
    alpha: float
    threshold: float
    observations: int
    history: Tuple[float, ...]
    ever_crossed: bool
    first_crossing_observation: Optional[int]


@dataclass(frozen=True)
class PromotionDecision:
    """Read-only result of a promotion state-machine observation."""

    challenger_id: str
    epoch: int
    observation: int
    score: float
    eta: float
    factor: float
    e_value: float
    alpha: float
    threshold: float
    threshold_crossed: bool
    promoted: bool
    state: PromotionStatus
    audit_record: AuditRecord

    @property
    def status(self) -> PromotionStatus:
        return self.state

    @property
    def wealth(self) -> float:
        return self.e_value


def _state_value(state: Union[str, PromotionStatus]) -> str:
    if isinstance(state, PromotionStatus):
        return state.value
    return str(state)


def _eta_state(eta: EtaRule) -> dict[str, Any]:
    """Describe an eta rule without attempting to serialize executable code."""

    if callable(eta):
        return {
            "kind": "callable",
            "identity": (
                f"{getattr(eta, '__module__', type(eta).__module__)}:"
                f"{getattr(eta, '__qualname__', type(eta).__qualname__)}"
            ),
            "value": None,
        }
    return {"kind": "constant", "identity": None, "value": validate_eta(eta)}


def _eta_from_state(state: Mapping[str, Any], supplied: Optional[EtaRule]) -> EtaRule:
    kind = state.get("kind")
    if kind == "constant":
        return validate_eta(cast(Number, state.get("value")))
    if kind == "callable":
        if supplied is None or not callable(supplied):
            raise PromotionError(
                "a callable eta rule must be supplied when restoring promotion state"
            )
        expected_identity = state.get("identity")
        actual_identity = (
            f"{getattr(supplied, '__module__', type(supplied).__module__)}:"
            f"{getattr(supplied, '__qualname__', type(supplied).__qualname__)}"
        )
        if expected_identity is not None and expected_identity != actual_identity:
            raise PromotionError("callable eta identity does not match promotion state")
        return supplied
    raise PromotionError("unsupported eta rule state")


class PromotionStateMachine:
    """Promotion state machine for one challenger.

    A challenger is promoted on the first threshold crossing.  Further
    observations may be recorded for diagnostics, but the state remains
    promoted until :meth:`reset` starts a new epoch.  This makes the decision
    equivalent to stopping at the first crossing while preserving an audit
    trail if a caller continues collecting data.
    """

    def __init__(
        self,
        challenger_id: str = "challenger",
        *,
        alpha: Number = DEFAULT_ALPHA,
        eta: EtaRule = DEFAULT_ETA,
        score_bound: Number = DEFAULT_SCORE_BOUND,
        initial_wealth: Number = 1.0,
        epoch: int = 0,
        incumbent_id: Optional[str] = None,
    ) -> None:
        if not isinstance(challenger_id, str) or not challenger_id:
            raise ValueError("challenger_id must be a non-empty string")
        self._challenger_id = challenger_id
        self._incumbent_id = incumbent_id
        self._state = PromotionStatus.ACTIVE
        self._process = EProcess(
            alpha=alpha,
            eta=eta,
            score_bound=score_bound,
            initial_wealth=initial_wealth,
            epoch=epoch,
            challenger_id=challenger_id,
        )

    @property
    def challenger_id(self) -> str:
        return self._challenger_id

    @property
    def incumbent_id(self) -> Optional[str]:
        return self._incumbent_id

    @property
    def state(self) -> PromotionStatus:
        return self._state

    @property
    def status(self) -> PromotionStatus:
        return self._state

    @property
    def promoted(self) -> bool:
        return self._state is PromotionStatus.PROMOTED

    @property
    def epoch(self) -> int:
        return self._process.epoch

    @property
    def alpha(self) -> float:
        return self._process.alpha

    @property
    def threshold(self) -> float:
        return self._process.threshold

    @property
    def e_value(self) -> float:
        return self._process.e_value

    @property
    def wealth(self) -> float:
        return self._process.wealth

    @property
    def observations(self) -> int:
        return self._process.observations

    @property
    def score_history(self) -> Tuple[float, ...]:
        return self._process.history

    @property
    def ever_crossed(self) -> bool:
        return self._process.ever_crossed

    @property
    def first_crossing_observation(self) -> Optional[int]:
        return self._process.first_crossing_observation

    @property
    def audit_records(self) -> Tuple[AuditRecord, ...]:
        return self._process.audit_records

    @property
    def audit_log(self) -> Tuple[AuditRecord, ...]:
        return self.audit_records

    @property
    def threshold_crossed(self) -> bool:
        return self._process.threshold_reached()

    def observe_score(
        self,
        score: Number,
        *,
        eta: Optional[EtaRule] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> PromotionDecision:
        """Record a bounded score and transition to promoted if it crosses."""

        score_value = _validate_score(score)
        eta_value = self._process.resolve_eta(eta)
        state_before = self._state
        crossed = self._process.would_cross(score_value, eta=eta_value)
        state_after = (
            PromotionStatus.PROMOTED
            if state_before is PromotionStatus.ACTIVE and crossed
            else state_before
        )
        update = self._process.update(
            score_value,
            eta=eta_value,
            state_before=state_before,
            state_after=state_after,
            metadata=metadata,
        )
        if state_after is PromotionStatus.PROMOTED:
            self._state = PromotionStatus.PROMOTED
        return PromotionDecision(
            challenger_id=self._challenger_id,
            epoch=update.epoch,
            observation=update.observation,
            score=update.score,
            eta=update.eta,
            factor=update.factor,
            e_value=update.e_value_after,
            alpha=self.alpha,
            threshold=update.threshold,
            threshold_crossed=update.threshold_crossed,
            promoted=self.promoted and state_before is PromotionStatus.ACTIVE,
            state=self._state,
            audit_record=update.audit_record,
        )

    def observe_utilities(
        self,
        challenger_utility: Number,
        incumbent_utility: Number,
        *,
        eta: Optional[EtaRule] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> PromotionDecision:
        """Record a utility pair after clipping and normalizing its difference."""

        challenger = _validate_utility(challenger_utility, "challenger utility")
        incumbent = _validate_utility(incumbent_utility, "incumbent utility")
        difference = challenger - incumbent
        audit_difference = difference if math.isfinite(difference) else None
        score = bounded_utility_difference(
            challenger,
            incumbent,
            bound=self._process.score_bound,
        )
        score_value = _validate_score(score)
        eta_value = self._process.resolve_eta(eta)
        state_before = self._state
        crossed = self._process.would_cross(score_value, eta=eta_value)
        state_after = (
            PromotionStatus.PROMOTED
            if state_before is PromotionStatus.ACTIVE and crossed
            else state_before
        )
        update = self._process.update(
            score_value,
            eta=eta_value,
            unclipped_difference=audit_difference,
            state_before=state_before,
            state_after=state_after,
            metadata=metadata,
        )
        if state_after is PromotionStatus.PROMOTED:
            self._state = PromotionStatus.PROMOTED
        return PromotionDecision(
            challenger_id=self._challenger_id,
            epoch=update.epoch,
            observation=update.observation,
            score=update.score,
            eta=update.eta,
            factor=update.factor,
            e_value=update.e_value_after,
            alpha=self.alpha,
            threshold=update.threshold,
            threshold_crossed=update.threshold_crossed,
            promoted=self.promoted and state_before is PromotionStatus.ACTIVE,
            state=self._state,
            audit_record=update.audit_record,
        )

    def observe(
        self,
        value: Number,
        incumbent_utility: Optional[Number] = None,
        *,
        eta: Optional[EtaRule] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> PromotionDecision:
        """Observe either a bounded score or a challenger/incumbent pair."""

        if incumbent_utility is None:
            return self.observe_score(value, eta=eta, metadata=metadata)
        return self.observe_utilities(
            value,
            incumbent_utility,
            eta=eta,
            metadata=metadata,
        )

    # Keep update names score-oriented; utility pairs have explicit names.
    update = observe_score
    update_utilities = observe_utilities

    def reset(
        self,
        *,
        epoch: Optional[int] = None,
        alpha: Optional[Number] = None,
        reason: str = "epoch reset",
    ) -> AuditRecord:
        """Reset wealth/history and return the immutable reset audit record."""

        record = self._process.reset(
            epoch=epoch,
            alpha=alpha,
            reason=reason,
            state_before=self._state,
            state_after=PromotionStatus.ACTIVE,
        )
        self._state = PromotionStatus.ACTIVE
        return record

    def check_optional_stopping(self) -> bool:
        return self._process.threshold_reached()

    def snapshot(self) -> "PromotionSnapshot":
        return PromotionSnapshot(
            challenger_id=self._challenger_id,
            incumbent_id=self._incumbent_id,
            state=self._state,
            epoch=self.epoch,
            e_value=self.e_value,
            alpha=self.alpha,
            threshold=self.threshold,
            observations=self.observations,
            score_history=self.score_history,
            ever_crossed=self.ever_crossed,
            first_crossing_observation=self.first_crossing_observation,
        )

    def state_dict(self) -> dict[str, Any]:
        """Return complete challenger state for a checkpoint envelope."""

        return {
            "version": 1,
            "schema": "topology_gate.promotion.state_machine",
            "challenger_id": self._challenger_id,
            "incumbent_id": self._incumbent_id,
            "state": self._state.value,
            "process": self._process.state_dict(),
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, Any],
        *,
        eta: Optional[EtaRule] = None,
    ) -> "PromotionStateMachine":
        if not isinstance(state, Mapping) or state.get("version") != 1:
            raise PromotionError("unsupported promotion state-machine version")
        if state.get("schema") != "topology_gate.promotion.state_machine":
            raise PromotionError("unsupported promotion state-machine schema")
        process_state = state.get("process")
        if not isinstance(process_state, Mapping):
            raise PromotionError("promotion state is missing its e-process")
        process = EProcess.from_state_dict(process_state, eta=eta)
        try:
            candidate_state = PromotionStatus(str(state.get("state")))
        except ValueError as exc:
            raise PromotionError("unsupported promotion state") from exc
        candidate = cls(
            str(state.get("challenger_id")),
            alpha=process.alpha,
            eta=process._default_eta,
            score_bound=process.score_bound,
            initial_wealth=process.initial_wealth,
            epoch=process.epoch,
            incumbent_id=(
                None
                if state.get("incumbent_id") is None
                else str(state.get("incumbent_id"))
            ),
        )
        candidate._process = process
        candidate._state = candidate_state
        return candidate

    def load_state_dict(
        self,
        state: Mapping[str, Any],
        *,
        eta: Optional[EtaRule] = None,
    ) -> "PromotionStateMachine":
        candidate = type(self).from_state_dict(state, eta=eta)
        self.__dict__.update(candidate.__dict__)
        return self


@dataclass(frozen=True)
class PromotionSnapshot:
    """Read-only state-machine snapshot."""

    challenger_id: str
    incumbent_id: Optional[str]
    state: PromotionStatus
    epoch: int
    e_value: float
    alpha: float
    threshold: float
    observations: int
    score_history: Tuple[float, ...]
    ever_crossed: bool
    first_crossing_observation: Optional[int]

    @property
    def promoted(self) -> bool:
        return self.state is PromotionStatus.PROMOTED


@dataclass(frozen=True)
class ChallengerState:
    """Read-only state returned by :class:`PromotionGate`."""

    challenger_id: str
    challenger_index: int
    epoch: int
    state: PromotionStatus
    e_value: float
    alpha: float
    threshold: float
    observations: int
    score_history: Tuple[float, ...]
    ever_crossed: bool

    @property
    def status(self) -> PromotionStatus:
        return self.state

    @property
    def promoted(self) -> bool:
        return self.state is PromotionStatus.PROMOTED


class GateStatus(str, Enum):
    OPEN = "open"
    PROMOTED = "promoted"


@dataclass
class _RegisteredChallenger:
    index: int
    machine: PromotionStateMachine


class PromotionGate:
    """Control promotion of multiple challengers under one alpha budget.

    Each registered challenger receives a one-based geometric slot, and each
    epoch receives a geometric share of that slot.  The sum of all possible
    allocations across all challengers and epochs is at most the gate's
    ``global_alpha``.  A gate promotes the first challenger whose own
    nonnegative e-process crosses its allocated threshold.
    """

    def __init__(
        self,
        incumbent_id: str = "incumbent",
        *,
        alpha: Number = DEFAULT_ALPHA,
        eta: EtaRule = DEFAULT_ETA,
        score_bound: Number = DEFAULT_SCORE_BOUND,
        initial_wealth: Number = 1.0,
    ) -> None:
        if not isinstance(incumbent_id, str) or not incumbent_id:
            raise ValueError("incumbent_id must be a non-empty string")
        self._incumbent_id = incumbent_id
        self._global_alpha = validate_alpha(alpha)
        self._default_eta = eta
        if not callable(eta):
            validate_eta(eta)
        self._score_bound = validate_score_bound(score_bound)
        self._initial_wealth = _finite_number(initial_wealth, "initial wealth")
        if self._initial_wealth <= 0:
            raise ValueError("initial wealth must be greater than zero")
        self._epoch = 0
        self._status = GateStatus.OPEN
        self._promoted_challenger_id: Optional[str] = None
        self._challengers: dict[str, _RegisteredChallenger] = {}
        self._allocated_alpha = 0.0
        self._audit: list[AuditRecord] = []

    @property
    def incumbent_id(self) -> str:
        return self._incumbent_id

    @property
    def global_alpha(self) -> float:
        return self._global_alpha

    @property
    def alpha(self) -> float:
        return self._global_alpha

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def status(self) -> GateStatus:
        return self._status

    @property
    def promoted_challenger_id(self) -> Optional[str]:
        return self._promoted_challenger_id

    @property
    def audit_records(self) -> Tuple[AuditRecord, ...]:
        return tuple(self._audit)

    @property
    def alpha_spent(self) -> float:
        """Alpha assigned to registered challenger/epoch slots so far."""

        return self._allocated_alpha

    @property
    def alpha_budget_remaining(self) -> float:
        return max(0.0, self._global_alpha - self._allocated_alpha)

    @property
    def alpha_budget_bound(self) -> float:
        return self._global_alpha

    @property
    def challenger_ids(self) -> Tuple[str, ...]:
        return tuple(self._challengers)

    def _allocation(self, index: int, epoch: Optional[int] = None) -> float:
        return geometric_alpha_allocation(
            self._global_alpha,
            index,
            epoch=self._epoch if epoch is None else epoch,
        )

    def _gate_audit(
        self,
        *,
        event: str,
        challenger_id: Optional[str],
        wealth_before: float,
        wealth_after: float,
        alpha: float,
        threshold: float,
        state_before: Union[str, GateStatus],
        state_after: Union[str, GateStatus],
        reason: Optional[str] = None,
        append: bool = True,
    ) -> AuditRecord:
        record = AuditRecord(
            event=event,
            epoch=self._epoch,
            observation=None,
            challenger_id=challenger_id,
            score=None,
            unclipped_difference=None,
            eta=None,
            factor=None,
            wealth_before=wealth_before,
            wealth_after=wealth_after,
            alpha=alpha,
            threshold=threshold,
            state_before=_state_value(state_before),
            state_after=_state_value(state_after),
            threshold_crossed=False,
            reason=reason,
        )
        if append:
            if len(self._audit) >= MAX_PROMOTION_AUDIT_RECORDS:
                raise PromotionError(
                    "promotion gate audit history exceeds the configured resource "
                    f"limit ({MAX_PROMOTION_AUDIT_RECORDS})"
                )
            self._audit.append(record)
        return record

    def register_challenger(
        self,
        challenger_id: str,
        *,
        eta: Optional[EtaRule] = None,
    ) -> ChallengerState:
        """Register a challenger and reserve its alpha slot permanently."""

        if not isinstance(challenger_id, str) or not challenger_id:
            raise ValueError("challenger_id must be a non-empty string")
        if challenger_id in self._challengers:
            raise ValueError(f"challenger {challenger_id!r} is already registered")
        index = len(self._challengers) + 1
        candidate_eta = self._default_eta if eta is None else eta
        if not callable(candidate_eta):
            validate_eta(candidate_eta)
        candidate_alpha = self._allocation(index, epoch=self._epoch)
        machine = PromotionStateMachine(
            challenger_id,
            alpha=candidate_alpha,
            eta=candidate_eta,
            score_bound=self._score_bound,
            initial_wealth=self._initial_wealth,
            epoch=self._epoch,
            incumbent_id=self._incumbent_id,
        )
        self._challengers[challenger_id] = _RegisteredChallenger(index, machine)
        self._allocated_alpha += candidate_alpha
        self._gate_audit(
            event="register",
            challenger_id=challenger_id,
            wealth_before=self._initial_wealth,
            wealth_after=self._initial_wealth,
            alpha=candidate_alpha,
            threshold=machine.threshold,
            state_before=self._status,
            state_after=self._status,
            reason=f"reserved challenger slot {index}",
        )
        return self.challenger_state(challenger_id)

    def _get(self, challenger_id: str) -> _RegisteredChallenger:
        try:
            return self._challengers[challenger_id]
        except KeyError as exc:
            raise KeyError(f"unknown challenger {challenger_id!r}") from exc

    def challenger_state(self, challenger_id: str) -> ChallengerState:
        registered = self._get(challenger_id)
        machine = registered.machine
        return ChallengerState(
            challenger_id=challenger_id,
            challenger_index=registered.index,
            epoch=machine.epoch,
            state=machine.state,
            e_value=machine.e_value,
            alpha=machine.alpha,
            threshold=machine.threshold,
            observations=machine.observations,
            score_history=machine.score_history,
            ever_crossed=machine.ever_crossed,
        )

    def allocation_for(self, challenger_id: str) -> float:
        return self._get(challenger_id).machine.alpha

    def observe_score(
        self,
        challenger_id: str,
        score: Number,
        *,
        eta: Optional[EtaRule] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> PromotionDecision:
        """Observe a score for a candidate while the gate is open."""

        if self._status is GateStatus.PROMOTED:
            raise PromotionClosedError(
                f"gate already promoted {self._promoted_challenger_id!r}; reset before observing"
            )
        registered = self._get(challenger_id)
        decision = registered.machine.observe_score(
            score,
            eta=eta,
            metadata=metadata,
        )
        if len(self._audit) >= MAX_PROMOTION_AUDIT_RECORDS:
            raise PromotionError("promotion gate audit history exceeds its resource limit")
        self._audit.append(decision.audit_record)
        if decision.promoted:
            gate_before = self._status
            self._status = GateStatus.PROMOTED
            self._promoted_challenger_id = challenger_id
            self._gate_audit(
                event="gate_promotion",
                challenger_id=challenger_id,
                wealth_before=decision.e_value,
                wealth_after=decision.e_value,
                alpha=decision.alpha,
                threshold=decision.threshold,
                state_before=gate_before,
                state_after=self._status,
                reason="first allocated threshold crossing",
            )
        return decision

    def observe_utilities(
        self,
        challenger_id: str,
        challenger_utility: Number,
        incumbent_utility: Number,
        *,
        eta: Optional[EtaRule] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> PromotionDecision:
        """Observe utilities for a candidate using the bounded score contract."""

        if self._status is GateStatus.PROMOTED:
            raise PromotionClosedError(
                f"gate already promoted {self._promoted_challenger_id!r}; reset before observing"
            )
        registered = self._get(challenger_id)
        decision = registered.machine.observe_utilities(
            challenger_utility,
            incumbent_utility,
            eta=eta,
            metadata=metadata,
        )
        if len(self._audit) >= MAX_PROMOTION_AUDIT_RECORDS:
            raise PromotionError("promotion gate audit history exceeds its resource limit")
        self._audit.append(decision.audit_record)
        if decision.promoted:
            gate_before = self._status
            self._status = GateStatus.PROMOTED
            self._promoted_challenger_id = challenger_id
            self._gate_audit(
                event="gate_promotion",
                challenger_id=challenger_id,
                wealth_before=decision.e_value,
                wealth_after=decision.e_value,
                alpha=decision.alpha,
                threshold=decision.threshold,
                state_before=gate_before,
                state_after=self._status,
                reason="first allocated threshold crossing",
            )
        return decision

    def observe(
        self,
        challenger_id: str,
        value: Number,
        incumbent_utility: Optional[Number] = None,
        *,
        eta: Optional[EtaRule] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> PromotionDecision:
        """Observe either a bounded score or a challenger/incumbent pair."""

        if incumbent_utility is None:
            return self.observe_score(
                challenger_id,
                value,
                eta=eta,
                metadata=metadata,
            )
        return self.observe_utilities(
            challenger_id,
            value,
            incumbent_utility,
            eta=eta,
            metadata=metadata,
        )

    update = observe_score
    update_utilities = observe_utilities

    def reset_epoch(
        self,
        *,
        incumbent_id: Optional[str] = None,
        reason: str = "epoch reset",
    ) -> Tuple[AuditRecord, ...]:
        """Reset every challenger into a fresh, separately funded epoch.

        If a candidate was promoted and no new incumbent is supplied, that
        candidate becomes the next epoch's incumbent.  All prior audit records
        remain available; only current e-values and histories reset.
        """

        old_status = self._status
        old_incumbent = self._incumbent_id
        if incumbent_id is not None:
            if not isinstance(incumbent_id, str) or not incumbent_id:
                raise ValueError("incumbent_id must be a non-empty string")
            next_incumbent = incumbent_id
        elif self._promoted_challenger_id is not None:
            next_incumbent = self._promoted_challenger_id
        else:
            next_incumbent = old_incumbent

        machine_epochs = [
            registered.machine.epoch for registered in self._challengers.values()
        ]
        # A per-challenger reset may have advanced one machine ahead of the
        # gate.  Skip directly to a common later epoch rather than attempting
        # to reuse that machine's epoch or alpha allocation.
        next_epoch = max(
            [self._epoch + 1]
            + [machine_epoch + 1 for machine_epoch in machine_epochs]
        )
        reset_records: list[AuditRecord] = []
        self._epoch = next_epoch
        self._incumbent_id = next_incumbent
        self._status = GateStatus.OPEN
        self._promoted_challenger_id = None
        for challenger_id, registered in self._challengers.items():
            machine = registered.machine
            candidate_alpha = self._allocation(registered.index, epoch=next_epoch)
            self._allocated_alpha += candidate_alpha
            machine._incumbent_id = next_incumbent
            reset_records.append(
                machine.reset(
                    epoch=next_epoch,
                    alpha=candidate_alpha,
                    reason=reason,
                )
            )
        gate_record = self._gate_audit(
            event="gate_reset",
            challenger_id=None,
            wealth_before=self._initial_wealth,
            wealth_after=self._initial_wealth,
            alpha=self._global_alpha,
            threshold=optional_stopping_threshold(self._global_alpha),
            state_before=old_status,
            state_after=self._status,
            reason=(
                f"{reason}; previous_incumbent={old_incumbent}; "
                f"new_incumbent={next_incumbent}"
            ),
            append=False,
        )
        # Gate-level records are appended after per-candidate resets so a
        # consumer can replay the reset as one ordered operation.
        if len(self._audit) + len(reset_records) + 1 > MAX_PROMOTION_AUDIT_RECORDS:
            raise PromotionError("promotion gate audit history exceeds its resource limit")
        self._audit.extend(reset_records)
        self._audit.append(gate_record)
        return tuple(reset_records) + (gate_record,)

    def reset_challenger(
        self,
        challenger_id: str,
        *,
        reason: str = "challenger epoch reset",
    ) -> AuditRecord:
        """Reset one challenger and reserve its next epoch's alpha share."""

        registered = self._get(challenger_id)
        machine = registered.machine
        next_epoch = machine.epoch + 1
        candidate_alpha = self._allocation(registered.index, epoch=next_epoch)
        self._allocated_alpha += candidate_alpha
        record = machine.reset(
            epoch=next_epoch,
            alpha=candidate_alpha,
            reason=reason,
        )
        if self._promoted_challenger_id == challenger_id:
            self._status = GateStatus.OPEN
            self._promoted_challenger_id = None
        if len(self._audit) >= MAX_PROMOTION_AUDIT_RECORDS:
            raise PromotionError("promotion gate audit history exceeds its resource limit")
        self._audit.append(record)
        return record

    reset = reset_epoch

    def snapshots(self) -> Tuple[ChallengerState, ...]:
        return tuple(self.challenger_state(cid) for cid in self._challengers)

    def state_dict(self) -> dict[str, Any]:
        """Return complete multi-challenger state for a checkpoint."""

        if len(self._audit) > MAX_PROMOTION_AUDIT_RECORDS:
            raise PromotionError("promotion gate audit history exceeds its resource limit")
        return {
            "version": 1,
            "schema": "topology_gate.promotion.gate",
            "incumbent_id": self._incumbent_id,
            "global_alpha": self._global_alpha,
            "score_bound": self._score_bound,
            "initial_wealth": self._initial_wealth,
            "epoch": self._epoch,
            "status": self._status.value,
            "promoted_challenger_id": self._promoted_challenger_id,
            "allocated_alpha": self._allocated_alpha,
            "eta": _eta_state(self._default_eta),
            "challengers": [
                {
                    "index": registered.index,
                    "state": registered.machine.state_dict(),
                }
                for registered in self._challengers.values()
            ],
            "audit_records": [record.to_dict() for record in self._audit],
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, Any],
        *,
        eta: Optional[EtaRule] = None,
    ) -> "PromotionGate":
        if not isinstance(state, Mapping) or state.get("version") != 1:
            raise PromotionError("unsupported promotion gate state version")
        if state.get("schema") != "topology_gate.promotion.gate":
            raise PromotionError("unsupported promotion gate state schema")
        gate_eta = _eta_from_state(state.get("eta", {}), eta)
        candidate = cls(
            str(state.get("incumbent_id")),
            alpha=cast(Number, state.get("global_alpha")),
            eta=gate_eta,
            score_bound=cast(Number, state.get("score_bound")),
            initial_wealth=cast(Number, state.get("initial_wealth")),
        )
        epoch = state.get("epoch")
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
            raise PromotionError("promotion gate epoch must be a nonnegative integer")
        try:
            status = GateStatus(str(state.get("status")))
        except ValueError as exc:
            raise PromotionError("unsupported promotion gate status") from exc
        challenger_raw = state.get("challengers", ())
        if isinstance(challenger_raw, (str, bytes, bytearray)):
            raise PromotionError("promotion challengers must be a sequence")
        if len(challenger_raw) > MAX_PROMOTION_HISTORY:
            raise PromotionError("promotion challenger state exceeds its resource limit")
        restored: dict[str, _RegisteredChallenger] = {}
        seen_indices: set[int] = set()
        for entry in challenger_raw:
            if not isinstance(entry, Mapping):
                raise PromotionError("promotion challenger entry must be a mapping")
            index = entry.get("index")
            if not isinstance(index, int) or isinstance(index, bool) or index < 1 or index in seen_indices:
                raise PromotionError("promotion challenger indices must be unique positive integers")
            machine_state = entry.get("state")
            if not isinstance(machine_state, Mapping):
                raise PromotionError("promotion challenger is missing state")
            machine = PromotionStateMachine.from_state_dict(
                machine_state,
                eta=gate_eta if eta is None else eta,
            )
            if machine.challenger_id in restored:
                raise PromotionError("duplicate promotion challenger id")
            restored[machine.challenger_id] = _RegisteredChallenger(index, machine)
            seen_indices.add(index)
        audit_raw = state.get("audit_records", ())
        if isinstance(audit_raw, (str, bytes, bytearray)):
            raise PromotionError("promotion gate audit_records must be a sequence")
        audit = [AuditRecord.from_dict(value) for value in audit_raw]
        if len(audit) > MAX_PROMOTION_AUDIT_RECORDS:
            raise PromotionError("promotion gate audit history exceeds its resource limit")
        allocated = _finite_number(
            cast(Number, state.get("allocated_alpha")),
            "promotion allocated alpha",
        )
        if allocated < 0.0 or allocated > candidate.global_alpha:
            # A valid finite run can allocate less than the global budget; it
            # must never restore a fabricated over-budget state.
            raise PromotionError("promotion allocated alpha exceeds global alpha")
        promoted_id = state.get("promoted_challenger_id")
        if promoted_id is not None and str(promoted_id) not in restored:
            raise PromotionError("promoted challenger is not registered")
        if status is GateStatus.PROMOTED and promoted_id is None:
            raise PromotionError("promoted gate is missing its challenger id")
        promoted_machines = [
            machine_id
            for machine_id, registered in restored.items()
            if registered.machine.promoted
        ]
        if status is GateStatus.PROMOTED:
            if promoted_machines != [str(promoted_id)]:
                raise PromotionError("promotion gate status disagrees with challenger state")
        elif promoted_machines:
            raise PromotionError("open promotion gate contains a promoted challenger")
        candidate._epoch = epoch
        candidate._status = status
        candidate._promoted_challenger_id = None if promoted_id is None else str(promoted_id)
        candidate._challengers = restored
        candidate._allocated_alpha = allocated
        candidate._audit = audit
        return candidate

    def load_state_dict(
        self,
        state: Mapping[str, Any],
        *,
        eta: Optional[EtaRule] = None,
    ) -> "PromotionGate":
        candidate = type(self).from_state_dict(state, eta=eta)
        self.__dict__.update(candidate.__dict__)
        return self


# Descriptive aliases for callers that use different vocabulary.
ChallengerPromotion = PromotionStateMachine
PromotionController = PromotionGate


__all__ = [
    "AuditRecord",
    "AlphaSpender",
    "ChallengerPromotion",
    "ChallengerState",
    "EProcess",
    "EProcessSnapshot",
    "EProcessUpdate",
    "GateStatus",
    "InvalidEtaError",
    "NULL_HYPOTHESIS",
    "PromotionClosedError",
    "PromotionController",
    "PromotionDecision",
    "PromotionError",
    "PromotionGate",
    "PromotionSnapshot",
    "PromotionState",
    "PromotionStateMachine",
    "PromotionStatus",
    "alpha_for_challenger",
    "bounded_score",
    "bounded_utility_difference",
    "check_optional_stopping_threshold",
    "clip_utility_difference",
    "geometric_alpha_allocation",
    "optional_stopping_threshold",
    "optional_stopping_threshold_reached",
    "predictable_betting_fraction",
    "resolve_betting_fraction",
    "validate_alpha",
    "validate_eta",
    "validate_score_bound",
]


class AlphaSpender:
    """Small explicit wrapper around the geometric multiple-challenger rule."""

    def __init__(self, alpha: Number = DEFAULT_ALPHA) -> None:
        self._alpha = validate_alpha(alpha)

    @property
    def alpha(self) -> float:
        return self._alpha

    def allocation(self, challenger_index: int, *, epoch: int = 0) -> float:
        return geometric_alpha_allocation(
            self._alpha,
            challenger_index,
            epoch=epoch,
        )

    def threshold(self, challenger_index: int, *, epoch: int = 0) -> float:
        return optional_stopping_threshold(
            self.allocation(challenger_index, epoch=epoch)
        )

    def total_possible_allocation(self) -> float:
        """Return the global upper bound, not a finite-registration total."""

        return self._alpha
