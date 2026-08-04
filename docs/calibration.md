# Calibration protocol

The topology detector's threshold is a policy parameter, not a theoretically
calibrated false-alarm level. Before using it to accelerate forgetting, run a
declared null experiment with `topology_gate.calibration.calibrate_null` and a
declared synthetic-shift experiment with `calibrate_shift`.

For `PersistentLaplacianCUSUM`, freeze the backend identity, cloud and spectrum
widths, Betti dimensions, prior-only calibration window, and forgetting map in
the study manifest. The controller's strict state and topology evidence
digests belong in the run artifact, but a passing synthetic shift is still not
authorization for market acceleration; the resulting null certificate must
bind to the controller identity and the exact dependence-preserving factory.
Its batch `detect(...)` facade can be passed directly to `calibrate_null` and
`calibrate_shift`; use `observe(...)` for the stateful causal run whose stream
state is checkpointed.

Each result records:

- detector identity and experiment configuration hash;
- finite trial count and horizon;
- false-alarm or detection rate;
- a 95% Wilson interval;
- censored average run length or detection delay; and
- the per-trial first-alarm/delay sequence.

The observation factory must encode the null or alternative explicitly. It is
called with a seeded NumPy generator, horizon, and feature count. The harness
does not block dependence, selection, or calibration leakage by itself; those
assumptions belong in the experiment manifest. A null result is evidence for
the tested finite horizon and factory only, not a universal guarantee for
market data.

`StationaryBlockBootstrap` provides a circular stationary block factory for a
finite source sequence. Its block length, restart probability, source digest,
and source identifier are included in the observation-factory identity that
the calibration result records. This preserves declared local serial
dependence for a null experiment; it does not prove that the source is the
right market null or remove selection effects.

`NullCalibrationResult.to_certificate(max_false_alarm_rate=...)` creates an
explicit finite-null certificate. The certificate is approved only when the
95% Wilson upper bound is no larger than the declared budget, and it is bound
to the detector and null-experiment identities. The causal numerical adapter
requires a matching approved certificate before it can use a detector factor
below its neutral maximum; without one, topology remains diagnostic and
forgetting stays neutral. The certificate's count, empirical rate, and Wilson
upper bound must be mutually consistent; hand-editing one field cannot create
an approved artifact. A certificate from a synthetic or dependent null is not
market calibration.

`calibrate_eprocess_null` is the corresponding finite simulation for the
promotion layer. It accepts a score factory that must return one-dimensional
scores in `[-1, 1]`, uses a predeclared constant `eta`, inspects the e-value at
every step, and stops each path at its first `1/alpha` crossing. The result
records the score-factory identity, threshold, crossing rate, Wilson interval,
and first-crossing sequence. This tests optional-stopping behavior of a
declared bounded stream; it cannot establish the required conditional-mean
null, validate a market utility, or replace an alpha-allocation and selection
audit.

`calibrate_promotion_null` runs the same finite experiment through the full
`PromotionGate`. Its score factory receives `(rng, horizon, challengers)` and
returns one bounded stream per registered challenger. Candidates are
pre-registered and the gate registration is explicitly sealed before
observations, their geometric alpha allocations are recorded, and each path
stops at the first gate promotion in registration order. A run may also
declare repeated gate epochs; paths without a promotion reset the gate and
spend the next preallocated geometric alpha slice. This makes
multi-challenger selection and repeated-testing spending visible in the
evidence. It still only tests the supplied finite score factory; it cannot
prove the conditional-mean null or market validity.

Minimum protocol for a research release:

1. Freeze detector configuration, source revision, dependency fingerprint, and
   seed schedule before looking at results.
2. Use a stationary/block null that preserves the serial and cross-sectional
   dependence relevant to the intended data.
3. Report both false-alarm probability and censored run length; do not report a
   threshold as `alpha` without this calibration evidence.
4. Use a separate shift family for power/delay, including shifts not used to
   tune the detector.
5. Keep the calibration report beside the walk-forward run manifest and spend
   challenger alpha separately from detector-threshold selection.
6. Run an optional-stopping e-process null simulation with the exact bounded
   utility and predictable eta rule used by the causal promotion path; report
   it as empirical evidence, not as a theorem or market guarantee.
7. When more than one challenger or gate epoch can compete, run the complete
   promotion-gate harness and report the challenger count, epoch count,
   per-slot alpha schedule, first promoted candidate, and selection rule
   alongside the crossing interval.
