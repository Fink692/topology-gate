# Calibration protocol

The topology detector's threshold is a policy parameter, not a theoretically
calibrated false-alarm level. Before using it to accelerate forgetting, run a
declared null experiment with `topology_gate.calibration.calibrate_null` and a
declared synthetic-shift experiment with `calibrate_shift`.

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
forgetting stays neutral. A certificate from a synthetic or dependent null is
not market calibration.

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
