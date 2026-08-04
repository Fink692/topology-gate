# Mechanism-localized continual learning

`topology_gate.mechanisms` is a dependency-light prototype for a modular
recursive learner. Each declared mechanism owns a feature slice and scalar
target. A prefix-only residual monitor uses the historical median and MAD
scale for that mechanism. If a residual exceeds the predeclared threshold,
the mechanism is marked shifted.

The update policy is:

- with no shifted mechanism, all modules receive the stable forgetting factor;
- during a localized shift, only shifted modules update, using the faster
  shift forgetting factor;
- unchanged modules remain frozen until the current transition is no longer a
  localized shift.

This tests the proposed “update changed mechanisms, not the whole model”
control idea. It does not identify causal mechanisms from observational market
data. The partition, feature ownership, targets, residual scale, and threshold
must be pre-registered and validated on data strictly before the final
evaluation window.

Run the synthetic diagnostic with:

```text
.venv\Scripts\python.exe examples\mechanism_localized_continual_learning.py
```

The output is a diagnostic receipt. It should show `transmission` as the
shifted module after the synthetic coefficient change and show that the
`volatility` learner was not updated on the first localized transition.

The state is JSON serializable and digest-bound. The digest protects the
configuration, learner states, step counter, and residual histories from
accidental mutation during checkpoint or replay restoration.

