# Endogenous Wasserstein robustness

`topology_gate.wasserstein` implements a finite online robust-learning
prototype. It uses absolute residual loss and the Euclidean 1-Wasserstein
upper-bound penalty

```text
|y - xᵀθ| + ρ √(1 + ||θ||²).
```

The ambiguity radius is selected at prediction time from a precomputed
instability score:

```text
ρ_t = clip(ρ_0 + c G_t, ρ_0, ρ_max).
```

The learner performs a clipped subgradient update. It never reads the current
target while deciding the radius. State is digest-bound and JSON serializable.

This is an explicit bounded-loss Wasserstein surrogate, not an adapted-
Wasserstein path solver and not evidence of robust market performance. A
market study must predeclare the ground metric, feature scaling, radius
mapping, clipping rule, and calibration split before opening the final holdout.

