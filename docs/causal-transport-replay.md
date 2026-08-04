# Causal transport replay

`topology_gate.transport` implements the first narrow transport-replay
prototype. A record can enter a decision-time batch only when both its source
step and label availability step are strictly before the requested decision.
The replay uses only the latest state estimate at or before that step:

\[
\tilde x_{i,t}=x_i+(\mu_t-\mu_i),\qquad
\tilde y_{i,t}=y_i+\tilde x_{i,t}^{\top}(\theta_t-\theta_i).
\]

Reliability is `exp(-displacement)`, bounded below by the declared minimum
weight. State estimates and feature locations are caller-supplied and must be
prefix-only. The implementation is a causal location translation and linear
parameter correction; it is not an adapted-Wasserstein or bicausal optimal
transport solver.

The replay identity, record/state limits, availability boundaries, and
checkpoint tamper detection are all explicit. Use it as a research adapter
under a declared model, not as a market-data validity guarantee.

Reproduce the bounded experiment with:

```powershell
$env:PYTHONPATH = 'src'
py -3.12 examples\causal_transport_replay.py
```
