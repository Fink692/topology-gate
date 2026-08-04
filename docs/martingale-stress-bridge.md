# Finite martingale stress bridge

`topology_gate.bridge` provides a bounded discrete stress-training primitive.
Given reference endpoint paths, it alternates KL projections to satisfy a
declared terminal-state marginal and conditional terminal drift for each
initial-state group. Zero drift gives a discrete martingale constraint; a
nonzero declared drift is appropriate when working under a physical-return
model rather than assuming discounted prices are martingales.

The result records path weights, terminal-marginal residuals, drift residuals,
relative entropy, iteration count, and a digest. Infeasible constraints or
failure to converge are errors. This is not a continuous-time
martingale-Schrödinger bridge, and it does not identify a pricing measure.

