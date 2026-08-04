# Adaptive rough-path memory

`topology_gate.signatures` computes truncated Chen signatures of a causal
piecewise-linear increment path. It maintains one recursive ridge learner per
predeclared signature depth and evaluates every depth on every settled target.
The next depth is selected only after the target settles, using clipped
squared loss and a declared switching cost.

This makes the memory decision explicit:

- signature depth is selected at a decision boundary;
- only the last declared window of increments is visible;
- candidate learners receive the same settled target as shadow experts;
- future increments cannot change an earlier signature or prediction.

The implementation is a finite path-signature prototype. It is not a claim
that any depth is economically useful, and it does not replace a preregistered
study of scaling, roughness, missing observations, and transaction costs.

