# Strict economic evaluation boundary

`topology_gate.economic` is intentionally narrower than the compatibility
backtest. It evaluates an already emitted position path only when the caller
supplies separate, timestamped evidence for:

- each observed realized return;
- each execution-cost rate (fee, spread, slippage, impact, and other);
- the decision, realization, execution, and availability times; and
- the cost-model identity.

The evaluator never treats a supervised target as a tradable return and never
turns missing returns or costs into zero. An abstention is retained as a row
with an explicit reason. An abstention after a non-zero position is rejected
unless the caller supplies an explicit execution row that closes the position;
this prevents hidden turnover and cost undercounting.

The cost fields are non-negative return-space rates per unit position turnover.
The evaluator computes component costs as `abs(position - previous_position)`
times each rate, including the configured initial position. This is an
accounting contract, not a bid/ask, impact, capacity, borrow, or liquidity
model. A market study must source and version those rates and attach the
universe, delistings, instrument units, capacity assumptions, and final-holdout
manifest separately.

Example:

```python
from topology_gate import (
    EconomicDecision,
    EconomicEvaluationConfig,
    ExecutionCost,
    RealizedReturn,
    evaluate_economic_path,
)

result = evaluate_economic_path(
    decisions=(EconomicDecision("d1", "target-1", 1, 0.5),),
    realized_returns={
        "target-1": RealizedReturn("target-1", 1, 2, 3, 0.01),
    },
    execution_costs={
        "target-1": ExecutionCost(
            "target-1", 1, 1, 1, "rates-v1", spread_rate=0.0002
        ),
    },
    config=EconomicEvaluationConfig(cost_model_id="rates-v1"),
)
```

The result is suitable for a reproducible accounting report. It is not, by
itself, evidence of predictive skill, calibrated risk, profitability, or
capacity.
