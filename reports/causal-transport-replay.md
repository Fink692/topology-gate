# Causal transport-replay synthetic experiment

Date: 2026-08-04  
Status: bounded prototype diagnostic; not market or economic evidence

The experiment used 256 observations, shifts at 64, 128, and 192, four seeds,
and two-step delayed labels. A raw causal RLS path was compared with a
transported replay batch using prefix-only parameter and feature-location
states. The final regime `[192, 256)` was treated as the held-out diagnostic.

| system | mean final-regime MSE |
|---|---:|
| Raw causal RLS | 0.0436627 |
| Causal transport replay | 0.0502417 |
| replay minus raw | +0.0065790 |

The prototype lost on average in this fixture, improving one of four seeds.
That is a useful negative result: the transport correction is not accepted as
an improvement merely because it is mathematically newer. The result does not
test adapted Wasserstein transport, market data, transaction costs, or
economic utility.

Every path stored 256 records and 256 prefix states. The per-path replay
identities are recorded in the example output and are bound to the exact
transport configuration.
