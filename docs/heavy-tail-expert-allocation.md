# Heavy-tail-aware expert allocation

`topology_gate.experts` implements the fourth research direction from the
proposal: full-information allocation among recursive strategy experts using
a Catoni location estimate, a declared switching-cost penalty, and optional
history resets at a detected change point.

The implementation assumes that shadow utility for every expert is available
at the same settlement boundary. It selects the expert for the next boundary,
so the current observation cannot affect its own action. The Catoni scale,
expert family, history limit, reset policy, and switching cost are part of the
configuration identity and checkpoint state.

The bounded synthetic diagnostic used four seeds, two known changes, and
Student-t noise with two degrees of freedom. Robust allocation averaged net
utility `33.0241` versus `32.9105` for an otherwise identical ordinary-mean
allocator, a difference of `0.1136`; it averaged 5.0 switches versus 6.5.
One of four paths favored the ordinary mean. This is a small control-layer
diagnostic, not a regret bound, market result, or promotion certificate.

Reproduce with:

```powershell
$env:PYTHONPATH = 'src'
py -3.12 examples\heavy_tail_expert_allocation.py
```
