# Heavy-tail expert allocation diagnostic

Date: 2026-08-04  
Status: bounded methodological diagnostic; not market or economic evidence

The experiment used 256 full-information utility rows, known change points at
128 and 192, four seeds (`11, 17, 23, 29`), Student-t noise with two degrees
of freedom, Catoni scale `0.08`, and a switching cost of `0.01`. Every expert's
utility was observed after the current action; the selected expert was used on
the next row.

| allocator | mean net utility | mean switches |
|---|---:|---:|
| Ordinary mean with reset | 32.9105 | 6.5 |
| Catoni robust with reset | 33.0241 | 5.0 |

The robust-minus-mean differences by seed were `0.1104`, `0.0000`, `0.7528`,
and `-0.4088`. The aggregate improvement is therefore a small, mixed synthetic
diagnostic rather than evidence of superiority. It does not establish a
heavy-tail regret guarantee, a conditional-mean e-process null, transaction
cost realism, capacity, or financial performance.

The allocator's state is checkpointed with a configuration identity, per-expert
history, current expert, and step counter. A change-point signal can clear
historical estimates before settling the new row; the signal itself must come
from a separately authorized detector in a full study.
