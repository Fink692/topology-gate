# Quantitative Topology Specialist Review

**Review date:** 2026-08-04
**Scope:** `src/topology_gate/topology.py`, `tests/test_topology.py`,
`docs/architecture.md`, `docs/agent-contracts.md`,
`docs/statistical-validity.md`, and the prior statistical reviews.  `pyproject.toml`
and the public/configuration boundary were checked only to assess integration
constraints.

## Disposition

The current implementation is an honest and useful deterministic graph-spectrum
approximation.  It is not a Vietoris–Rips filtration, persistent homology
implementation, or persistent Laplacian.  The optional
`persistent_laplacian_backend` is only a callable escape hatch: it receives
`(canonical_cloud, n_eigenvalues)` and returns eigenvalues.  That interface does
not carry enough information to specify a reproducible persistent-topology
calculation.

An exact backend is feasible, but it should be introduced as a separately
specified, bounded reference backend.  “Exact” should mean exact for a declared
finite Vietoris–Rips construction and exact coefficient-field persistence
pairing.  Persistent-Laplacian eigenvalues are generally real numerical
quantities; a float64 eigensolver supplies deterministic reference values with
residual/tolerance evidence, not exact symbolic real numbers.

Recommendation: retain the current method ID
`knn_normalized_laplacian_approximation`, do not silently replace it, and add a
versioned topology contract before adding an exact implementation.  If an exact
backend exceeds a resource or numerical bound, it must return `DEGRADED` or
`INSUFFICIENT_HISTORY` and prevent an accelerated-forgetting/gate decision.  It
must not truncate the complex and return a plausible-looking spectrum.

## What the current code establishes

The implementation has several sound engineering properties:

- finite-input and bounded-size validation;
- causal delay-cloud construction;
- deterministic coordinate canonicalization before the graph spectrum;
- a symmetric union-kNN normalized graph Laplacian;
- finite warm-up values with explicit `valid` and `calibrated` masks;
- a callable seam that makes the non-persistent default claim visible.

The tests establish determinism, prefix causality, permutation stability,
finite outputs, a pure-Python eigensolver path, and visibility of the backend
seam.  They do not establish filtration correctness, persistence pairs,
homology, persistent-Laplacian construction, null calibration, or a calibrated
alarm interpretation.  This agrees with the prior statistical reviews and the
disclaimer in `docs/statistical-validity.md`.

The exact current boundary is important:

| Current behavior | Consequence for an exact backend |
| --- | --- |
| `spectral_summary` calls `backend(canonical_cloud, n_eigenvalues)` | No metric, homology dimensions, coefficient field, filtration cutoff/grid, pair `(s,t)`, or spectral normalization is specified. |
| `point_cloud_features` normalizes and then calls `spectral_summary` with `assume_normalized=True` | A new backend must not inherit the current double-normalization/scale metadata ambiguity. |
| `cloud_window` allows up to 1,024 points and point dimension up to 4,096 | A full VR complex has at most `sum(C(n,k+1))` simplices through dimension `k`; these limits are not safe exact-VR limits. |
| `SpectralSummary` assumes a short sorted spectrum, pads with `2.0`, clips the default spectrum to `[0,2]`, and calls the second value algebraic connectivity | Those semantics are for the normalized graph approximation, not for unnormalized higher-order persistent Laplacians. |
| The backend returns only eigenvalues | No persistence intervals, Betti/nullity, filtration digest, solver residual, status, or resource diagnostics can be audited. |
| The cloud is coordinate-only and point order is sorted coordinates | Production point identity, duplicate policy, timestamps, and as-of provenance are unavailable at the backend boundary. |
| Backend identity is a callable module/name | It is not an algorithm version, normalized configuration digest, dependency identity, or numerical-policy identity. |

The architecture and agent contracts already require the missing choices:
metric, filtration, coefficient field, homology dimensions, tie rules,
duplicate policy, matrix convention, convergence behavior, bounded complex
size, and a reference state committed strictly before the current cloud.

## Mathematical scope to implement

### 1. Finite bounded Vietoris–Rips filtration

Choose and version all of the following before coding:

- metric: recommended first version is Euclidean distance on one causally
  defined, robustly scaled point cloud;
- vertex identity and duplicate policy: production input should include stable
  point IDs.  If equal coordinates are retained, treat distinct IDs as
  distinct vertices at distance zero; if they are coalesced, record the map.
  Coordinate-only occurrence indices are acceptable only for a test adapter;
- `max_homology_dimension`, with simplices through
  `max_homology_dimension + 1` so deaths in the requested dimension can be
  observed;
- `max_radius`: full pairwise diameter for an uncensored finite cloud, or a
  declared cutoff with right-censored intervals;
- `max_simplices`, `max_boundary_nonzeros`, and matrix/eigensolver limits;
- filtration scale policy: all distinct pairwise distances is the exact finite
  VR event set.  A coarse grid is a different, grid-discretized method and must
  be named as such;
- coefficient field for persistence, recommended as fixed `F_2` in version 1;
- total order for equal filtration values: `(birth_radius, dimension,
  canonical_vertex_tuple)` is simple and ensures faces precede cofaces at a tie.

For a finite metric point set (P), a simplex σ is in

\[
  \mathrm{VR}_\varepsilon(P)\quad\Longleftrightarrow\quad
  \max_{u,v\in\sigma} d(u,v)\leq\varepsilon.
\]

Its birth value is its diameter.  Vertices have birth zero.  The following is
reference pseudocode; the actual enumerator should use clique extension and
branch-and-bound rather than blindly materializing all combinations.

```text
build_vr_filtration(points, cfg):
    P = canonicalize_points(points, point_id_order=cfg.point_id_order)
    require finite coordinates and len(P) <= cfg.max_vertices

    D = symmetric_distance_matrix(P, metric=cfg.metric)
    # Compute each i < j once, use a fixed coordinate accumulation order,
    # copy D[j,i] = D[i,j], and record the distance-policy digest.

    simplices = []
    for v in canonical_vertices(P):
        append(simplex(vertices=(v,), dim=0, birth=0.0))

    for dim in 1 .. cfg.max_homology_dimension + 1:
        for vertices in enumerate_cliques(
                D,
                size=dim + 1,
                radius=cfg.max_radius,
                vertex_order=cfg.point_id_order):
            birth = max(D[i,j] for i,j in pairs(vertices))
            if birth <= cfg.max_radius:
                append(simplex(vertices=vertices, dim=dim, birth=birth))
                if len(simplices) > cfg.max_simplices:
                    return DEGRADED(RESOURCE_LIMIT)

    sort simplices by (birth, dim, vertices)
    assign stable simplex_id = position in that sorted order
    verify every face has a lower simplex_id and is present
    build boundary adjacency for every simplex dim > 0
    return Filtration(simplices, D, cfg.digest)
```

The check must be `<=`, not `<`; otherwise exact zero distances and equal
distance ties are changed.  No approximate distance binning belongs in the
reference path unless it is explicitly part of the algorithm version.

### 2. Persistence intervals

Use a separate exact coefficient-field calculation.  With `F_2`, an oriented
boundary is represented as a sorted bit set and column addition is XOR.  A
standard reduced-boundary algorithm is sufficient for a bounded reference
implementation:

```text
compute_intervals(F, requested_dims):
    reduced = {}                 # simplex_id -> reduced boundary bitset
    pivot_column = {}            # low row -> column simplex_id
    zero_columns = set()
    death = {}                   # birth simplex_id -> death simplex_id

    for j, sigma in enumerate(F.simplices):
        column = boundary_bitset(sigma, F)       # empty for vertices
        while column is not empty:
            p = largest_set_bit(column)
            prior = pivot_column.get(p)
            if prior is None:
                break
            column = column XOR reduced[prior]

        if column is empty:
            zero_columns.add(j)                   # candidate class birth
        else:
            p = largest_set_bit(column)
            pivot_column[p] = j
            reduced[j] = column
            death[p] = j

    intervals = {q: [] for q in requested_dims}
    for b in sorted(zero_columns):
        q = F.simplices[b].dim
        if q not in intervals:
            continue
        if b in death:
            d = death[b]
            intervals[q].append(
                (F.simplices[b].birth, F.simplices[d].birth, b, d)
            )
        else:
            intervals[q].append((F.simplices[b].birth, +infinity, b, None))

    sort each interval list by (birth, death, birth_simplex_id)
    return Persistence(intervals, coefficient_field=F_2,
                       filtration_digest=F.digest)
```

Intervals use the usual half-open convention `[birth, death)`.  Zero-length
intervals may be retained in the audit artifact and excluded only by an
explicit summary policy.  An `infinity` endpoint means “not killed in the
computed filtration”; it is an essential class only when the full relevant
complex was computed.  A radius cutoff or simplex cap must mark the result
right-censored.  The algorithm must not call every uncensored H_q bar a true
essential class when the complex was deliberately truncated.

For `F_2`, the intervals are exact for the ordered finite complex.  They are
not field-independent statements: the coefficient field is part of the public
topology identity.  The construction also needs the usual boundary checks

\[
  \partial_q\partial_{q+1}=0
\]

over the selected field before any intervals are accepted.

### 3. Persistent Laplacian for one filtration pair

Persistence intervals and persistent-Laplacian spectra are related but are not
the same calculation.  For a pair (K_s \subseteq K_t), use an explicitly
oriented real chain complex with the standard simplex inner product.  Let

- (B_q^s) be the oriented boundary matrix (C_q(K_s)\to C_{q-1}(K_s));
- (B_{q+1}^t) be the oriented boundary matrix in (K_t), with rows ordered
  by q-simplices of (K_t);
- (E) select the q-simplices in (K_t\setminus K_s).

The allowed ((q+1))-chains are those whose boundary has no component outside
`K_s`.  If columns of `Z` are a basis of `null(E B_{q+1}^t)`, and `I` selects
q-simplices in `K_s`, then (A=I B_{q+1}^t Z) and

\[
  \Delta_q^{s,t}
  = (B_q^s)^\mathsf{T}B_q^s
    + A(Z^\mathsf{T}Z)^{-1}A^\mathsf{T}.
\]

An orthonormal basis `Q` may be used instead, giving the up term `A A^T`.
The nullity of this correctly constructed operator is the persistent Betti
number for the pair under the stated assumptions.  The nullspace restriction
is essential: taking an arbitrary submatrix of a boundary matrix and
multiplying it by its transpose is not, in general, a persistent Laplacian.

```text
persistent_laplacian(F, q, scale_s, scale_t, cfg):
    Ks = subcomplex_at(F, scale_s)
    Kt = subcomplex_at(F, scale_t)       # require Ks subset Kt

    Sq = ordered_simplices(Ks, dimension=q)
    Bq = oriented_boundary(Ks, dimension=q)          # rows q-1, cols Sq
    Bt = oriented_boundary(Kt, dimension=q+1)        # rows q in Kt

    rows_in  = q_simplices(Kt) intersect Sq
    rows_out = q_simplices(Kt) minus Sq
    E = Bt[rows_out, :]
    Z = deterministic_nullspace(E, method=cfg.nullspace_method)
    validate_residual(E @ Z, cfg.nullspace_tolerance)

    A = Bt[rows_in, :] @ Z
    if number_of_columns(Z) == 0:
        up = zero_matrix(len(Sq), len(Sq))
    else:
        G = Z.T @ Z
        up = A @ solve_spd(G, A.T)       # do not form an explicit inverse

    down = Bq.T @ Bq
    Delta = symmetrize(down + up)
    eigenvalues, eigenvectors, residual =
        deterministic_symmetric_eigensolve(Delta, cfg.solver_policy)
    require residual <= cfg.eigen_residual_tolerance
    require min(eigenvalues) >= -cfg.negative_roundoff_tolerance
    return PersistentSpectrum(q, scale_s, scale_t, eigenvalues,
                              residual, cfg.operator_digest)
```

`Z` can be obtained by deterministic rational/fraction-free elimination and
then converted to float64, or by a pinned SVD/QR path with a documented rank
tolerance.  The first option gives the cleanest finite-complex reference
semantics; the second is acceptable only with rank, residual, and conditioning
diagnostics.  Use increasing vertex order for simplex orientation.  If
eigenvectors are not public outputs, do not expose their unstable signs.

The spectrum is an unnormalized higher-order combinatorial spectrum.  It is
not bounded by `[0, 2]`, and its matrix dimension is the number of q-simplices
in `K_s`, not the number of points.  A valid summary should therefore carry,
at minimum:

```text
(q, scale_s, scale_t, coefficient_field, operator_digest,
 sorted_eigenvalues, zero_multiplicity, first_positive_eigenvalue,
 trace, positive_spectrum_entropy, residual, numerical_tolerance,
 right_censored, status)
```

`zero_multiplicity` is a numerical nullity under the declared tolerance.  It
may be compared with the exact persistence Betti number as a consistency
check, but it is not an inferential confidence measure.  “Algebraic
connectivity” should be reserved for a clearly specified q=0 spectrum; for
general q use `first_positive_eigenvalue`.  Do not pad missing eigenvalues with
`2.0`; return a validity mask or a fixed slot definition with explicit
missingness.  Do not combine spectra from different q or different `(s,t)`
pairs into one unlabelled vector.

## Required package-boundary changes

The current callable seam can remain as a compatibility adapter, but it cannot
be the exact backend contract.  The target protocol should receive an
immutable `PointCloudWindow` and a frozen `PersistentTopologyConfig`, plus a
reference state committed before the current anchor, and return a typed
`TopologyObservation`.  The configuration needs fields equivalent to:

```text
metric_id, metric_parameters, duplicate_policy,
max_vertices, max_radius, max_homology_dimension,
coefficient_field, simplex_order, max_simplices,
max_boundary_nonzeros, selected_scales_or_pairs,
operator_inner_product, nullspace_method, rank_tolerance,
eigensolver, eigen_residual_tolerance, negative_eigenvalue_tolerance,
status/degradation policy, algorithm_version
```

The observation should include the filtration digest, canonical point IDs,
interval digest or bounded interval artifact, Betti/nullity summaries, selected
spectra, residuals, condition/resource diagnostics, and a status.  The backend
identity must hash the normalized configuration, algorithm version, numerical
backend/version, precision, thread policy, and solver tolerances.  A callable
qualname alone is not sufficient for replay identity.

Integration also needs an explicit decision about scale.  The current code
fits robust location/scale separately for each cloud (and the feature path
normalizes before the spectral call).  A persistent spectrum can then change
because coordinates were rescaled, even when the underlying shape is the
same.  For comparable rolling observations, either use a past-only transform
state and record its fingerprint, or declare the per-cloud normalization as
part of the metric and interpret every spectrum conditionally on that changing
metric.  The former is preferable for a change detector.

The topology reference must be updated only after the current observation has
been scored.  Exact backend state, transform state, configuration identity,
and any interval/spectrum digest that affects future decisions must be in the
checkpoint.  Backend/resource failure must preserve the last valid reference
and prohibit a SHIFT/accelerated-forgetting decision.

## Implementation risks

1. **Combinatorial explosion.**  A full VR complex has
   (\sum_{j=0}^{D}\binom{n}{j+1}) simplices through dimension `D`; boundary
   reduction can have substantial fill-in, and a dense q-Laplacian eigensolve
   is cubic in the number of q-simplices.  The existing 1,024-point cloud cap
   is not an exact-VR budget.  Start with an exact-backend vertex cap in the
   tens, a low homology dimension, and a hard simplex/matrix budget.
2. **Truncation can create false topology.**  A cutoff or dimension cap makes
   intervals right-censored.  A resource limit must be a typed outcome, never
   an implicit subsampling or partial complex.
3. **Tie and duplicate semantics.**  Zero distances, equal edge lengths, and
   simultaneous simplex births affect interval pairings and boundary ordering.
   Point IDs, distance equality policy, simplex order, and interval endpoint
   convention must be versioned.
4. **Coefficient-field/operator mismatch.**  `F_2` persistence and real
   oriented Laplacians are compatible components but not interchangeable
   matrices.  A field change can change Betti numbers; an inner-product change
   can change eigenvalues.
5. **Numerical rank and zero tests.**  Persistent nullity is exact in theory;
   float64 nullity depends on rank and eigenvalue tolerances.  Report residuals,
   condition estimates, and negative-roundoff repairs.  A repaired value is
   not automatically a valid topology result.
6. **Rolling-scale comparability.**  Per-cloud robust normalization, clipping,
   and overlapping windows change the metric.  Exact construction does not
   make the resulting CUSUM or forgetting rule calibrated.
7. **Repeated work.**  `detect` recomputes a bounded prefix and each valid cloud
   is freshly processed.  A persistent backend may dominate runtime in
   `observe`; caching or incremental updates must be proven equivalent and
   included in the state identity.
8. **API and summary mismatch.**  One `n_eigenvalues` value cannot represent
   multiple homology dimensions and filtration pairs.  A richer result must be
   versioned rather than squeezed into the existing graph-summary fields.
9. **Backend reproducibility.**  Third-party reductions, BLAS threading, SVD
   rank decisions, and eigenvalue ordering can vary.  The release path needs a
   pinned CPU reference environment and explicit cross-platform tolerances.
10. **Statistical overclaim.**  Exact finite topology only removes one
    implementation ambiguity.  The rolling robust transform, adaptive
    whitening, reflected CUSUM, and score-to-forgetting map remain exploratory
    until null calibration and dependence assumptions are separately addressed.

## Golden-test cases

The topology gate should have independent mathematical goldens, not only
round-trip tests against the implementation under test.

| Case | Required golden evidence |
| --- | --- |
| Empty cloud, one vertex, and two identical-coordinate vertices | Typed insufficient/valid status; vertex births at 0; duplicate policy is visible; zero-distance edge behavior is deterministic. |
| Three collinear points at `0, 1, 3` | H0 intervals: one `[0,1)`, one `[0,2)`, one `[0,∞)` in the uncensored full 1-skeleton; no H1 interval. |
| Equilateral triangle of side 1 | At radius 1, all edges and the 2-simplex enter under the documented tie order; H0 has two finite `[0,1)` bars and one essential bar; any zero-length H1 bar is handled by the declared policy. For the filled triangle with `K=L`, oriented q=0 spectrum is `[0,3,3]`, q=1 is `[3,3,3]`, and q=2 is `[3]`. |
| Four vertices of a unit square | H0 has three `[0,1)` bars and one essential bar; H1 has `[1,√2)` for the full VR filtration. At the 1-skeleton, q=0/q=1 cycle spectra are `[0,2,2,4]`; at the complete graph at `√2`, q=0 is `[0,4,4,4]`. |
| Triangle boundary `K` into filled triangle `L` | Persistent H1 rank is zero; the q=1 persistent Laplacian has zero nullity and the hand-oriented spectrum `[3,3,3]`. This catches the incorrect “submatrix times transpose” construction. |
| Permutations and equal-distance ties | All permitted point permutations produce the same canonical filtration digest, intervals, ordered spectra, and status. Near-ties follow the exact configured policy rather than an accidental sort or set order. |
| Boundary identity | For every generated complex, verify `∂q ∂q+1 = 0` over `F_2` and with oriented integer boundaries. |
| Pair monotonicity | `K_s ⊆ K_t`; persistent Betti/nullity is computed for every configured pair; changing `t` cannot remove a simplex from the pair’s ambient complex. |
| Radius and simplex cap | The backend returns right-censored/degraded status with no spectrum when a cap is exceeded; it never silently drops simplices. |
| Solver failure and ill-conditioning | Inject non-convergence, rank ambiguity, and negative-roundoff cases; verify typed status, unchanged reference state, and no accelerated gate. |
| Causal reference | Alter future observations or future revisions and verify every earlier filtration digest, interval digest, spectrum, topology status, and gate record is unchanged. The current cloud is not used to update the reference used to score itself. |
| Persistence/checkpoint | Fresh and restored runs produce identical canonical artifacts, configuration/backend identities, statuses, and selected spectra under the pinned reference environment. |
| Independent oracle | Compare small cases to a separately implemented boundary reduction and a separately implemented persistent-Laplacian matrix/eigensolve. A third-party library may be a test oracle, but must not be the only oracle. |

These tests should also assert that the exact method is never reported as the
current `knn_normalized_laplacian_approximation`, and that graph-summary
assumptions such as `[0,2]` clipping and `2.0` padding are not applied to the
persistent operator.

## Recommendation and release gate

1. **Keep the alpha default unchanged.** Its current name and documentation are
   correct. Do not call its output a Betti number, persistence diagram, or
   persistent-Laplacian spectrum.
2. **Approve a topology ADR before implementation.** Fix metric, point identity,
   duplicates, `F_2`, homology dimensions, radius/event policy, cap behavior,
   simplex order, real inner product, solver/tolerances, summary slots, and
   status semantics.
3. **Build a small exact reference first.** Use exact `F_2` reduction and a
   deterministic CPU float64 persistent-Laplacian path for small clouds. Make
   the independent goldens above pass before optimizing enumeration, sparse
   algebra, or incremental rolling updates.
4. **Expose it as an optional/versioned backend.** A new protocol and result
   type are preferable. A compatibility adapter may select one configured
   `(q,s,t)` spectrum, but it must not pretend that one eigenvalue sequence is
   the complete persistent result.
5. **Set a lower exact-workload bound.** If the current cloud exceeds the exact
   backend cap, return a visible degraded/abstain status. Use a deterministic
   subsample or sparse approximation only as a separately named method.
6. **Calibrate the downstream detector separately.** An exact backend does not
   validate the current CUSUM threshold, alarm probability, forgetting policy,
   or promotion/e-process claims. Those still require the null, dependence,
   optional-stopping, and selection evidence demanded by the prior reports.

The release recommendation is therefore **research-only for now**. The exact
backend can become a topology-correct numerical component after the finite
construction, typed status/diagnostic contract, goldens, resource benchmark,
and replay evidence pass. It still should not be marketed as a calibrated
change detector or validated trading signal without the separate statistical
work.

## References

- [Mémoli, Wan, and Wang, *Persistent Laplacians: Properties, Algorithms and Implications*](https://doi.org/10.1137/21M1435471).
  The nullity result and the nullspace-restricted persistent-Laplacian
  construction are the relevant mathematical reference.
- [Bauer, *Ripser: efficient computation of Vietoris–Rips persistence barcodes*](https://doi.org/10.1007/s41468-021-00071-5).
  The simplexwise VR ordering and boundary-reduction conventions are useful
  implementation references.
- Local contract sources: `docs/architecture.md`,
  `docs/agent-contracts.md`, and `docs/statistical-validity.md`.
