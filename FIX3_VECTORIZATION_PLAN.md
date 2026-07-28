# Fix 3: vectorize TF maps across datasets — design and plan

Preparation document for the structural fix from
`MEMORY_OPTIMIZATION_HANDOFF.md` (item 3). Written 2026-07-28 on
branch `feature_tf_memory_optimization` after fix 1 (cheap Hessian
fixes) was implemented and item 2 (traceback stripping) was measured
and discarded. Baseline to beat: 5.3 GB retained after tracing
`neg_log_prob_and_gradient`, 7.9 GB after one eager Hessian.

## Why the graph is O(num datasets) today

`CompoundMap` delegates to 10 map classes; each one loops over
reactions/datasets in `_prepare_propagate` and registers one closure
(`_atomic_propagate`) per dataset via `_add_lists`
(`gmapy/mappings/tf/cross_section_base_map_tf.py`). Propagation runs
selector-gather -> closure -> `Distributor` scatter per closure and
`tf.add_n`s 227 outputs; every closure instantiates its own
`PiecewiseLinearInterpolation` whose `searchsorted`/`gather`/arith ops
are duplicated into the graph (~150 fwd + ~600 bwd ops each).
`tape.jacobian(..., experimental_use_pfor=True)` additionally builds
and caches a pfor graph per closure during eager Hessian evaluation.

## Map class inventory (227 atomic functions, a205d16 evaluation)

With `I` a piecewise-linear interpolation operator from a reaction's
prior mesh to the dataset's energies (weights fixed at construction),
`n_d` a per-dataset normalization parameter, `x_r` the parameter
sub-vector of reaction r:

| class (MT) | #fns | per-point formula |
|---|---|---|
| CrossSectionMap (1) | 33 | `I x_r` |
| CrossSectionShapeMap (2) | 52 | `n_d * (I x_r)` |
| CrossSectionRatioMap (3) | 20 | `(I1 x_a) / (I2 x_b)` |
| CrossSectionRatioShapeMap (4) | 57 | `n_d * (I1 x_a) / (I2 x_b)` |
| CrossSectionTotalMap (5) | 2 | `sum_k I_k x_k` |
| CrossSectionAbsoluteRatioMap (7) | 2 | `(I1 x_a) / (I2 x_b + I3 x_c)` |
| CrossSectionShapeOfSumMap (8) | 4 | `n_d * sum_k I_k x_k` |
| CrossSectionShapeOfRatioMap (9) | 47 | `n_d * (I1 x_a) / (I2 x_b + I3 x_c)` |
| CrossSectionFissionAverageMap (6) | 2 | SACS: normalized spectrum-averaged integral |
| CrossSectionRatioOfSacsMap (10) | 8 | ratio of two SACS integrals |

Modifiers (applied on top of propagated values):
`RelativeErrorMap` (`y_i * x_relerr(i)`, artificially chunked in
loops of 100) and `EnergyDependentUSUMap`.

## Unified algebraic form (covers 217 of 227 functions)

Everything except MT:6/10 is one expression over the full parameter
vector `x` (length n) producing all m experimental points at once:

    y = (N x + g0) * (A x) / (B x + b0)        (elementwise)

- `A` (m x n sparse): numerator interpolation. Each row has <= 2
  nonzeros per contributing reaction (sum maps MT:5/8 put several
  reactions' entries in the same row — replicating `tf.add_n`).
- `B` (m x n sparse): denominator interpolation for ratio-type rows
  (MT:3/4/7/9; MT:7/9 rows contain entries of two reactions);
  zero rows elsewhere. `b0` in {0,1}^m: 1 exactly where B's row is
  empty, so the denominator degenerates to 1.
- `N` (m x n sparse): one nonzero (value 1) per shape-type row
  (MT:2/4/8/9) in the column of the dataset's `norm_` parameter;
  `g0` in {0,1}^m: 1 where there is no normalization parameter.

All three matrices are compile-time constants; only `x` is a
variable. Propagate = 3 `tf.sparse.sparse_dense_matmul` + a handful
of elementwise ops, independent of the number of datasets. The
backward pass is equally small. Expected: 206k graph ops -> O(100).

### Analytic Jacobian (no tape, no pfor)

With `num = A x`, `den = B x + b0`, `g = N x + g0`:

    J = diag(g / den) A  -  diag(y / den) B  +  diag(num / den) N

i.e. three COO value-scalings (row-indexed gathers on the tensors
above) concatenated and coalesced — reuse
`coalesce_sparse_matrices` from fix 1. `tape.jacobian` and pfor are
dropped entirely for these 217 functions; this is what actually
removes the +2.6 GB of pfor caches AND most of the traced graph.

### SACS maps (MT:6/10, 10 functions) stay as-is initially

Their integrals are nonlinear in the spectrum parameters (log-log
segments) and there are only 10 of them (~5% of the graph). Phase 1
keeps the existing per-dataset path for them (including their
`tape.jacobian`); vectorizing them later via shared integration
grids + `segment_sum` is optional polish.

## Implementation steps

1. DONE numpy weight builder: `piecewise_linear_interp_matrix`
   (`gmapy/mappings/tf/vectorized_helpers.py`) replicates
   `PiecewiseLinearInterpolation` semantics (unsorted meshes,
   clipping, outside-range -> 0) as COO triplets;
   tested against the TF class in
   `tests/test_vectorized_interp_matrix.py`.
2. DONE block emitters: instead of duplicating the pandas selection
   logic, each of the 8 algebraic map classes attaches a
   vectorization spec (`roles`/`src_ens`/`tar_en` per operand) to the
   dataset lists it already registers via `_add_lists`, and a single
   generic `CrossSectionBaseMap.vectorized_blocks()` turns the specs
   into per-dataset COO blocks in global indices (`num`/`den`
   triplets + `norm_col`). SACS maps have no spec and raise
   NotImplementedError. Parity proven in
   `tests/test_vectorized_blocks.py` (blocks vs `propagate`, 1e-12,
   per map class on the test database).
3. DONE `VectorizedCompoundMap`
   (`gmapy/mappings/tf/vectorized_compound_map_tf.py`): subclass of
   `CompoundMap` overriding only `_orig_propagate`/`_orig_jacobian`,
   so modifier maps, `reduce` handling and the interface are
   inherited. Builds A, B, N, b0, g0 as constants from the step-2
   blocks (duplicate COO entries coalesced in numpy at
   construction); SACS maps fall back to their per-dataset path; a
   construction-time check asserts every target row is claimed by at
   most one dataset (the algebraic form cannot represent sums into
   a shared row). Jacobian = analytic row-scaled COO, merged with
   the legacy parts via `coalesce_sparse_matrices` — no tape/pfor.
   Parity proven in `tests/test_vectorized_compound_map_tf.py`
   (propagate for reduce=True/False, Jacobian vs the tape-based one,
   gradient under tf.function).
4. LARGELY DONE parity + real-size measurement (2026-07-28): on the
   a205d16 priortable/exptable, propagate/gradient/Jacobian checksums
   of `VectorizedCompoundMap` and `CompoundMap` agree to the last
   printed digit. Memory/time for `sum(propagate)` + gradient under
   `tf.function` and two eager `jacobian` calls:
   - CompoundMap: +2.5 GB / 75 s tracing; eager jacobian ~91 s per
     call, ~+1.4 GB retained (pfor caches), peak 3.0 GB.
   - VectorizedCompoundMap: +0.24 GB / 5.4 s tracing; eager jacobian
     ~8 s per call (dominated by the legacy SACS datasets), no
     memory growth.
   Jacobian nnz differs slightly (19312 vs 19315): the numpy builder
   drops explicit zero entries that the tape-based path stores.
   Still open from this step: end-to-end posterior numbers (needs
   step 5, the pickles bake in closures over the old map).
5. Swap-in: point the evaluation setup (parent repo
   `evaluation/01_model_preparation.py`, `RestrictedMap` wrapper) at
   the vectorized map and rebuild the pickles; the pickled `post`
   holds closures over the old CompoundMap, so re-running model
   preparation is required to see the effect in the pipeline.
6. Later (optional): vectorize MT:6/10 (concatenated integration
   grids + `segment_sum`, analytic derivative of the log-log segment
   integrals), vectorize `EnergyDependentUSUMap`, and remove the
   100-chunking in `RelativeErrorMap._prepare_propagate`.

## Semantic quirks found while reading (preserve or resolve)

- `PiecewiseLinearInterpolation.__call__` for a single-point mesh
  (`mapping_elements_tf.py:19-22`): the variable named `zero` is
  actually `tf.constant((1,))`, so target points NOT matching the
  single mesh point get value 1, not 0. Single-point meshes DO occur
  (thermal constants at 2.53e-8 MeV in the 2017 test database, used
  by plain and ratio maps), but in all cases the target energies
  exactly match the mesh point, so the odd branch is dead in
  practice. The numpy builder maps the matching case linearly and
  raises on non-matching targets instead of copying the constant-1
  behavior.
- Selector/Distributor index bookkeeping assumes target indices are
  unique per dataset; overlapping rows BETWEEN maps are summed
  (`tf.add_n` semantics) — the COO union in A reproduces this.
- Duplicate-energy meshes: `argsort` is stable; the numpy builder
  must (and does) use stable sort to pick identical segments.

## Verification assets

- Real-size data: pickles under
  `.../neutron-standards-evaluation/output/a205d16/evaluation/output`
  (see handoff doc for paths and the profiling script; run from this
  clone with PYTHONPATH set to the clone root).
- Small data: `tests/test_compound_map_tf.py` and
  `tests/test_cross_section_maps_tf.py` compare TF against numpy
  maps and run in minutes.
