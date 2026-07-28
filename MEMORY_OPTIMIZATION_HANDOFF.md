# Handoff: TF memory consumption investigation (2026-07-28)

Context document for working on gmapy's TensorFlow memory footprint.
Produced by an investigation in the `neutron-standards-evaluation`
project (parent repo of this submodule checkout). Everything needed to
continue is in this file; no other session context is required.

## Problem statement

Running the evaluation pipeline in TF mode (e.g.
`evaluation/02_parameter_optimization.py` in the parent repo) consumes
more than 10 GB of main memory. Suspected cause: how the TF graph is
created. Investigation confirmed this and identified the mechanisms.

## Measurements (reproducible, see script at the bottom)

Replay of the optimization building blocks against the `a205d16`
evaluation output (paths below), RSS after each stage:

| stage                                              | RSS     |
|----------------------------------------------------|---------|
| TF import + loading pickles                        | 0.6 GB  |
| 1st `neg_log_prob_and_gradient` (incl. tracing)    | 5.3 GB  |
| 2nd call (cached graph, no growth)                 | 5.3 GB  |
| 1 eager `neg_log_prob_hessian` call                | 7.9 GB  |

Graph statistics of the traced `neg_log_prob_and_gradient`:
- outer graph: 2.5k ops; nested function graphs: **206k ops** total
  - `propagate`: 34k ops, materialized twice (inference + forward copy)
  - backward pass of propagate: **138k ops** (~4x forward)
- The compound map registers **227 atomic dataset functions**
  (CrossSectionMap 33, ShapeMap 52, RatioMap 20, RatioShapeMap 57,
  AbsoluteRatioMap 2, ShapeOfRatioMap 47, ShapeOfSumMap 4, TotalMap 2,
  FissionAverageMap 2, RatioOfSacsMap 8).
- ~150 forward ops and ~600 backward ops per dataset; at TF's per-op
  overhead (NodeDef protos, constants stored twice, per-op Python
  traceback) this is ~20 KB/op => the observed ~4.7 GB graph.
- Related in-code evidence: comment in `gmapy/tf_uq/inference.py`
  (`generate_MCMC_chain`): decorating `run_chain` with `tf.function`
  led to >30 GB and OOM kill -- same phenomenon at larger scale.

## Root causes (with code pointers)

1. **Per-dataset graph construction.** `CompoundMap` and
   `CrossSectionBaseMap` (`gmapy/mappings/tf/cross_section_base_map_tf.py`)
   build the graph via a Python loop over 227 atomic functions
   (selector gather -> atomic propagate -> Distributor scatter,
   `tf.add_n` at the end). Graph size grows linearly with the number
   of datasets; the gradient multiplies it by ~4.

2. **Per-dataset `tape.jacobian` with pfor.**
   `_generate_atomic_jacobian` (same file, bottom) calls
   `tape.jacobian(..., experimental_use_pfor=True)` per dataset; pfor
   builds and caches a vectorized graph per atomic function (227 of
   them) during eager Hessian evaluation (+2.6 GB retained).

3. **Sequential `tf.sparse.add` accumulation.**
   `CrossSectionBaseMap.jacobian` and `CompoundMap._orig_jacobian`
   chain ~227 `tf.sparse.add` calls, creating progressively growing
   intermediate sparse tensors (O(n^2 * nnz) traffic).

4. **Jacobian recorded on persistent tapes for nothing.**
   `MultivariateNormalLikelihoodWithCovParams._log_prob_hessian_offdiag_part`
   (and the covpar parts) in `gmapy/tf_uq/custom_distributions.py`
   execute `j = self._jacfun(pars)` INSIDE a persistent `GradientTape`
   although J does not depend on `covpars`; the tape records all
   intermediates of the full Jacobian computation.

## Recommended fixes (ordered by effort)

1. Cheap Hessian fixes (hours):
   - move `jacfun(pars)` outside the persistent tapes /
     `tf.stop_gradient` it in `_log_prob_hessian_offdiag_part`;
   - reuse one dense J for the GLS part instead of re-deriving;
   - replace the `tf.sparse.add` chains: collect COO triplets of all
     datasets, build ONE `SparseTensor` (concat + `tf.sparse.reorder`).
   DONE (2026-07-28, branch `feature_tf_memory_optimization` off
   `dev`): bit-identical results, but RSS barely moved (Hessian
   7.88 -> 7.84 GB). The retained memory is dominated by the pfor
   graph caches (root cause 2), not tape recordings or sparse
   chains; the fix still removes one full Jacobian evaluation per
   Hessian call and the O(n^2) sparse accumulation.
2. Traceback stripping: TESTED, NOT WORTH IT, DISCARDED.
   `sys.tracebacklimit = 0` is a no-op on TF 2.16 (op tracebacks
   are captured by the C++ `_tf_stack.extract_stack`, which ignores
   it). Proper stripping (monkeypatching
   `tensorflow.python.util.tf_stack.extract_stack` to return one
   shared pre-captured stack object) works but saves only ~0.1 GB
   (tracing 5.31 -> 5.22 GB, Hessian 7.84 -> 7.71 GB): in TF 2.16
   the C++ capture stores compact interned frames, so the ~20 KB/op
   is nearly all NodeDef protos and doubly-stored constants.
3. Structural fix (days, the real one — design + prepared groundwork
   in `FIX3_VECTORIZATION_PLAN.md`): vectorize each map class
   ACROSS datasets -- concatenated gathers, one shared sparse
   interpolation operator, `segment_sum`-style scatters, vectorized
   ratio arithmetic. Graph becomes O(1) in the number of datasets
   (206k ops -> a few hundred). For linear maps (plain xs, shape,
   sums -- the majority) the Jacobian IS the interpolation matrix
   known at construction; `tape.jacobian`/pfor can be dropped there.

## Environment note for THIS clone

Run all profiling FROM THE ROOT OF THIS CLONE with the parent
project's venv python (path below). Verified: started from the clone
root, `import gmapy` resolves to this clone's code (script-dir
precedence over site-packages), so code changes here take effect
without any (re)install. If you run from elsewhere, you may silently
profile the submodule checkout of the evaluation repo instead --
check `gmapy.__file__` first.

## Test data for profiling

Evaluation outputs (pickles) usable to reproduce the measurements:
- `/home/gschnabel/Seafile/OmegaSpace/neutron-standards-evaluation/output/a205d16/evaluation/output/01_model_preparation_output.pkl`
  (objects: `post`, `likelihood`, `compmap`, `priortable`, `exptable`, ...)
- `/home/gschnabel/Seafile/OmegaSpace/neutron-standards-evaluation/output/a205d16/evaluation/output/02_parameter_optimization_output.pkl`
  (object: `optres` -- posterior position to evaluate at)
Python environment with matching TF:
- `/home/gschnabel/Seafile/OmegaSpace/neutron-standards-evaluation/venv/bin/python`

## Profiling script (baseline; rerun after each change)

```python
import os, resource

def mem(tag):
    with open('/proc/self/statm') as f:
        rss = int(f.read().split()[1]) * os.sysconf('SC_PAGE_SIZE') / 1e9
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print(f'[MEM] {tag:55s} rss={rss:6.2f} GB  peak={peak:6.2f} GB', flush=True)

mem('start')
import numpy as np
import tensorflow as tf
mem('after tf import')
from gmapy.data_management.object_utils import load_objects

base = ('/home/gschnabel/Seafile/OmegaSpace/'
        'neutron-standards-evaluation/output/a205d16/evaluation/output')
post, = load_objects(f'{base}/01_model_preparation_output.pkl', 'post')
optres, = load_objects(f'{base}/02_parameter_optimization_output.pkl', 'optres')
x = tf.constant(np.array(optres.position), dtype=tf.float64)
mem('after loading pickles')

nlpg = tf.function(post.neg_log_prob_and_gradient)
v, g = nlpg(x)
mem('after 1st neg_log_prob_and_gradient (incl. tracing)')
v, g = nlpg(x)
mem('after 2nd neg_log_prob_and_gradient (cached graph)')

h = post.neg_log_prob_hessian(x)
mem('after neg_log_prob_hessian (eager)')
```

Baseline numbers to beat: 5.3 GB after gradient tracing, 7.9 GB after
one Hessian evaluation.

Caveat: the pickles bake in the gmapy code state at evaluation time
via pickled closures ONLY for some objects; `post`/`likelihood` hold
references to functions defined in the current gmapy package, so code
changes in gmapy take effect when re-loading the pickle in a fresh
process. If unpickling fails after refactoring, rebuild the objects by
running `evaluation/01_model_preparation.py` in the parent repo.
