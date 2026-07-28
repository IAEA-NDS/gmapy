# Branch organization

## `dev`

The canonical development branch. All feature branches are based on
and merged back into this branch. The former `new_features` branch
served this role and has been retired; its history is fully contained
in `dev`.

## `without_scikit_sparse`

A thin portability overlay on top of `dev`: it adds only the ad-hoc
removal of the `scikit-sparse` dependency (which requires system
libraries not available everywhere) and of the `float128` type (not
supported on all platforms). These removals **break parts of the
non-tensorflow evaluation path** (see the failing tests
`test_gma_database_class.py::test_abserr_nugget_has_negligible_impact_on_evaluation`
and `::test_evaluation_with_temporary_datapoints_removal`); the
tensorflow-based fitting is unaffected.

The branch exists solely so that the `neutron-standards-evaluation`
repository, which only uses the tensorflow path, can reference an
installable gmapy commit via its git submodule. It is updated by
**merging** `dev` into it (never by rebasing, so that commits pinned
by submodules remain reachable). Commits pinned by the evaluation
repository are additionally marked with tags (`eval/...`).

Medium-term plan: make `scikit-sparse` an optional dependency with a
runtime fallback and guard `float128` behind a capability check, so
that this overlay branch can be retired and `dev` referenced directly.

## Historical branches

`pointwise_cf252` (2022) and `feature_multiple_spectra` are markers of
earlier work; the former's approach was superseded by the current
mapping architecture (point-wise spectrum support now lives in `dev`
via the `pointwise-fission-spectrum` prior block type), the latter
(specific SACS maps) is unmerged.
