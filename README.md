# Supporting Information: A Controlled, Failure-Inclusive Evaluation Protocol for Nonlinear-Manifold Reduced-Order Models by Kyungdoe Han and Hochan Son.

This package contains the scientific source, the analysis and figure scripts, the results summary, and the supplementary tables (`supplementary_data/`, Supplementary Data 1).

This package contains the scripts, the source package, and the aggregated data needed to regenerate the figures and summary. The retrospective per-case outputs and container images are not included. The separate prospective replication is an exception: its 160 de-identified result JSONs, canonical first-look receipt/summary, and frozen protocol/analyzer are included under `eval/` so that every prospective state-error statistic can be independently re-derived. That directory also contains a sanitized, byte-inventoried snapshot of the exact frozen scientific source and runner.


## Requirements

- Python 3.11.8 for the acceptance-tested environment
- `numpy`, `scipy`, `torch`, and `matplotlib`; exact tested versions are in
  `requirements-lock.txt` (macOS/arm64)

Recreate the tested Python package set, then install `mgp` in editable mode:

```
pip install -r requirements-lock.txt
pip install -e . --no-deps
```

## Layout

```
scripts/    scripts (figures, summaries, acceptance statistics, solution fields)
src/mgp/    model and benchmark source package
results/    inputs and generated results
eval/       prospective frozen protocol, first-look record, and de-identified results
```
