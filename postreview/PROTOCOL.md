# Post-review sensitivity protocol (FROZEN)

Protocol id: `mgp-postreview-sensitivity-v1`.
Written 2026-10-27 in response to reviewer 2 (comments 2 and 4) of DCE-2026-0101.
Frozen by the commit that adds this file in this form; the analysis rules below are
not changed after launch. All results are exploratory sensitivity analyses specified
after the prospective outcomes were known. They qualify the interpretation of P1--P5
and D1--D6 and do not replace the frozen prospective decisions.

## Code and gates

Code: `MGP_DCE_R1/postreview/` (`solvers.py`, `run_variant.py`, `retrain_variant.py`,
`aggregate_variants.py`, `chtc/`). The frozen `mgp` package of the submitted
replication repository is used unchanged; the variants subclass its ROM classes.

Local gates passed before freezing (macOS arm64, torch 2.7.0, one thread):
- `newton:25:on` reproduces the archived Newton-M-LSPG relative state errors of
  `pv1__reference_front_layer__in_domain__seed_1000` to 2.0e-14 and `mg:25` to 3.9e-16.
- Retraining at the frozen budget (60 epochs, data seed as initialization seed)
  reproduces the archived decoder of `pv1__reference_nonlinear_diffusion__in_domain__seed_1000`
  to 1.3e-15 in every weight.
- Unit tests (`test_solvers.py`): Gauss--Newton solves a linear least-squares problem
  in one step; the instrumented Newton matches `mgp.newton.solve_nonlinear_system`
  bitwise; fallback-off terminates as stagnated.

On CHTC the same gate is re-evaluated from the job outputs (`reproduction_gate` field
of every `*.variants.json`) before any variant is interpreted. Cases whose gate
difference exceeds 1e-9 are listed as platform-sensitive and excluded from no analysis.

## R-A: nonlinear-solver variants on the saved decoders (comment 4)

Data: the 160 sealed archives of the first prospective block
(`/staging/k/khan99/mgp_acceptance_prospective_v1_outputs/sealed/`), saved decoders,
archived held-out parameters and references; training-mean start. The second block
(nearest-snapshot start, 20 seeds) follows in a separate submission once a runner for
its result schema exists; its design is the same.

Variants (each changes one thing relative to the archived Newton-M-LSPG, which uses a
25-iteration cap and the unconditional 1e-3-scaled fallback step):

| id | algorithm | cap | fallback | isolates |
|---|---|---|---|---|
| `newton:25:on` | Newton on the stationarity equation | 25 | on | reproduction gate (archived arm) |
| `gn:25:on` | Gauss--Newton | 25 | on | curvature treatment |
| `newton:25:off` | Newton | 25 | off (terminate as stagnated) | fallback step |
| `newton:200:on` | Newton | 200 | on | iteration cap |
| `gn:200:off` | Gauss--Newton | 200 | off | all three |
| `mg:200` | MG | 200 | n/a | matched budget for MG |
| `lm:200` | LM-M-LSPG | 200 | n/a | matched budget for LM |

Shared: stopping quantity ||J_r^T r||_2 <= 1e-8; backtracking scales 1, 1/2, ..., 1/128
on 0.5 ||r||^2; initial coordinates = encoded training-mean candidate; one thread;
deterministic algorithms; every finite return scored.

Summaries (descriptive; no confirmatory p-values), produced by `aggregate_variants.py`:
1. Seed-level log ratios of cell means (Eq. 11 form) for MG/`gn:25:on`,
   `gn:25:on`/`newton:25:on`, `newton:25:off`/`newton:25:on`, `newton:200:on`/`newton:25:on`,
   `gn:200:off`/`newton:25:on`, `mg:200`/`gn:200:off`, and `lm:200`/`gn:200:off`, with sign
   counts over the 20 seeds, cellwise and suite-aggregate.
2. Stopping-unsuccessful and stagnated shares per variant and cell.
3. Share of iterations with a fallback step and with a non-descent direction
   (directional derivative >= 0) for the Newton variants.
4. Share of returns worse than the zero prediction (error > 1) and worse than the
   starting state, per variant and cell.

Interpretation rule (fixed before launch): every outcome is reported. If Gauss--Newton
closes most of the MG/Newton-M-LSPG gap, the paper says that P2 mostly reflects the
Newton globalization; if the gap persists under a descent-direction algorithm with
identical globalization, the paper says so. P2's prospective decision stands as a
statement about the frozen implementation either way.

Submission: `chtc/variants_block1.sub`, queue `chtc/queue_block1.txt` (160 jobs, one per
archive). Outputs: `/staging/k/khan99/mgp_postreview_v1/variants_block1/<case_id>.variants.json`.

## R-B: training budget and initialization (comment 2)

Data: the 80 in-domain archives of the first block (one decoder per family and seed);
the OOD partner archive supplies the OOD held-out cases, evaluated with the same
retrained decoder (`--extra-archive`).

Rows per (family, seed): (epochs, initialization seed) in
{(240, s), (960, s), (960, s + 100000), (960, s + 200000)}, where s is the data seed.
Everything else (training data, scaler, objective weights, Adam settings, batch size,
minibatch order seeded by s) is the frozen protocol's. Arms evaluated on every new
decoder: `mg:25`, `newton:25:on`, `gn:25:on`, `lm:25`; the affine arms do not depend on
the decoder and are not rerun. Per case the run records e_ae, e_pol (polished decoder
reconstruction of the reference), and the archived POD projection error e_proj.

Summaries: training loss histories and late-epoch decrease at each budget; training-set
reconstruction error; P1-style (archived POD-G / MG) seed ratios at each budget and
initialization with sign counts, cellwise and suite-aggregate; the share of cases with
e_pol < e_proj; MG and Newton/GN/LM errors against budget.

Interpretation rule: the 60-epoch frozen result remains the prospective record; the
budget and initialization results are sensitivity analyses, and any change of
direction is reported as such.

Submission: `chtc/retrain_block1.sub`, queue `chtc/queue_retrain_block1.txt` (320 jobs).
Outputs: `/staging/k/khan99/mgp_postreview_v1/retrain_block1/<case_id>.ep<E>_init<S>.{retrain.json,decoder.pt}`.

The second-block retraining (1600 epochs, best-loss checkpoint, learning-rate decay)
is specified separately once the 2D runner exists.

## Execution

Container: `/staging/k/khan99/containers/mgp-chtc-2de21fabe869050b.sif` (the prospective
blocks' image). One CPU, 4 GB memory, 4 GB disk per job; flocking allowed. A 2 s +
random pause separates CHTC commands, and the open ControlMaster socket is reused.
Jobs that are evicted or lose transfer are resubmitted unchanged; a job that fails
inside the solver is kept as a failure and reported. No analysis code is changed after
the first output is opened; later additions are labeled as such.

## Addendum A1 (2026-10-27, after the block-1 launch, before any output was opened): second-block R-A

Runner `run_variant_2d.py`; job `chtc/run_variant_2d_case.sh`; submission `chtc/variants_block2.sub`;
queue `chtc/queue_block2.txt` (40 archives: 20 seeds x 2 regimes), nearest-training-snapshot
start only. The second-block results do not store reference states, so the full-order
reference is re-solved and checked against the archived QoI. Variants: `newton:25:on`
(gate against the archived `mgp_lspg_error_l2`), `gn:25:on`, `newton:25:off`, `newton:200:on`,
`gn:200:off`; `mg:200` and `lm:200` are omitted in two dimensions to bound the cost (MG met
its stopping rule on 221--233 of 240 solves there, and LM is archived). Summaries and
interpretation rule as in R-A. Outputs: `/staging/k/khan99/mgp_postreview_v1/variants_block2/`.

Addendum A2 (2026-10-27, same time as A1): the job bundle's `src` is the exact frozen source of the second prospective block (commit 87b870f, identical to the project HEAD), which is a superset of the first block's source (b6456e9) and contains the 2304-unknown problem that the sanitized replication package omits. The first-block jobs already queued (cluster 7236157/7236158) carry the sanitized package source; the two differ only in docstrings, an unused QoI variant, and a default-preserving training-code option, and the local gate reproduced the archived first-block returns to 2e-14 with the sanitized source. The per-job `reproduction_gate` field records the agreement on CHTC either way.

## Addendum A3 (2026-10-27, before any R-A or R-B output was analyzed): second-block R-B

Runner `retrain_variant_2d.py` with the training loop `train_variants.py` (a copy of the frozen
loop with an optional cosine schedule and a best-training-loss checkpoint; verified bitwise
against the frozen loop on this platform for 5 and 400 epochs). As in the frozen block-2
builder, the projected-penalty graph is skipped when its weight is zero. Submission
`chtc/retrain_block2.sub`, queue `chtc/queue_retrain_block2.txt`: 20 seeds x {const 400,
const 1600, cosine 400} = 60 jobs. Each job trains one decoder from the in-domain archive's
training data (identical to the OOD archive's), and evaluates `mg:25` and `newton:25:on` with
both the final and the best-training-loss weights on the 12 in-domain and 12 OOD held-out
cases (nearest-snapshot start; references re-solved and checked against the archived QoI).

The const-400 row reproduces the frozen recipe on a different node. Locally, the retrained
loss history matches the archived one exactly at epoch 1 and departs at round-off level after
epoch 28--33, exceeding 1% after epoch ~103 (seeds 5000 and 5002), the same behavior as the
in-domain/OOD pairs of the archived block. The const-400 row is therefore a round-off
replicate of the archived training and measures training variability at the frozen recipe.

Summaries: D1/D4-style (MG vs archived POD-G) and D2/D5-style (MG vs Newton-M-LSPG) seed log
ratios for each (schedule, checkpoint), with sign counts; final/best loss ratio and best epoch;
training and held-out reconstruction errors; e_pol against the POD projection error.
Interpretation rule as in R-B.

Addendum A4 (2026-10-27, operational): the staging directory reached its 1000-file quota while the
first clusters were running, so job outputs could not be written there and 23 jobs were held at
output transfer. All outputs are now returned to `/home/khan99/mgp_postreview_v1_out/` (no file
limit): the outputs already written were moved there, the output remaps of every queued job were
edited with paced `condor_qedit` calls, and the held jobs were released to rerun unchanged. Inputs
(sealed archives, container) are still read from staging. No analysis rule changes.

Addendum A5 (2026-10-27, logged for transparency): after four first-block outputs had been opened
to check the on-cluster reproduction gate, `aggregate_variants.py` was corrected. Its contrast
"MG / GN-25-on" referenced `mg:25`, which the first-block queue does not run, so it would have
been empty; it now uses `mg:200` (the archived MG contrast was already present). An implicit MG
gate was added (agreement of `mg:200` with archived MG wherever the archived solve stopped within
25 iterations). No other analysis rule changed.

Addendum A6 (2026-10-27, post hoc, after the first-block R-A and R-B outputs were opened): two
additions, both labeled post hoc wherever reported. (1) `residual_comparison.py` compares, case by
case, the full-order residual norm at the `gn:200:off` return with that at the `mg:200` return
(the recorded `ambient_residual_norm` is the residual whose squared norm M-LSPG minimizes). It
separates "larger error with smaller residual" (the least-squares objective prefers the less
accurate state) from "larger error with larger residual" (a stationary point that is not the best
available). (2) `ledger.py` and `chtc/fetch_ledger_inputs.sh` record a retrieval ledger: every
queued job is matched to its output files, decoder weights, and stdout/stderr logs on CHTC and
locally by size and sha256, together with the job's exit record and content checks. No prespecified
summary or interpretation rule changed.

Addendum A7 (2026-10-28, operational, after both 2D clusters drained): the block-2 aggregates were
produced with the prespecified scripts and saved under separate names so that the block-1 files are
never overwritten:
`python3 aggregate_variants.py --results results/variants_block2 --out <tmp>`, then
`solver_variants_{cases,cells,seed_ratios}.csv` copied to `../supplementary_data/solver_variants_2d_*.csv`;
`python3 aggregate_retrain_2d.py --results results/retrain_block2 --out ../supplementary_data`
(writes `retrain2d_{cases,seed_ratios,training}.csv`). Because the second-block directions are
regime-specific, the 2D results are read per regime; the suite row that `aggregate_variants.py`
prints averages opposite directions and is not reported. One contrast was added after the 2D
outputs were opened and is labeled as such where reported: archived MG against `gn:200:off`, per
regime. The residual comparison of A6 is not run in 2D, because `mg:200` was not run there (A1).
No analysis rule changed.

Addendum A8 (2026-10-28, corrections to the summary code after the outputs were inspected; no run
or analysis rule changed): (1) `si_tables.py` keyed the first-block seed-ratio rows by contrast and
family without the regime, so the "cell range" column of SI Table S22 covered only the four OOD
cells. Keyed by regime, the ranges cover all eight cells; two printed ranges change (fallback
off/on 0.978-1.007; MG-200/GN-200-off 0.146-0.801). (2) `aggregate_variants.py` now pairs each
returned error with its own start error and does not count a return equal to its start within a
relative 1e-12 as worse than the start. This changes only `share_error_above_start` for
`newton:25:off`, which is not reported. (3) `aggregate_retrain_2d.py` writes its case columns in a
fixed order; the values are unchanged. (4) The unit test of the fallback-off path in
`test_solvers.py` negated the equation, which leaves the Newton step unchanged; it now uses a merit
minimized at the start, so no step is accepted.

Addendum A9 (2026-11-01, post hoc, written after all R-A and R-B outputs had been read and while
the run below was in progress, before any pre-M output on a retrained decoder was read): pre-M on the retrained first-block decoders. R-B scored MG and the
least-squares arms on the retrained decoders but not pre-M. The polished reconstruction of these
decoders beats the best affine approximation in most cases of seven of the eight cells, so this
run asks whether an online solve under the inverse-Jacobian metric attains that advantage over
POD-G. Code: `precond_retrained.py`. For each of the 320 retrained decoders (`ep240_init{s}`,
`ep960_init{s}`, `ep960_init{s+100000}`, `ep960_init{s+200000}`), pre-M (LM on
0.5 ||B0 R||^2, archived B0, archived solver settings, encoded training-mean start) is solved on
the twelve in-domain and twelve OOD held-out cases of the same seed. The affine arms keep their
archived errors. Gates, run first: (1) pre-M with each of the 160 archived decoders reproduces the
archived pre-M errors; (2) MG with each retrained decoder reproduces the R-B MG errors. The run is
local (macOS arm64, torch 2.7.0, one thread per process). Summaries are descriptive: per budget,
the seed-level geometric ratios pre-M/POD-G and pre-M/pre-P (Eq. 11 form) with sign counts, the
cellwise seed counts, and the largest relative difference between pre-M and e_pol. No
confirmatory p-values. The result is reported whatever its direction.

A9 result note (2026-11-01, after the run). Gate (1) passed: pre-M with the 160 archived
decoders reproduces the archived pre-M errors to 6.7e-16 at most. Gate (2) passed only for
converged solves: local MG with the retrained decoders matches the R-B MG errors to 1.6e-6 in the
7210 of 7680 cases where both solves converge. In the other 470 (465 capped in both runs, 5 with a
different convergence flag) the errors differ, up to 0.98, which is consistent with round-off
between the two platforms growing over unconverged iterations. The reported pre-M/MG ratios
therefore pair local pre-M with the R-B (CHTC) MG errors. Pre-M converges in at least 95.8% of
cases in every cell and budget, and its affine comparators are archived. Summaries:
`results/precond_retrained/precond_retrained_summary.csv`; aggregator checked against
Table S24 (P1 ratios reproduced exactly).
