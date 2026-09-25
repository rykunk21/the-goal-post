# Ryan: elapsed-clock reset v1

The requested reset is implemented in a separate research dataset. **The elapsed-time clock remains. Halftime is not a physical state, and the matrix cannot sample an artificial stop transition.** Original tables and the live engine remain unchanged. This version is not yet a fully validated training/simulation release: final-play modeling and one encountered support gap remain explicit.

## What changed

- Removed all 150 `destination = -1` edge types. New sparse flattened shape: **[2, 9, 2732]**, retaining the 72 physical down/distance/field-position states. No edge goes to or from an artificial terminal state.
- Rebuilt raw counts, probabilities and earlier-game imputation for both NFL and college tables, with the same inventory of 6,313 games. The 3,887 previously rejected/missing games still have null matrices.
- Historical imputation remains earlier UTC dates only, same team first and same league second. **Defaults now stay within the same time bucket**, pooling only the score-lead categories. No filled probabilities become donors; markings and donor logs stay separate.
- Simulation advances the elapsed-time clock. The buckets select probabilities; they do not advance time themselves. A half ends at clock exhaustion, and halftime possession/reset is handled outside the transition matrix.

## Final-play evidence: preserved, but not a complete generative model

Our extractor introduced `-1` because there was no observed next decision within the half. That was a boundary convention, not an nflverse administrative play mistaken for a football state. Its additional use as an unconditional simulation stop caused the earlier failure.

We cannot simply rename that unobserved next state as a real state. All 4,852 half-final segments are retained in **`data/boundary_evidence.parquet`**, with their source identifiers, scoring rewards and historical timing. Their successor is explicitly unknown. They are excluded from the normalized football-to-football transition fit. The NFL portion retains 303 scoring segments / 1,003 points; college retains 220 scoring segments / 877 points. Full evidence replay—modeled transitions plus boundary evidence plus opening rewards—matches all 2,426 parsed regulation scores exactly.

**This preservation does not mean those last-play rewards are learned by the new transition matrix.** Excluding boundary-censored outcomes can bias the estimated scoring behavior. That remaining issue must be addressed before claiming faithful game reconstruction or training readiness. Matrix counts alone no longer replay the full score; use `score-evidence-audit.parquet` and the boundary table.

The existing simulation approximation also withholds the reward for an entire segment if its sampled duration exceeds the remaining clock. A segment can bundle multiple raw events; this is not an exact model of a last snap, untimed down, touchdown conversion or kickoff. No clock clipping was used to invent those rewards. College provider timing limitations remain unresolved; this release does not certify college simulation.

## Packers home / Falcons away rerun

Same 2026 games and equal own-offense blend as before, 50,000 attempts per variant, seed 20260924. Regulation only; no home-field bonus, opponent-defense adjustment or VAE training. Final-score columns and boundary rewards were not loaded by the simulator.

| Version | Completed | Missing-row failures | Artificial stops |
|---|---:|---:|---:|
| Rebuilt raw matrices | 87 / 50,000 | 49,913 | 0 |
| Rebuilt historically imputed matrices | 49,591 / 50,000 | 409 | 0 |

All completed games reach both clock boundaries. No run hit the segment cap. Failure rows remain `[-1, -1]`, not completed zero scores. The 87 raw completions are a severely selected subset and must not be treated as a valid baseline score forecast.

The imputed failures all reach state 41 with <=30 seconds left: third down, 0–3 yards to gain, >80 yards from the opponent goal. Across the frozen NFL snapshot there is only one example of that state/time combination, a half-final run with no observed successor. See `missing-state-diagnostic.json`; no cross-time or nearest-state fallback was introduced to hide the gap.

The following distributions condition on the 49,591 completed imputed runs. Failed paths are not missing at random; the small failure percentage does not establish negligible bias.

| Distribution | Mean μ | Standard deviation σ |
|---|---:|---:|
| Green Bay + Atlanta | 37.16 | 13.54 |
| Green Bay − Atlanta | +15.96 | 12.23 |

Mean individual scores: Green Bay 26.56, Atlanta 10.60. Sigma is distribution spread, not standard error. Plots are in `matchup/packers-falcons-distributions.png` and `.pdf`; full PMFs and paired scores are included. These are diagnostic outputs, not betting projections.

Earlier output had total mean 30.90 and margin +13.62 with 52.08% of games suffering an artificial early stop. This reset changes terminal handling, transition-fitting population, imputation time conditioning and timing fallback. It is not an isolated causal estimate of the effect of removing the stop alone.

## Verification

**12 tests passed. Independent full-data verification passed.** Checks reconstruct counts directly from preserved observations and imputed probabilities independently from chronological donors; verify exact team/date provenance, same-time conditioning, source hashes, original metadata, null rows and score evidence. Tests cover clock-only termination, rejection of terminal keys, defensive scoring, possession switches, missing-row failure, zero-duration loop caps, explicit segment censoring, timing constraints, observed-zero preservation, donor identities and reproducibility.

Conservative reachable-support checks across every context pass for 0/865 NFL rows and 20/1,561 college rows under the new same-time restriction. These checks are stricter than sampled-path success. Do not confuse `matrix_status = parsed_regulation` with complete generative support, or treat the old training launcher as compatible. Unresolved states are marked in `data/imputation-provenance.parquet`.

## Files / reproduction

- `data/nfl_games.parquet`, `data/college_games.parquet`: revised imputed per-game tables; identical schemas across leagues.
- `raw/`: matching tables before imputation for comparison.
- `data/transition_set.json`: required versioned flattening catalog.
- `data/parsed_segments.parquet`: observed football-to-football segments only.
- `data/boundary_evidence.parquet`: preserved final segments with unobserved successors.
- `data/imputation-provenance.parquet`, `data/donor-history-log.json`: separate markings and earlier-game donor logs.
- `data/score-evidence-audit.parquet`: observed-score accounting and support limitations per parsed game.
- `verification.json`, `tests.log`, `PROTOCOL.json`: checks and exact design choices.
- `matchup/`: rerun scores, probabilities, assumptions, diagnostics and plots.

Python dependencies: numpy, pandas, pyarrow, pytest, matplotlib. Run `python -m pytest test_reset.py -q`, `python simulate_reset.py`, then `python plot_distributions.py`. Rebuilding with `build_reset.py` requires the preserved sibling `ryan-game-tables-v3/data` and empty `raw/` and `data/` output directories; it refuses to overwrite a dataset. `verify_reset.py` also requires that preserved source. Do not delete a shipped dataset just to run a rebuild; work in another versioned directory.

The catalog remains a descriptive subset of the all-season vocabulary, not a train-only vocabulary. No training, live betting changes, new collection or updates to the previous launchers were performed.

The next modeling work is final-play/censored-successor treatment and principled support for the identified sparse state. The reset is reproducible and inspectable, but those remaining limitations prevent declaring the entire generative model fixed.
