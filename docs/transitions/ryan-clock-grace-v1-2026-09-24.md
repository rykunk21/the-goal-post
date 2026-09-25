> Historical September 24 design/run record. Paths below describe the original artifact packages. Use the root README for current package commands. Generated data is not checked in.

# Under-10-second completion approximation

The owner authorized accepting near-expiration simulations in the score distributions. The new simulator applies this at a **missing probability row with 0 < remaining half-clock < 10 seconds**. Exactly 10 seconds does not qualify. Supported transitions continue normally, including within the final 10 seconds. No terminal football state was added.

An accepted stop retains the accumulated score and ends that half approximately. A first-half acceptance still runs the entire second half. An invalid input, runtime exception or segment-cap failure is never converted to success. Missing rows at 10 seconds or more remain incomplete. The elapsed-time clock remains in use.

This is an explicit approximation: no additional play is sampled and no new points are awarded for the omitted seconds. It does not prove that all final-play scoring was captured. The existing final-segment evidence and segment-duration approximation limitations remain.

## Paired 50,000-game Packers home / Falcons away rerun

- Original strict completions: **49,591**.
- Additional games included under the rule: **151**.
- Total included: **49,742 / 50,000** (99.484%).
- Still incomplete: **258** (0.516%).
- The original strict score array was reproduced exactly. Every previously completed game's score is unchanged.

There were 152 accepted half endings across the rerun. 1 affected paths still failed later; 0 paths used grace in both halves. There are 151 additional whole-game completions. See the per-game outcomes and full missing-row event logs for that distinction. No segment caps were hit.

| Distribution | Strict mean | New mean | New standard deviation |
|---|---:|---:|---:|
| Total: GB + ATL | 37.158759 | 37.156990 | 13.540536 |
| Margin: GB − ATL | 15.964429 | 15.966326 | 12.231658 |

The means each change by less than 0.002 points in this matchup. This small sensitivity does not establish that the rule is harmless in other teams, field positions or score situations. The displayed distributions include both strict and approximate completions and exclude the 258 remaining incomplete paths. They are exploratory, not calibrated betting forecasts.

## Reproducibility and scope

The runner captures the RNG state at the start of each game in the original strict sequence, then replays that same game under the approximation. Extra second-half draws in recovered paths cannot reshuffle all subsequent simulations. Both variants use the same seed (20260924), probabilities, observed timing and 2026 team blend.

**14 tests passed.** They cover the exact threshold, first-half continuation, supported late-play scoring, strict completion, invalid inputs, loop caps, unchanged scores and a grace-accepted first half followed by a later failure. Paired full-run assertions verify every original completed score and the exact strict baseline.

No datasets, original reset artifacts, training launchers or live engine settings were changed. This folder versions only the simulation completion rule and its readout.

Files:

- `simulator.py`: revised simulator, default grace 10 seconds; 0 disables it.
- `run_comparison.py`: paired baseline/revised execution.
- `test_grace.py`, `tests.log`: verification.
- `simulation-outcomes.parquet`: each game's inclusion, grace usage, failure reason and scores.
- `missing-row-events.parquet`: every new-run missing row, time remaining, half and acceptance disposition.
- `baseline-missing-row-events.parquet`: original 409 stops for comparison.
- `scores.npy`: revised paired home/away scores; incomplete games remain [-1, -1].
- `results.json`, `comparison.json`, `settings.json`: distributions, diagnostics and hashed inputs.
- `total-pmf.csv`, `home_minus_away-pmf.csv`: exact empirical probability masses.
- `distributions.png`, `distributions.pdf`: updated plots.

For a full rerun, place this folder beside the preserved `ryan-clock-reset-v1-2026-09-24` package, then run `python run_comparison.py` and `python plot_distributions.py`. Tests: `python -m pytest test_grace.py -q`. Dependencies: numpy, pandas, pyarrow, matplotlib and pytest. This supplement does not duplicate the existing full dataset.
