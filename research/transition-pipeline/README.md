# Current extraction and elapsed-clock simulation research

Code export prepared for Ryan, September 25, 2026. This is a reviewable research snapshot, not a trained VAE or a validated betting model. No production betting integration is included.

The sibling directory names preserve the current scripts' relative imports and data paths:

1. `ryan-game-tables-v3/code`: NFL nflverse extraction, college FBS/FCS adapters, whole-game ESPN fallbacks, one-row-per-game tables and count/reward verification. Each populated row contains BOTH team directions plus team IDs, scores and date. Missing/rejected games remain explicit null rows.
2. `ryan-clock-reset-v1-2026-09-24`: removes artificial terminal edges, preserves censored boundary evidence separately, rebuilds probabilities, and imputes only missing rows from earlier UTC dates. Same team first, same league second; same time bucket, pooled score context. Observed counts remain unchanged and provenance is separate.
3. `ryan-clock-grace-v1-2026-09-24`: current simulator and paired comparison. Keeps elapsed time; accepts a missing row only with strictly less than ten seconds remaining, retaining accumulated scores. Supported transitions still run normally. Failures are not zero-score games.

## Local verification (no data download required)

Use Python 3.12 or newer, create a virtual environment, then from this directory:

```sh
python -m pip install -r requirements.txt
python -m pytest -q ryan-game-tables-v3/code ryan-clock-reset-v1-2026-09-24/test_reset.py ryan-clock-grace-v1-2026-09-24/test_grace.py
```

58 tests passed in the original Python 3.12 environment. Dependencies specify minimum versions rather than a tested lockfile. Small public-provider fixtures are included. Tests do not fetch data.

## Full dataset and simulation reproduction

Large generated data, raw downloads, plots and prior-run verification artifacts are intentionally not checked in. The version-specific READMEs describe the original complete packages; their data/artifact references are prerequisites, not files included in this code-only export.

Supply the frozen September 23 input snapshot: `nfl-schedule.csv`, `nfl-2024.parquet` through `nfl-2026.parquet`, separate 2023 nflverse PBP, `cfb_schedules-2023.parquet` through `cfb_schedules-2026.parquet`, and `cfb-2023.parquet` through `cfb-2026.parquet`. The ESPN-derived 2026 archive and raw summary directory are optional whole-game fallback inputs. The cutoff remains frozen in the builder; newer downloads can revise historical inputs and need not reproduce the original byte hashes.

```sh
python ryan-game-tables-v3/code/build_tables.py --sources /path/to/frozen-sources --nfl2023 /path/to/play_by_play_2023.parquet --raw-summaries /path/to/raw-summaries --output ryan-game-tables-v3/data
python ryan-game-tables-v3/code/verify_tables.py --help
python ryan-clock-reset-v1-2026-09-24/build_reset.py
python ryan-clock-reset-v1-2026-09-24/verify_reset.py
python ryan-clock-reset-v1-2026-09-24/simulate_reset.py
python ryan-clock-grace-v1-2026-09-24/run_comparison.py
python ryan-clock-grace-v1-2026-09-24/plot_distributions.py
```

Use new empty output directories for rebuilding. The paired comparison requires the strict reset simulation first. Packers/Falcons and the 2026 blend are deliberately fixed reproduction examples. It samples empirical transition matrices; no VAE training is performed by these scripts. The optional raw downloader now takes explicit `--previous-college-table` and `--output` arguments and performs network requests only when executed, not imported.

## Known limitations retained

Regulation only; no overtime model. Final-segment rewards are preserved as evidence but not fully learned by the transition matrix. Segment-duration censoring can omit late scoring. Near-clock-expiry acceptance is an approximation, and remaining incomplete simulations are excluded with explicit counts. College clock semantics and incomplete standalone FCS coverage remain unresolved. The transition vocabulary comes from the descriptive snapshot rather than a train-only split. Full-dataset reproduction was not rerun for this code export; original run results are documented in the version READMEs.

`SOURCE-MANIFEST.json` records the original source-copy hashes. The optional download utility was adapted for portable CLI paths and simulator trailing whitespace normalized; model and extraction algorithms are unchanged. Generated training data and an updated VAE launcher are separate deliverables.
