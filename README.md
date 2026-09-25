# GoalPost: football transition extraction and simulation

The active transition implementation is `src/goalpost/transitions`. It extracts NFL and college football play-by-play into one row per game, imputes unobserved conditional rows using strictly earlier games, and simulates paired home/away regulation scores with an elapsed-time clock. This replaces the previous `NFLTransitionModel` absorbing-outcome implementation; the two tensor formats are not interchangeable.

## What is implemented

- **Game rows:** NFL and college tables share a schema containing game/date/team identifiers, final and regulation scores, flattened probabilities and counts, starting conditions and explicit parsing status. Both team directions are present, ordered home then away. Rejected or missing games retain null matrices rather than disappearing from the inventory. College scope includes FBS games and standalone FCS games, with incomplete coverage disclosed.
- **State representation:** 72 down/distance/field-position states; nine contexts from three remaining-half-time buckets and three score-lead buckets. Sparse edge keys encode source, destination, possession switch, offensive points and opponent points. Tensor shape is `[2, 9, E]`; `E` and flattening order come from the versioned catalog (2,732 edges in the September 24 reset dataset), never a hard-coded VAE input size.
- **Extraction:** nflverse NFL PBP, college archive adapters and whole-game ESPN fallbacks. Ordered segments preserve intervening scoring, special-teams and defensive events. Administrative halftime markers do not become football states. The initial extraction preserves censored endpoints; the reset removes those endpoints from the fitted matrix and retains boundary scoring evidence separately.
- **Imputation:** fill only unobserved conditional rows, using earlier UTC dates, the same team first and the same league second. Keep time buckets fixed and pool score-lead categories. Original observed rows remain unchanged; imputed values never become donors. Donor logs and masks remain separate from the flattened probabilities.
- **Simulation:** use both teams' empirical 2026 matrix blends for the fixed Packers-home/Falcons-away example. Two 1,800-second halves advance using observed segment durations; halftime reset is outside the matrix. A missing row with strictly `0 < seconds remaining < 10` may retain the accumulated score and end that half approximately. Supported late transitions still run normally. Other failures remain incomplete, not completed zero scores.
- **Outputs:** paired scores, inclusion/failure diagnostics, exact probability masses for total (`home + away`) and margin (`home - away`), and plots labeled with mean and standard deviation. Grace-policy comparison replays each strict game's RNG state to preserve already-completed scores.

This code does **not** train a VAE, certify college clock semantics, model overtime, or establish calibrated betting probabilities. Existing `kalman_vae` components remain a separate scaffold; the obsolete training example and hybrid/terminal-state demos were removed because their inputs are incompatible. `goalpost.simulator.run` now re-exports the elapsed-clock array API rather than the retired latent/drive simulator.

## Install and test

Use Python 3.12 (the CI version):

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
```

Default tests under `tests/transitions` are offline, with small public-provider fixtures. They cover extraction, scoring, clock handling, imputation and package entry points. Older network-fetch integration tests remain under `src/tests` and are not part of the offline default suite. Unit success is not full-dataset validation.

`.github/workflows/unit-tests.yml` runs this same suite for pull requests targeting `axiom/architecture-scaffold` and pushes to that branch, including PR merges. It uses read-only repository permissions and no provider credentials. A fork PR may require a maintainer to approve its first workflow run.

## Build and verify the dataset

Generated data lives outside the installed package. Set an absolute artifact root; default is `./artifacts/transitions` relative to the working directory:

```sh
export GOALPOST_TRANSITION_DATA="$PWD/artifacts/transitions"
python -m goalpost.transitions.build_tables \
  --sources /path/to/frozen-sources \
  --nfl2023 /path/to/play_by_play_2023.parquet \
  --raw-summaries /path/to/raw-summaries \
  --output "$GOALPOST_TRANSITION_DATA/extracted"
python -m goalpost.transitions.verify_tables "$GOALPOST_TRANSITION_DATA/extracted"
python -m goalpost.transitions.build_reset
python -m goalpost.transitions.verify_reset
```

Required frozen inputs: `nfl-schedule.csv`; `nfl-2024.parquet` through `nfl-2026.parquet`; separate 2023 NFL PBP; `cfb_schedules-2023.parquet` through `cfb_schedules-2026.parquet`; and `cfb-2023.parquet` through `cfb-2026.parquet`. Optional `espn_cfb_pbp-2026.parquet` and `--raw-summaries` provide whole-game fallbacks. The extraction cutoff remains September 23, 2026. Raw data and generated tables are not bundled; newer upstream revisions may not reproduce the original snapshot. Use new empty output directories; the reset refuses nonempty dataset outputs.

The reset writes `reset/raw`, `reset/data`, boundary evidence, donor provenance and score-accounting audits. `audit_v3` compares supplied extraction versions. `audit_reset` is an optional historical preflight requiring the old terminal-state dataset at `$GOALPOST_TRANSITION_DATA/legacy-imputed`; it is not part of the current rebuild.

The optional raw-summary fetcher performs public network requests only when explicitly invoked:

```sh
python -m goalpost.transitions.fetch_raw_snapshot \
  --previous-college-table /path/to/previous/college_games.parquet \
  --output /path/to/raw-summaries
```

## Simulate and plot

After the reset data exists:

```sh
python -m goalpost.transitions.simulate_reset
python -m goalpost.transitions.run_comparison
python -m goalpost.transitions.plot_distributions
# Optional strict-only baseline plot:
python -m goalpost.transitions.plot_reset
```

The strict baseline is written under `reset/matchup`; the paired current-policy results and plots under `grace`. The example is fixed to GB versus ATL, 2026 contributing games, 50,000 attempts and seed 20260924. Final-score labels are not simulation inputs. Low-level use: `goalpost.transitions.simulator.run(keys, probabilities, timing_records, n=..., seed=..., grace_seconds=10)` returns scores, diagnostics, missing-row events and per-run dispositions. Set grace to zero for strict clock completion.

For stored rows, `read_game_rows.unpack(row)` restores the catalog-shaped tensor and `team_view(row, team_id)` puts the requested team first without discarding the opponent. The removed `NFLTransitionModel` class and its 468-element flattened representation must not be used for these tables.

## Simulate the upcoming NFL week

```sh
# Use the locally cached ESPN schedule, fetching it on a cache miss:
python -m goalpost.transitions.weekly --simulations 50000

# Reproduce a run using its saved CSV and explicit selection/cutoff:
python -m goalpost.transitions.weekly \
  --schedule /path/to/saved/schedule.csv \
  --season 2026 --week 3 --game-type REG \
  --as-of 2026-09-25T14:00:00Z \
  --data "$GOALPOST_TRANSITION_DATA/reset/data" \
  --output "$GOALPOST_TRANSITION_DATA/weekly/replay-week-3" \
  --simulations 50000 --seed 20260924
```

By default the CLI reads ESPN’s current-week scoreboard through a local cache. It chooses the season/week/game-type of the nearest upcoming non-preseason kickoff within seven days in that response, then considers that whole NFL week. Thus a Friday run retains Sunday/Monday matchups but reports Thursday as already played/started. Season identity is preserved through January playoffs. An explicit week requires both `--season` and `--week`. The ESPN adapter converts UTC kickoffs to Eastern for the compatible CSV and retains the exact UTC kickoff separately. Saved nflverse-format CSVs still use Eastern kickoff strings per the [nflverse schedule dictionary](https://nflreadr.nflverse.com/articles/dictionary_schedules.html). Missing times, already-started games, recorded scores and invalid identities are exclusions; an empty/stale schedule does not produce a successful slate.

### ESPN cache behavior

No weekly CSV preparation is required. `python -m goalpost.transitions.weekly` uses [ESPN’s public NFL scoreboard](https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard) automatically.

- Cache location: `$GOALPOST_TRANSITION_DATA/schedule-cache` (default `artifacts/transitions/schedule-cache`). `--schedule-cache PATH` overrides it.
- Default lifetime: **3,600 seconds**. A valid hit makes no HTTP request. Missing, expired, corrupt, future-dated or incompatible entries trigger a fresh pull. `--cache-max-age-seconds 0` disables reuse.
- `--refresh-schedule` forces a new pull regardless of age. Use it when checking a recent kickoff/status change; a cache is a snapshot, not a live guarantee.
- Entries are keyed by provider request, separating the current-week endpoint from explicit season/week requests. Explicit postseason selections translate NFL weeks to ESPN rounds; preseason and unsupported rounds are not simulated.
- The cache stores the raw JSON, request URL, retrieval timestamp, adapter version and content hash. A valid response is written atomically only after normalization; failed refreshes leave the prior file intact but do not use stale data. Cache age uses the real retrieval clock, never the simulation’s `--as-of` date.
- ESPN home/away designations determine orientation. `LAR` maps to dataset ID `LA`, and `WSH` to `WAS`; unknown teams fail validation. The normalized game ID matches the NFL season/week/away/home form; ESPN’s event ID is retained separately.
- Only explicit `STATUS_SCHEDULED` events are eligible. Their `0–0` placeholder scores become blank CSV fields. Final, in-progress, postponed and canceled events remain excluded. No missing or unknown status is assumed scheduled.

```sh
python -m goalpost.transitions.weekly --refresh-schedule
python -m goalpost.transitions.weekly --cache-max-age-seconds 900
```

`--schedule /path/to/schedule.csv` bypasses ESPN and its cache completely for reproducible offline runs. An empty/current-week response with no upcoming games produces an explicit empty-slate result rather than inventing another week. The cache is intended for sequential local CLI runs; concurrent cold-cache invocations may each perform a pull, though atomic replacement keeps the cache valid.

For each team, blend its supported own-offense rows equally across eligible **current-season** matrices, retaining their earlier-game imputation. Both matrix history and league timing samples must precede the earlier of the run's as-of day and target kickoff day (UTC); same-day and target-game observations are excluded. Missing current-season team history is a blocked matchup, never silently replaced with a different team or season. The schedule fetch does not update the underlying frozen matrices. Each result records its contributing game IDs/dates and timing population so stale or sparse history is visible. This cutoff is an added weekly-run safeguard; it does not remove the original descriptive vocabulary limitation or turn a retrospective run into a historically recorded forecast.

The runner reuses the existing elapsed-clock simulation and strictly-under-ten-second missing-row rule. The seed is derived independently from the root seed and game ID, so changing slate order does not change another game's random sequence. It runs 50,000 attempts **per matchup** by default; use a smaller number for engineering smoke tests, not final probability estimates.

A new timestamped directory under `artifacts/transitions/weekly` contains:

- `schedule.csv`, `schedule-espn.json` (for ESPN runs) and `report.json`: normalized CSV, exact raw ESPN response, source/input hashes, original retrieval time, cache hit/age, selection/exclusions, per-game results and coverage limitations.
- One directory per selected game: `scores.npy` (home then away), `simulation-outcomes.parquet`, `missing-row-events.parquet`, `result.json`, exact total and home-margin PMF CSVs, plus `distributions.png` and `.pdf` with μ and σ labeled.
- Blocked or zero-completion matchups remain explicit in the report; no distribution is invented for them. Partially completed matchups show their full attempt/completion denominators and approximation counts.

Existing output directories are refused rather than overwritten. A run exits 0 when every selected matchup has a distribution, 2 for an empty or partially blocked slate, and nonzero on a data/network/runtime failure (saved in `report.json`). An expired cache is never used as a fallback on fetch failure. This command does not access sportsbook odds, send notifications, place bets, or train a VAE.

## Known limitations

Censored final-segment scoring is preserved as evidence but not completely learned by the fitted transition matrix. Segment-duration censoring and the under-ten-second approximation can omit late scoring. Incomplete simulations are excluded explicitly, so displayed distributions are conditional on completion. College clocks and standalone FCS coverage remain incomplete. The descriptive vocabulary was built on the full frozen snapshot rather than a train-only split; chronological imputation alone does not make this a leakage-free forecasting evaluation. There is no opponent-defense adjustment, home-field model, overtime model or demonstrated out-of-sample edge in this implementation.

Historical design decisions and original large-run results are in `docs/transitions`; their paths refer to the original artifact packages. This package refactor preserves the algorithms and adds importable modules, external artifact paths and offline CI. No production betting service is changed.
