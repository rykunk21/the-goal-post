# One-game transition tables — version 3

September 23, 2026. This version adds a raw ESPN summary fallback for current college games. It recovers 27 additional matrices, including 19 standalone FCS games. See `RYAN-READOUT.md` for results and remaining limits.

This is the revised layout requested by Ryan: **one row represents one game**. The two primary tables, `data/nfl_games.parquet` and `data/college_games.parquet`, have the same column names and types. College scope includes FBS games and standalone FCS games, including their lower-division opponents; standalone lower-division games are excluded.

The former long transition-element table was a supporting view, not this requested deliverable. This separate version corrects that interface. The earlier dataset, launcher, source notebook and live engine remain unchanged. No training or live-policy change accompanies this export.

## Minimum fields

| Column | Meaning |
| --- | --- |
| `game_id` | Stable source game identifier, unique within its league table |
| `transition_probabilities_flat` | Actual per-game empirical transition probabilities in a versioned, common flattening order |
| `home_team`, `away_team` | Team names/abbreviations from the schedule |
| `home_team_id`, `away_team_id` | Persistent provider identities; use league plus team ID as a key |
| `home_final_score`, `away_final_score` | Full-game score, including overtime where played |
| `game_date` | NFL calendar date or college scheduled UTC timestamp, identified by `date_precision` |

Additional fields preserve what is needed to interpret and validate the matrix: `matrix_shape`, `transition_set_version`, transition counts, source-row sample sizes, regulation scores, half-start state/scoring information, source identity, season, divisions, and `matrix_status`/`matrix_issue`.

Missing or rejected parses retain a game row with team/date/final-score metadata and a **null** matrix. A null matrix is not an all-zero observed game. Filter `matrix_status == 'parsed_regulation'` before loading matrices. The coverage report and exceptions list retain all failures; do not train on a silently complete-cased table without examining that selection.

## Shape and probability meaning

The vector reshapes to `[2, 9, E]`, with exact `E` and ordered edge keys in `data/transition_set.json`. It is a sparse encoding of the existing decision-segment transition set, not a new conventional square state matrix.

- Direction 0: home offense against away defense. Direction 1: away offense against home defense.
- Context: three half-clock bands crossed with trailing/tied/leading, measured at the segment boundary.
- Edge key: `(source, destination, possession_switch, own_points, opponent_points)`.
- Flattening is C order: direction, then context, then ordered edge key.
- Probabilities normalize over edges sharing the same source state, within each game's direction/context. Absent source rows are zero with zero source count, not estimates that every possible transition is impossible.

Both tables use the same descriptive edge catalog, built from all successfully parsed observations. No observed edge is dropped to fit the previous 998-edge training vocabulary. Consequently **this catalog is for transition-set design and inspection, not a vocabulary fitted only on a training split**. Before evaluating a predictive model, freeze its representation without using held-out outcomes and define chronological training/validation/test windows. Do not silently pass these wider arrays to the earlier launcher.

## Loading one game or one team's history

```python
import sys
sys.path.insert(0, 'code')
from read_game_rows import read, unpack, team_view

nfl = read('data/nfl_games.parquet')
college = read('data/college_games.parquet')
row = nfl.loc[nfl.matrix_status.eq('parsed_regulation')].iloc[0]
probabilities = unpack(row)
home_offense_and_defense = team_view(row, row.home_team_id)
away_offense_and_defense = team_view(row, row.away_team_id)
```

The game stays a single row; team-specific views reorder the two directions. A team history should select games involving that team's ID and retain the opponent. Whether to train one model per team or a shared VAE conditioned on team/history remains an architectural decision for the meeting. The earlier launcher implements the latter research approach; this export does not quietly change it.

## Reproducing scores is two different checks

**Exact extraction check:** count each transition's rewards to the correct team, add scoring before each half's first decision, and recover the regulation score. The exporter performs this both from ordered extracted events and from matrix counts/rewards. Final scores are checked against separately stored game/schedule records; those records may share the upstream provider, so this is not claimed as independent-provider truth.

**Stochastic reconstruction check:** simulate from a game's estimated probabilities and compare the resulting score distribution with that game's outcome. A probability matrix alone loses visit counts/order and does not determine a unique final score. Simulation also needs starting states, clock/termination behavior and any overtime mechanism. Exact count/reward replay does not establish that a simulated distribution is calibrated, and same-game reconstruction does not establish pregame predictive skill.

Matrices cover regulation. Full final and regulation scores are both included so overtime is not silently forced into a regulation-only chain. `parsed_segments.parquet` preserves event order, rewards, elapsed time and source row references. NFL play IDs refer to nflverse play IDs. College IDs depend on `pbp_source`: cfbfastR `game_row_number`, ESPN-derived `game_play_number`, or raw-summary canonical ordinal. For raw summaries, `raw_source_row_map.parquet` maps every canonical ordinal to the exact string play ID and original drive/play array positions. Do not interpret these ordinals as original ESPN IDs.

College clock stamps can represent end-of-play observations. The table labels this difference explicitly; it does not assert that the NFL clock model can be reused unchanged for college. College overtime and era-dependent rules still need separate simulation handling.

## Coverage and freshness

The intended span is 2023 through the completed 2026 games available at the September 23 snapshot. Use `data/coverage.json` for actual received/parsed counts and latest game dates. A recently updated file is not proof that every game is present or parsed successfully.

Primary sources are nflverse NFL play-by-play and SportsDataverse's college play-by-play/schedule archives. FBS and FCS are classified from the season-specific schedule. Four current college games lack primary play-by-play; older seasons have additional gaps. An explicit whole-game ESPN-derived fallback recovered 141 otherwise rejected 2026 games after team, final-score, sequence and clock checks. `pbp_source` identifies these rows and `fallback_audit.json` preserves both successful and failed attempts. No valid primary parse was replaced and no plays were spliced across sources. The NCAA archive was inspected but needs additional period/clock/scoring normalization before it can fill a row safely.

**Current standalone FCS remains incomplete:** all 150 completed FCS-versus-FCS games appear in the current inventory. This version recovers 19 from raw ESPN summaries, leaving 131 rejected. Historical standalone FCS retains 371 matrices. Across all college games, 1,561 matrices pass the existing extraction checks; 3,660 are rejected and 205 lack primary play-by-play. These are descriptive research matrices, not a complete or representative training population.

The additional fallback requested all 325 previously unparsed current-season games. All responses were saved and hashed. It accepted 27 and rejected 298. It preserves the response's drive/play array order, validates header identities/date/finals, and never reorders by numeric ID, score, or clock. No prior accepted game is replaced. Raw summaries also contain inconsistencies: scoreboard reversals (176), backward/missing clocks (101), invalid scoreboards (8), no drive data (6), schedule/date mismatches (4), team identity mismatches (2), and reversed periods (1).

`clock_quality.parquet` provides per-game timing diagnostics for every parsed game. Among the 27 recovered games, zero-duration decision segments range from 0% to 82.3%; the longest zero-duration run is 18 segments. Some zero durations are legitimate, but passing monotonicity does not establish a reliable clock model. Do not interpret these rows as clearance for college clock-based simulation or production prediction.

All source files used are pinned by hash in provenance. `SOURCE-NOTES.md` records the source checks and limitations. This is a snapshot with a reproducible local builder, not a newly scheduled collector. Future refreshes must preserve old versions and report added/changed/excluded games.

## Verification and rebuilding

Python 3.12, NumPy, pandas, PyArrow and pytest are sufficient; no model API or credentials are needed. The raw archives remain in the local source workspace because they are much larger than the derived tables. Rebuilding requires those named, pinned files; the builder performs no downloads.

```sh
python -m pytest code -q
python code/verify_tables.py data
python code/build_tables.py --sources /path/to/source-archives \
  --nfl2023 /path/to/play_by_play_2023.parquet --output data-new \
  --reuse-parsed /path/to/ryan-game-tables-v2/data \
  --raw-summaries /path/to/pinned-raw-summary-jsons
```

The builder refuses a nonempty output directory. College normalization explicitly distinguishes ordinary decisions from kickoffs/conversions/administrative rows, retains defensive scoring, checks team identity and schedule finals, and rejects scoreboard reversals or unknown play types. It does not use a monotone-score patch to conceal inconsistent raw data.

Version 3 preserves every previously accepted event and its source metadata. `code/audit_v3.py` checks that invariant against version 2 and creates the clock diagnostics. `code/verify_tables.py` independently checks the exported probability/count relationships, score replay and identical schemas. Full training has not started and the prior launcher still points to its earlier frozen dataset.
