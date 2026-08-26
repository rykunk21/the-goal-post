# GoalPost Data Pipeline — Implementation Status

## Completed Fixes (2026-08-26)

### 1. Fixed NFLVerseSource.parse()
- **Issue:** Grouping by `drive` didn't guarantee chronological play order
- **Fix:** Changed to `fixed_drive` (cleaned) with `maintain_order=True`, sorted by `play_id` before grouping
- **Result:** Plays now appear in correct chronological order within each drive

### 2. Fixed Drive Result Inference
- **Issue:** `_infer_drive_result()` only checked last play, returned `None` for most drives
- **Fix:** Now reads `fixed_drive_result` column from nflverse (already computed)
- **Mapping:**
  - "Touchdown" → TOUCHDOWN
  - "Field goal" → FIELD_GOAL
  - "Punt" → PUNT
  - "Turnover" → TURNOVER
  - "Turnover on downs" → TURNOVER_ON_DOWNS
  - "Safety" → SAFETY
  - "End of half" → END_OF_HALF
  - "Opp touchdown" → TURNOVER (defensive TD)
  - "Missed field goal" → TURNOVER

### 3. Added Score Tracking
- **Issue:** `score_diff` and `time_remaining` hardcoded to 0
- **Fix:** 
  - `points_scored` extracted from plays (TD=6, FG=3, Safety=2, XP=1, 2PT=2)
  - Drive end state now includes `points_scored` from that drive
  - Game-level `home_score` and `away_score` populated from nflverse

### 4. Added End-of-Drive Transitions
- **Issue:** Only play-level transitions (within-drive) computed; missing the key prediction target
- **Fix:** `compute_state_transitions()` now outputs:
  - **Play-level:** (state_before_play, play_type, state_after_play) for each play in drive
  - **Drive-level:** (drive_start_state, "td"/"fg"/"punt"/etc., drive_end_state) — the actual outcome

### 5. Added Validation Script
- `scripts/validate_pipeline.py` fetches 2023 season and prints:
  - Game/possession/play counts
  - Sample drives with play details
  - Transition counts (play-level vs drive-level)
  - Team statistics (TD rate, punt rate, turnover rate)
  - Sanity checks against expected NFL rates

## Validation Results (2023 Season)

| Metric | Value | Expected | Status |
|--------|-------|----------|--------|
| Games | 285 | 272 (17 wks × 16 games) | ✓ (includes playoffs) |
| Total Drives | 6,330 | ~6,000-7,000 | ✓ |
| Drives/Game | 22.2 | ~22-26 | ✓ |
| TD Rate | 20.6% | ~20% | ✓ |
| Punt Rate | 37.0% | ~40% | ✓ |
| Turnover Rate | 14.0% | ~12-15% | ✓ |
| None Results | 0% | 0% | ✓ |

## What Works Now

1. **Data Ingestion:** Fetch any season 1999–present from nflverse
2. **Domain Model:** Clean Game/Possession/Play objects with correct drive results
3. **Transition Extraction:** Both play-level and drive-level transitions
4. **Team Statistics:** Basic rates computed per team (TD%, punt%, turnover%, yards/drive)

## Next Steps

### Phase 2: Transition Model (This Week)

1. **Discretize States**
   - Yardline bins: 0-20, 20-40, 40-50, 50-60, 60-80, 80-100
   - Time remaining bins: by quarter, by 2-min drill flag
   - Score differential bins: blowout (>14), close (-7 to +7), tied

2. **Empirical Transition Matrix**
   - Learn P(next_state | current_state, offense_team, defense_team)
   - Use drive-level transitions as training data
   - Compute team-specific adjustment factors

3. **Feature Engineering**
   - Encode team as vector: [td_rate, fg_rate, punt_rate, turnover_rate, yards_per_drive]
   - Add contextual features: is_home, quarter, score_diff

4. **Baseline Model**
   - Start with empirical (historical frequency) model
   - Add team-specific priors (team A's TD rate vs team B's TD allowed rate)
   - This gives working predictions before representation learning

### Phase 3: Representation Learning

1. **Team Embedding:** Compress team history into latent vector z
2. **Neural Transition Model:** Replace empirical with network conditioning on z_home, z_away
3. **Simulator:** Monte Carlo forward pass using learned transition model

## Files Modified

- `src/goalpost/data/nflverse_source.py` — Fixed parse, result inference, score tracking
- `src/goalpost/data/nfl_drive_extractor.py` — Added drive-level transitions, team stats
- `src/goalpost/scripts/validate_pipeline.py` — New validation script
