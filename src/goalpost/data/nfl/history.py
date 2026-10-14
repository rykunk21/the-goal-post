# Purpose: check reported historical input games against the maintained NFLverse schedule.
# Public API: NFLHistory.check loads evidence; assess accepts an existing DataFrame.
# Consumers: MarketTableJob checks this coverage before requesting live or stored quotes.
# Terminology: a "donor" is a historical game that the imported simulation report lists
# as a source of input data, under results[].provenance.teams[team].games[].game_id.
# For example, if Cleveland has a completed game that is missing from Cleveland's list,
# the report is blocked even if the upcoming matchup and current odds are available.
# This adapter compares game IDs; it does not choose or weight inputs, run simulations,
# inspect their underlying plays, or prove that the model actually consumed each game.

from datetime import datetime
from typing import Any

import nflreadpy
import pandas as pd

from goalpost.data.base.odds import timestamp


# Purpose: detect missing completed same-season games in each team's reported input history.
# Consumers: the shared CLI/TUI report job; passing verifies reported coverage only.
# The schedule supplies the comparison set; the imported report supplies the claimed inputs.
# Neither this comparison nor a passing result establishes model calibration or an edge.
class NFLHistory:
    # Purpose: fetch schedule evidence only when imported simulations are still upcoming.
    # Callers: MarketTableJob.run; failures stop the job before a paid odds request.
    def check(self, report: dict[str, Any], now: datetime) -> dict[str, Any]:
        results = [
            result
            for result in report["results"]
            if result["status"] == "simulated" and timestamp(result["game"]["kickoff_utc"]) > now
        ]
        if not results:
            return {"status": "not_required", "reason": "no_upcoming_simulations", "errors": []}
        seasons = sorted({int(result["game"]["season"]) for result in results})
        # Use the maintained library's downloader/cache rather than a separate raw CSV client.
        schedule = nflreadpy.load_schedules(seasons=seasons).to_pandas()
        return self.assess(schedule, results, now)

    # Purpose: compare reported historical input IDs with completed same-season team games.
    # Callers: check and offline tests; supplied schedule evidence is never regraded.
    # Both matchup teams are checked separately. Only schedule gameday values before the
    # current UTC date enter the prior-game set; missing scores are not inferred here.
    def assess(
        self, schedule: pd.DataFrame, results: list[dict[str, Any]], now: datetime
    ) -> dict[str, Any]:
        required = {
            "season",
            "game_id",
            "gameday",
            "game_type",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
        }
        if not required.issubset(schedule) or schedule.game_id.duplicated().any():
            raise ValueError("NFLverse schedule identity/schema is invalid")
        at = pd.Timestamp(now)
        if at.tzinfo is None:
            raise ValueError("History check time must include a timezone")
        dates = pd.to_datetime(schedule.gameday, utc=True, errors="raise")
        prior = schedule[(dates < at.floor("D")) & schedule.game_type.ne("PRE")]
        if prior.empty or pd.to_datetime(prior.gameday, utc=True).max() < at - pd.Timedelta(
            days=14
        ):
            raise ValueError("No recent NFL schedule evidence")
        completed = prior[prior.home_score.notna() & prior.away_score.notna()]
        errors: list[dict[str, Any]] = []
        not_due: list[str] = []
        pending = prior[prior.home_score.isna() | prior.away_score.isna()]
        for game in pending.to_dict("records"):
            kickoff = pd.NaT
            try:
                kickoff = (
                    pd.Timestamp(str(game["gameday"]) + " " + str(game.get("gametime")))
                    .tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT")
                    .tz_convert("UTC")
                )
            except (ValueError, TypeError):
                pass
            if pd.notna(kickoff) and kickoff >= at.floor("D"):
                not_due.append(game["game_id"])
            else:
                errors.append({"reason": "prior_date_result_unknown", "game_id": game["game_id"]})
        for result in results:
            game = result["game"]
            # "donors" retains the export's terminology for the reported per-team source games.
            donors = result.get("provenance", {}).get("teams", {})
            for team in (game["home"], game["away"]):
                expected = completed[
                    (completed.season == game["season"])
                    & ((completed.home_team == team) | (completed.away_team == team))
                ]
                used = {entry["game_id"] for entry in donors.get(team, {}).get("games", [])}
                missing = sorted(set(expected.game_id) - used)
                if missing:
                    errors.append(
                        {
                            "reason": "simulation_omits_completed_history",
                            "game_id": game["game_id"],
                            "team": team,
                            "missing_games": missing,
                        }
                    )
        return {
            "status": "blocked" if errors else "passed",
            "completed_games": len(completed),
            "source": "nflreadpy.load_schedules",
            "errors": errors,
            "unresolved_not_due": not_due,
            "scope": "Reported donor coverage only; not independent data or calibration validation",
        }
