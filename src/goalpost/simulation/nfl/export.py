# Purpose: Bind saved PMFs to their schedule identity and original simulation denominators.
# Public API: WeeklyExport.load and its games/report/unavailable fields.
# Consumers: MarketTableJob validation and comparison; no simulation is rerun here.

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from goalpost.data.base.odds import timestamp
from goalpost.data.nfl.espn_export import normalize

from .contracts import NFLGame, ScoreDistribution


# Purpose: Keep usable game distributions alongside unavailable-game evidence and source report.
# Consumers: MarketTableJob validation, history checks and report denominators.
@dataclass
class WeeklyExport:
    games: list[NFLGame]
    report: dict[str, Any]
    unavailable: list[dict[str, Any]]

    # Purpose: Cross-check schedule orientation, event identity, counts and contained PMF paths.
    # Callers: MarketTableJob.run before any source load; rejects failed/unfinished exports.
    @classmethod
    def load(cls, directory: Path) -> "WeeklyExport":
        directory = directory.resolve()
        report, schedule, normalized_schedule, by_event = _read_export_metadata(directory)
        games, unavailable, seen = [], [], set()
        for result in report["results"]:
            identity = result["game"]
            game_id = identity["game_id"]
            _validate_game_id(game_id, seen)
            # Bind the exported identity to both saved schedules before reading any PMF.
            names, kickoff = _validate_identity(identity, schedule, normalized_schedule, by_event)
            attempted, retained, approximate = _simulation_counts(result)
            # Keep unavailable games in the denominator without trying to load their PMFs.
            if result["status"] != "simulated":
                unavailable.append(
                    {
                        "game_id": game_id,
                        "status": result["status"],
                        "attempted": attempted,
                        "retained": retained,
                        "approximate": approximate,
                    }
                )
                continue
            paths = _pmf_paths(directory, game_id)
            games.append(
                NFLGame(
                    game_id=game_id,
                    home=names["home"],
                    away=names["away"],
                    kickoff=kickoff,
                    total=ScoreDistribution.from_csv(paths[0], retained),
                    home_margin=ScoreDistribution.from_csv(paths[1], retained),
                    attempted=attempted,
                    retained=retained,
                    approximate=approximate,
                )
            )
        return cls(games, report, unavailable)


# Purpose: Read the report and both schedule views, rejecting unfinished or duplicate identities.
# Callers: WeeklyExport.load before per-game validation.
def _read_export_metadata(
    directory: Path,
) -> tuple[dict[str, Any], list[dict[str, str]], list[dict[str, Any]], dict[str, Any]]:
    report = json.loads((directory / "report.json").read_text())
    if report.get("regulation_only") is not True:
        raise ValueError("Only explicitly regulation-scoped exports are supported")
    if report.get("status") not in ("complete", "incomplete", "no_scheduled_matchups"):
        raise ValueError("Simulation export is unfinished or failed")
    with (directory / "schedule.csv").open(newline="") as handle:
        schedule = list(csv.DictReader(handle))
    raw_schedule = json.loads((directory / "schedule-espn.json").read_text())
    normalized_schedule = normalize(raw_schedule).to_dict("records")
    events = raw_schedule["events"]
    event_ids = [str(event["id"]) for event in events]
    if len(set(event_ids)) != len(event_ids):
        raise ValueError("Duplicate ESPN event identity")
    by_event = dict(zip(event_ids, events))
    return report, schedule, normalized_schedule, by_event


# Purpose: Reject unsafe or duplicate game directories before schedule matching.
# Callers: WeeklyExport.load; seen tracks identities across the entire export.
def _validate_game_id(game_id: str, seen: set[str]) -> None:
    if not isinstance(game_id, str) or game_id in ("", ".", ".."):
        raise ValueError("Invalid game directory")
    if Path(game_id).name != game_id or "\\" in game_id:
        raise ValueError("Unsafe game directory")
    if game_id in seen:
        raise ValueError("Duplicate simulation game identity")
    seen.add(game_id)


# Purpose: Cross-check season, team orientation and kickoff against one provider event.
# Callers: WeeklyExport.load; returns provider display names only after all joins agree.
def _validate_identity(
    identity: dict[str, Any],
    schedule: list[dict[str, str]],
    normalized_schedule: list[dict[str, Any]],
    by_event: dict[str, Any],
) -> tuple[dict[str, str], datetime]:
    game_id = identity["game_id"]
    matches = [row for row in schedule if row["game_id"] == game_id]
    if len(matches) != 1:
        raise ValueError("Simulation game needs one schedule identity")
    scheduled = matches[0]
    if type(identity.get("season")) is not int or identity["season"] != int(
        scheduled["season"]
    ):
        raise ValueError("Simulation and schedule seasons differ")
    provider_matches = [
        row
        for row in normalized_schedule
        if row["espn_event_id"] == scheduled["espn_event_id"]
    ]
    if len(provider_matches) != 1 or any(
        provider_matches[0][field] != scheduled[field]
        for field in ("game_id", "home_team", "away_team")
    ):
        raise ValueError("Provider event and exported schedule identity differ")
    kickoff = timestamp(identity["kickoff_utc"])
    if (identity["home"], identity["away"], kickoff) != (
        scheduled["home_team"],
        scheduled["away_team"],
        timestamp(scheduled["kickoff_utc"]),
    ):
        raise ValueError("Simulation and schedule orientation/kickoff differ")
    # Join through the exported provider event ID, not hard-coded abbreviation aliases.
    event = by_event[scheduled["espn_event_id"]]
    if len(event["competitions"]) != 1 or timestamp(event["date"]) != kickoff:
        raise ValueError("Ambiguous competition or mismatched kickoff")
    competitors = event["competitions"][0]["competitors"]
    if len(competitors) != 2 or {team["homeAway"] for team in competitors} != {
        "home",
        "away",
    }:
        raise ValueError("Missing or ambiguous home/away identity")
    names = {team["homeAway"]: team["team"]["displayName"] for team in competitors}
    return names, kickoff


# Purpose: Preserve attempted/retained/approximate counts while rejecting impossible totals.
# Callers: WeeklyExport.load for simulated and unavailable games alike.
def _simulation_counts(result: dict[str, Any]) -> tuple[int, int, int]:
    attempted, retained = result["attempted"], result.get("n", 0)
    approximate = result.get("diagnostics", {}).get("grace_completed_games", 0)
    counts = (attempted, retained, approximate)
    if (
        any(type(count) is not int for count in counts)
        or not 0 <= approximate <= retained <= attempted
        or attempted <= 0
    ):
        raise ValueError("Invalid simulation denominators")
    return attempted, retained, approximate


# Purpose: Resolve PMF paths and reject symlinks escaping the export root.
# Callers: WeeklyExport.load only for games marked simulated.
def _pmf_paths(directory: Path, game_id: str) -> list[Path]:
    folder = directory / game_id
    paths = [folder / "total-pmf.csv", folder / "home_minus_away-pmf.csv"]
    if any(not path.resolve().is_relative_to(directory) for path in paths):
        raise ValueError("PMF outside export directory")
    return paths
