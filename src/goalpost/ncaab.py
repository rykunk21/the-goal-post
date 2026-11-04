# Purpose: port Ryan's 505c9df StatBroadcast event-chain research into GoalPost's ABCs.
# Public API: StatBroadcastSource, NCAATransitionModel, NCAAEngine, build_model, main.
# Consumers: local research imports and python -m goalpost.ncaab; no network or paid odds.
# Usage: python -m goalpost.ncaab --manifest sources.json --as-of 2026-11-03
#   --home temple --away uab --n-sims 50000 --output artifacts/ncaab/run-001
# Manifest: {"schema_version": 1, "competition": "ncaa-men", "games": [
#   {"game_id": "609967", "xml": "609967.xml", "sha256": "<64 hex characters>",
#    "date": "2026-02-19", "home_team": "temple", "away_team": "uab",
#    "home_source_id": "TEM", "away_source_id": "UAB"}]}
# Optional home_play_id/away_play_id declare exact reviewed play-level aliases (e.g. WM/W&M).
# Manifest classification/aliases must be reviewed by the caller; XML alone cannot
# certify NCAA membership. Use owned/authorized exports. Live access is not enabled.
# Model v2 corrections: rotate both axes for historical venue changes; reset chains
# at halftime; score the first event; preserve empirical event counts (not possessions);
# reject missing reachable rows instead of generating uniform probabilities. Training
# excludes games on/after as-of, overtime and silent bad-file skips. This deliberately
# changes the legacy output and is not calibrated or validated for sportsbook contracts.

from __future__ import annotations

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .data.base.models import Game
from .data.base.source import DataSource
from .representation.base.models import TransitionModel
from .simulation.base.engine import Simulator
from .simulation.base.models import GameOutcome

STATE_NAMES: tuple[str, ...] = tuple(
    f"{side}_{event}"
    for side in ("V", "H")
    for event in ("made_2", "made_3", "made_ft", "miss_2", "miss_3", "miss_ft", "turnover")
)
POINTS: np.ndarray = np.array([2, 3, 1, 0, 0, 0, 0] * 2, dtype=np.int64)
SWAP: np.ndarray = np.r_[7:14, 0:7]
MODEL_VERSION: str = "statbroadcast-event-chain-v2"


# Purpose: retain validated regulation events and immutable source provenance alongside Game.
# Consumers: StatBroadcastSource and build_model; possessions stay empty (events are not
# possessions).
@dataclass
class NCAAGame(Game):
    periods: tuple[tuple[int, ...], ...] = ()
    source_sha256: str = ""
    overtime_periods: int = 0


# Purpose: make absent or non-string boundary values fail before they become ambiguous identities.
# Callers: StatBroadcastSource.parse and _parse_xml.
def _text(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing/non-string {key}")
    return value


# Purpose: map shot/turnover events without treating an absent side or unknown shot as home/two.
# Callers: _parse_xml; scoring delta disambiguates legacy untyped made baskets.
def _state(play: ET.Element, scores: list[int]) -> int | None:
    action = play.get("action", "")
    if action not in {"GOOD", "MISS", "TURNOVER"}:
        return None
    side = play.get("vh")
    if side not in {"V", "H"}:
        raise ValueError("Modeled event has no valid V/H side")
    offset = 0 if side == "V" else 7
    shot = play.get("type", "")
    if action == "TURNOVER":
        return offset + 6
    if action == "GOOD":
        new_scores = [int(play.attrib[key]) for key in ("vscore", "hscore")]
        own = offset // 7
        delta = new_scores[own] - scores[own]
        if delta not in {1, 2, 3} or new_scores[1 - own] != scores[1 - own]:
            raise ValueError("Made basket disagrees with cumulative score")
        expected = 3 if "3PTR" in shot else 1 if "FT" in shot else 2
        if shot and expected != delta:
            raise ValueError("Shot type disagrees with score increment")
        return offset + {2: 0, 3: 1, 1: 2}[delta]
    if "3PTR" in shot:
        return offset + 4
    if "FT" in shot:
        return offset + 5
    if shot not in {"JUMPER", "LAYUP", "DUNK", "TIPIN", "2PTR"}:
        raise ValueError("Unrecognized missed shot type")
    return offset + 3


# Purpose: verify final men’s two-half XML, identities, chronology and scores before retaining
# events.
# Callers: StatBroadcastSource.parse; no provider calls or inference of team aliases occur here.
def _parse_xml(raw: bytes, row: dict[str, Any]) -> NCAAGame:
    text = raw.decode("utf-8-sig")
    if "<!DOCTYPE" in text.upper() or "<!ENTITY" in text.upper():
        raise ValueError("XML declarations/entities are not accepted")
    root = ET.fromstring(text)
    venue, status = root.find("venue"), root.find("status")
    if root.tag != "bbgame" or venue is None or status is None:
        raise ValueError("Expected bbgame with venue and status")
    if status.get("complete") != "Y":
        raise ValueError("Game is not final")
    if venue.get("gender", "M") != "M" or "women" in venue.get("competitionname", "").lower():
        raise ValueError("Only reviewed men's NCAA sources are supported")
    if venue.get("sbid") and venue.get("sbid") != row["game_id"]:
        raise ValueError("StatBroadcast event ID mismatch")
    game_date = datetime.strptime(venue.attrib["date"], "%m/%d/%Y").date()
    if game_date != date.fromisoformat(row["date"]):
        raise ValueError("Manifest/XML date mismatch")
    teams = root.findall("team")
    if len(teams) != 2 or {team.get("vh") for team in teams} != {"V", "H"}:
        raise ValueError("Expected exactly one visitor and home team")
    final_scores = [0, 0]
    for team in teams:
        side = team.get("vh")
        key = "away" if side == "V" else "home"
        identity = row[f"{key}_source_id"]
        if (
            team.get("id") != identity
            or venue.get("visid" if side == "V" else "homeid") != identity
        ):
            raise ValueError("Manifest/XML team identity mismatch")
        if team.get("code", "").upper().startswith(("NAIA", "NJCAA")):
            raise ValueError("Non-NCAA competition in source")
        line = team.find("linescore")
        if line is None:
            raise ValueError("Missing final linescore")
        final_scores[0 if side == "V" else 1] = int(line.attrib["score"])
    periods = root.findall("plays/period")
    if len(root.findall("plays")) != 1 or sum(len(p.findall("play")) for p in periods) != len(
        list(root.iter("play"))
    ):
        raise ValueError("Play records outside the declared periods")
    if len(periods) < 2 or [p.get("number") for p in periods] != [
        str(i + 1) for i in range(len(periods))
    ]:
        raise ValueError("Missing/duplicate/unordered periods")
    if int(status.attrib["period"]) != len(periods):
        raise ValueError("Final status period disagrees with play-by-play")
    scores = [0, 0]
    regulation: list[tuple[int, ...]] = []
    for index, period in enumerate(periods):
        duration = 1200 if index < 2 else 300
        if period.get("time") != ("20:00" if index < 2 else "05:00"):
            raise ValueError("Expected men's 20-minute halves and five-minute overtime")
        states: list[int] = []
        last_clock = duration
        for play in period.findall("play"):
            minutes, seconds = map(int, play.attrib["time"].split(":"))
            clock = minutes * 60 + seconds
            if not 0 <= seconds < 60 or not 0 <= clock <= last_clock:
                raise ValueError("Invalid or increasing play clock")
            last_clock = clock
            state = _state(play, scores)
            if state is not None:
                if play.get("team") and play.get("team") != row.get(
                    "away_play_id" if state < 7 else "home_play_id",
                    row["away_source_id" if state < 7 else "home_source_id"],
                ):
                    raise ValueError("Play team disagrees with side")
                states.append(state)
                scores[state // 7] += int(POINTS[state])
            for side, key in enumerate(("vscore", "hscore")):
                if key in play.attrib and int(play.attrib[key]) != scores[side]:
                    raise ValueError("Play score does not reconcile to recognized baskets")
        if not states:
            raise ValueError("Empty period cannot train a model")
        if index < 2:
            regulation.append(tuple(states))
    if scores != final_scores:
        raise ValueError("Play-by-play does not reconcile to final score")
    return NCAAGame(
        game_id=row["game_id"],
        season=game_date.year + int(game_date.month >= 7),
        home_team=row["home_team"],
        away_team=row["away_team"],
        home_score=scores[1],
        away_score=scores[0],
        date=datetime.combine(game_date, datetime.min.time()),
        periods=tuple(regulation),
        source_sha256=hashlib.sha256(raw).hexdigest(),
        overtime_periods=len(periods) - 2,
    )


# Purpose: implement the shared source interface for reviewed, hash-bound local XML exports.
# Consumers: local CLI and callers using DataSource.load; acquisition/scheduling are explicitly
# absent.
class StatBroadcastSource(DataSource):
    # Purpose: bind a manifest and conservative exclusive training-date cutoff.
    # Callers: main and DataSource clients.
    def __init__(self, manifest: Path, as_of: date) -> None:
        self.manifest = manifest.resolve()
        self.as_of = as_of
        self.seasons: set[int] | None = None
        self.manifest_sha256 = ""

    # Purpose: select seasons from supplied exports without implicitly downloading anything.
    # Callers: DataSource.load or explicit offline selection.
    def fetch(self, seasons: list[int]) -> None:
        if not seasons or any(type(year) is not int or year < 1900 for year in seasons):
            raise ValueError("At least one valid season is required")
        self.seasons = set(seasons)

    # Purpose: reject unsupported scheduling rather than return a misleading empty slate.
    # Callers: generic DataSource clients; use a separately verified schedule for upcoming games.
    def fetch_upcoming(self, as_of: datetime) -> list[Game]:
        raise NotImplementedError("Local StatBroadcast exports do not provide an upcoming schedule")

    # Purpose: load every declared source with strict hash, identity, date and completeness checks.
    # Callers: DataSource.load and main; any invalid row aborts instead of silently biasing history.
    def parse(self) -> list[NCAAGame]:
        manifest_bytes = self.manifest.read_bytes()
        self.manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        manifest = json.loads(manifest_bytes)
        if not isinstance(manifest, dict):
            raise ValueError("Manifest must be a JSON object")
        if manifest.get("schema_version") != 1 or manifest.get("competition") != "ncaa-men":
            raise ValueError(
                "Manifest must declare schema_version=1 and reviewed competition=ncaa-men"
            )
        rows = manifest.get("games")
        if not isinstance(rows, list) or not rows:
            raise ValueError("Manifest has no games")
        games: list[NCAAGame] = []
        ids: set[str] = set()
        hashes: set[str] = set()
        matches: set[tuple[str, str, str]] = set()
        for row in rows:
            try:
                if not isinstance(row, dict):
                    raise ValueError("Each manifest game must be an object")
                for key in (
                    "game_id",
                    "xml",
                    "sha256",
                    "date",
                    "home_team",
                    "away_team",
                    "home_source_id",
                    "away_source_id",
                ):
                    _text(row, key)
                if (
                    row["home_team"] == row["away_team"]
                    or row["home_source_id"] == row["away_source_id"]
                ):
                    raise ValueError("Home/away identities must differ")
                if date.fromisoformat(row["date"]) >= self.as_of:
                    raise ValueError("Training game is on/after as-of; same-day games are excluded")
                for key in ("away_play_id", "home_play_id"):
                    if key in row:
                        _text(row, key)
                away_ids = {row["away_source_id"], row.get("away_play_id", row["away_source_id"])}
                home_ids = {row["home_source_id"], row.get("home_play_id", row["home_source_id"])}
                if away_ids & home_ids:
                    raise ValueError("Home/away play aliases overlap")
                path = (self.manifest.parent / row["xml"]).resolve()
                if (
                    not path.is_relative_to(self.manifest.parent)
                    or path.stat().st_size > 20_000_000
                ):
                    raise ValueError(
                        "XML must be a bounded local file within the manifest directory"
                    )
                raw = path.read_bytes()
                digest = hashlib.sha256(raw).hexdigest()
                if digest != row["sha256"]:
                    raise ValueError("Source SHA256 mismatch")
                match = (row["date"], *sorted((row["home_team"], row["away_team"])))
                if row["game_id"] in ids or digest in hashes or match in matches:
                    raise ValueError("Duplicate game or source")
                game = _parse_xml(raw, row)
                ids.add(game.game_id)
                hashes.add(digest)
                matches.add(match)
                if self.seasons is None or game.season in self.seasons:
                    games.append(game)
            except (KeyError, TypeError, ValueError, OSError, ET.ParseError) as error:
                raise ValueError(f"Invalid StatBroadcast manifest game: {error}") from error
        if not games:
            raise ValueError("No games in requested seasons")
        return games


# Purpose: represent a team's empirical regulation chain with own-team states always in rows 0..6.
# Consumers: NCAAEngine through TransitionModel; starts/lengths accompany matrices to prevent
# defaults.
class NCAATransitionModel(TransitionModel):
    # Purpose: initialize empty empirical arrays; an empty model cannot be simulated.
    # Callers: build_model and from_flat.
    def __init__(
        self, team_id: str, game_id: str = "history", opponent_id: str = "opponents"
    ) -> None:
        super().__init__(team_id, game_id, opponent_id)
        self.counts = np.zeros((14, 14))
        self.starts = np.zeros((2, 14))
        self.events_per_half = np.zeros(2)
        self.sources: list[dict[str, Any]] = []

    # Purpose: expose the retained legacy fourteen shot/turnover states.
    # Callers: generic TransitionModel consumers.
    @property
    def state_space_size(self) -> int:
        return 14

    # Purpose: declare this model's single pooled transition context.
    # Callers: generic TransitionModel consumers.
    @property
    def context_count(self) -> int:
        return 1

    # Purpose: permit empirical event transitions without inventing unseen probabilities.
    # Callers: generic TransitionModel consumers.
    @property
    def validity_mask(self) -> np.ndarray:
        return np.ones((1, 14, 14), dtype=bool)

    # Purpose: populate this model from one already validated game in its own team orientation.
    # Callers: TransitionModel clients; multi-game histories use build_model.
    def extract_from_game(self, game: NCAAGame) -> None:
        model = build_model([game], self.team_id)
        self.counts, self.starts = model.counts, model.starts
        self.events_per_half, self.sources = model.events_per_half, model.sources

    # Purpose: return empirical probabilities and expose unobserved rows as empty.
    # Callers: TransitionModel clients and regression tests.
    def get_probabilities(self, context: int, state: int) -> dict[int, float]:
        if context != 0 or not 0 <= state < 14:
            raise ValueError("Invalid NCAA context/state")
        row = self.counts[state]
        total = row.sum()
        return {int(i): float(row[i] / total) for i in np.flatnonzero(row)}

    # Purpose: serialize counts, empirical starts and half lengths together (226 values).
    # Callers: offline model persistence; provenance must be retained separately.
    def flatten(self) -> np.ndarray:
        return np.r_[self.counts.ravel(), self.starts.ravel(), self.events_per_half]

    # Purpose: restore numeric state only; no source/provenance claim is synthesized.
    # Callers: TransitionModel persistence and tests.
    @classmethod
    def from_flat(
        cls, array: np.ndarray, team_id: str, game_id: str, opponent_id: str
    ) -> NCAATransitionModel:
        values = np.asarray(array, dtype=float)
        if values.shape != (226,) or not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Expected 226 finite nonnegative NCAA values")
        model = cls(team_id, game_id, opponent_id)
        model.counts = values[:196].reshape(14, 14).copy()
        model.starts = values[196:224].reshape(2, 14).copy()
        model.events_per_half = values[224:].copy()
        return model


# Purpose: combine recent games after rotating both transition axes into own/opponent orientation.
# Callers: main and NCAATransitionModel.extract_from_game; source parsing must precede this step.
def build_model(
    games: list[NCAAGame], team_id: str, n_games: int = 5, weighting: str = "recency"
) -> NCAATransitionModel:
    if type(n_games) is not int or n_games < 1 or weighting not in {"equal", "recency"}:
        raise ValueError("Invalid history length or weighting")
    history = sorted(
        [game for game in games if team_id in (game.home_team, game.away_team)],
        key=lambda game: (game.date, game.game_id),
        reverse=True,
    )[:n_games]
    if not history:
        raise ValueError(f"No validated history for exact team ID {team_id}")
    weights = np.ones(len(history)) if weighting == "equal" else 2.0 ** -np.arange(len(history))
    weights /= weights.sum()
    model = NCAATransitionModel(team_id)
    for game, weight in zip(history, weights):
        if len(game.periods) != 2 or any(not half for half in game.periods):
            raise ValueError("Game must contain two nonempty validated regulation halves")
        for half, events in enumerate(game.periods):
            states = np.asarray(events, dtype=int)
            if (states < 0).any() or (states >= 14).any():
                raise ValueError("Event state outside NCAA state space")
            if team_id == game.home_team:
                states = SWAP[states]
            np.add.at(model.counts, (states[:-1], states[1:]), weight)
            model.starts[half, states[0]] += weight
            model.events_per_half[half] += len(states) * weight
        model.sources.append(
            {
                "game_id": game.game_id,
                "date": game.date.date().isoformat(),
                "sha256": game.source_sha256,
                "weight": float(weight),
                "overtime_periods_excluded": game.overtime_periods,
            }
        )
    return model


# Purpose: run Ryan's shared fourteen-state chain with corrected identity and empirical-only
# support.
# Consumers: generic Simulator clients and main; results are regulation research, including ties.
class NCAAEngine(Simulator):
    # Purpose: return shared GameOutcome records from reproducible empirical half chains.
    # Callers: main and Simulator clients; incomplete reachable support fails the entire run.
    def simulate(
        self,
        home_model: TransitionModel,
        away_model: TransitionModel,
        n_sims: int = 10000,
        seed: int = 0,
    ) -> list[GameOutcome]:
        if not isinstance(home_model, NCAATransitionModel) or not isinstance(
            away_model, NCAATransitionModel
        ):
            raise TypeError("NCAAEngine requires NCAATransitionModel inputs")
        if home_model.team_id == away_model.team_id or type(n_sims) is not int or n_sims < 1:
            raise ValueError("Distinct teams and positive simulation count required")
        for model in (home_model, away_model):
            values = model.flatten()
            if values.shape != (226,) or not np.isfinite(values).all() or (values < 0).any():
                raise ValueError("Invalid empirical model values")
            if (model.starts.sum(axis=1) <= 0).any() or (model.events_per_half < 1).any():
                raise ValueError("Each team requires empirical half starts and event counts")
        home_counts = home_model.counts[np.ix_(SWAP, SWAP)]
        counts = np.vstack((away_model.counts[:7], home_counts[7:]))
        starts = (away_model.starts + home_model.starts[:, SWAP]) / 2
        if (starts.sum(axis=1) <= 0).any():
            raise ValueError("Missing empirical half-start distribution")
        reachable = set(np.flatnonzero(starts.sum(axis=0)))
        pending = list(reachable)
        while pending:
            state = pending.pop()
            destinations = set(np.flatnonzero(counts[state]))
            if not destinations:
                raise ValueError(
                    f"Missing empirical transitions for reachable state {STATE_NAMES[state]}"
                )
            new = destinations - reachable
            reachable.update(new)
            pending.extend(new)
        row_sums = counts.sum(axis=1, keepdims=True)
        probs = np.divide(counts, row_sums, out=np.zeros_like(counts), where=row_sums > 0)
        lengths = (away_model.events_per_half + home_model.events_per_half) / 2
        if (lengths < 1).any() or (lengths > 4000).any():
            raise ValueError("Missing/invalid empirical half event counts")
        rng = np.random.default_rng(seed)
        visitor = np.zeros(n_sims, dtype=np.int64)
        home = np.zeros(n_sims, dtype=np.int64)
        cumulative = probs.cumsum(axis=1)
        cumulative[:, -1] = 1.0
        for half in range(2):
            states = rng.choice(14, size=n_sims, p=starts[half] / starts[half].sum())
            for step in range(int(round(lengths[half]))):
                if step:
                    draws = rng.random(n_sims)
                    states = (draws[:, None] >= cumulative[states]).sum(axis=1)
                visitor += np.where(states < 7, POINTS[states], 0)
                home += np.where(states >= 7, POINTS[states], 0)
        return [
            GameOutcome(
                home_score=int(h),
                away_score=int(v),
                metadata={
                    "model_version": MODEL_VERSION,
                    "scope": "regulation_only",
                    "research_only": True,
                },
            )
            for h, v in zip(home, visitor)
        ]


# Purpose: export reusable score arrays and provenance with no live scraping, odds calls or betting.
# Callers: python -m goalpost.ncaab; output directory must be new to preserve prior evidence.
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Offline StatBroadcast regulation research simulator"
    )
    parser.add_argument(
        "--manifest", type=Path, required=True, help="Reviewed hash-bound XML manifest"
    )
    parser.add_argument(
        "--as-of", type=date.fromisoformat, required=True, help="Exclude games on/after YYYY-MM-DD"
    )
    parser.add_argument("--home", required=True, help="Exact canonical manifest team ID")
    parser.add_argument("--away", required=True, help="Exact canonical manifest team ID")
    parser.add_argument("--n-games", type=int, default=5)
    parser.add_argument("--weighting", choices=("equal", "recency"), default="recency")
    parser.add_argument("--n-sims", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise ValueError("Output already exists; choose a new run directory")
        source = StatBroadcastSource(args.manifest, args.as_of)
        games = source.parse()
        home_model = build_model(games, args.home, args.n_games, args.weighting)
        away_model = build_model(games, args.away, args.n_games, args.weighting)
        outcomes = NCAAEngine().simulate(home_model, away_model, args.n_sims, args.seed)
        home = np.array([game.home_score for game in outcomes])
        away = np.array([game.away_score for game in outcomes])
        args.output.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(
            args.output / "scores.npz",
            home_scores=home,
            away_scores=away,
            totals=home + away,
            margins=home - away,
        )
        report = {
            "model_version": MODEL_VERSION,
            "source_commit": "505c9df2dff5967372b8692a36f89a333baf77b1",
            "scope": "regulation_only",
            "research_only": True,
            "as_of": args.as_of.isoformat(),
            "manifest_sha256": source.manifest_sha256,
            "home_team": args.home,
            "away_team": args.away,
            "seed": args.seed,
            "weighting": args.weighting,
            "n_games_requested": args.n_games,
            "attempted": args.n_sims,
            "retained": len(outcomes),
            "home_sources": home_model.sources,
            "away_sources": away_model.sources,
            "home_win_probability": float(np.mean(home > away)),
            "away_win_probability": float(np.mean(away > home)),
            "tie_probability": float(np.mean(home == away)),
            "mean_home": float(home.mean()),
            "mean_away": float(away.mean()),
            "actual_contract_ev": None,
            "actual_contract_kelly": None,
            "limitations": [
                "Unvalidated event-chain model; no overtime or sportsbook contract mapping",
                "Fixed weighted mean event counts; sparse support fails, no synthetic fallback",
                "Manifest NCAA classification and team aliases require source review",
                "As-of filters game dates, not proof these files were available pregame",
            ],
        }
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"report": str(args.output / "report.json"), "retained": len(outcomes)}))
    except (ValueError, TypeError, OSError) as error:
        parser.exit(1, f"StatBroadcast research run failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
