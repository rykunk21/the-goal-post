# Purpose: verify the StatBroadcast port offline with synthetic exports and independent score
# checks.
# Public API: pytest cases for source validation, ABC integration, venue rotation and CLI evidence.
# Consumers: unit-tests CI; fixtures never request provider data or paid odds.

from __future__ import annotations

import hashlib
import json
import sys
from datetime import date, datetime
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pytest

from goalpost.data.base.source import DataSource
from goalpost.ncaab import (
    MODEL_VERSION,
    NCAAEngine,
    NCAAGame,
    NCAATransitionModel,
    StatBroadcastSource,
    build_model,
    main,
)
from goalpost.representation.base.models import TransitionModel
from goalpost.simulation.base.engine import Simulator


# Purpose: create fully reconciled synthetic XML with all fourteen event states in both halves.
# Callers: export fixture and parser regression cases; production code never creates fixture data.
def xml_game(overtime: bool = False) -> bytes:
    root = ET.Element("bbgame")
    ET.SubElement(
        root, "venue", sbid="sample", date="02/01/2026", visid="A", homeid="H", gender="M"
    )
    ET.SubElement(root, "status", complete="Y", period="3" if overtime else "2")
    lines = [
        ET.SubElement(ET.SubElement(root, "team", vh=vh, id=identity), "linescore")
        for vh, identity in (("V", "A"), ("H", "H"))
    ]
    plays = ET.SubElement(root, "plays")
    scores = [0, 0]
    for half in range(3 if overtime else 2):
        period = ET.SubElement(
            plays, "period", number=str(half + 1), time="20:00" if half < 2 else "05:00"
        )
        for step, state in enumerate(list(range(14)) * 2):
            side, event = divmod(state, 7)
            scores[side] += (2, 3, 1, 0, 0, 0, 0)[event]
            clock = (1200 if half < 2 else 300) - step - 1
            ET.SubElement(
                period,
                "play",
                vh="V" if side == 0 else "H",
                team="A" if side == 0 else "H",
                action="GOOD" if event < 3 else "MISS" if event < 6 else "TURNOVER",
                type=("JUMPER", "3PTR", "FT", "LAYUP", "3PTR", "FT", "")[event],
                time=f"{clock // 60:02}:{clock % 60:02}",
                vscore=str(scores[0]),
                hscore=str(scores[1]),
            )
    for line, score in zip(lines, scores):
        line.set("score", str(score))
    return ET.tostring(root)


# Purpose: bind a local test export to a reviewed manifest using its exact bytes.
# Callers: source validation and CLI tests.
def manifest_file(tmp_path: Path, raw: bytes | None = None) -> Path:
    raw = xml_game() if raw is None else raw
    (tmp_path / "game.xml").write_bytes(raw)
    row = {
        "game_id": "sample",
        "xml": "game.xml",
        "sha256": hashlib.sha256(raw).hexdigest(),
        "date": "2026-02-01",
        "home_team": "home",
        "away_team": "away",
        "home_source_id": "H",
        "away_source_id": "A",
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"schema_version": 1, "competition": "ncaa-men", "games": [row]}))
    return path


# Purpose: parse a synthetic export through the public source API and an exclusive cutoff.
# Callers: model and simulator tests.
def load_game(tmp_path: Path, raw: bytes | None = None) -> NCAAGame:
    return StatBroadcastSource(manifest_file(tmp_path, raw), date(2026, 2, 2)).parse()[0]


# Purpose: check shared interfaces, scored fields, season boundaries and unavailable scheduling.
# Callers: pytest.
def test_source_abc_and_season_selection(tmp_path: Path) -> None:
    source = StatBroadcastSource(manifest_file(tmp_path), date(2026, 2, 2))
    assert isinstance(source, DataSource)
    game = source.load([2026])[0]
    assert (game.home_score, game.away_score, game.season) == (24, 24, 2026)
    assert len(game.periods) == 2 and game.possessions == []
    with pytest.raises(NotImplementedError, match="schedule"):
        source.fetch_upcoming(datetime(2026, 2, 2))
    with pytest.raises(ValueError, match="No games"):
        source.load([2025])
    with pytest.raises(ValueError, match="season"):
        source.fetch([])


# Purpose: reject incomplete, contradictory, wrongly oriented or unsupported source data.
# Callers: pytest parameterization; every mutation is rebound to its true hash before parsing.
@pytest.mark.parametrize(
    "case",
    [
        "live",
        "women",
        "naia",
        "event_id",
        "team",
        "side",
        "play_team",
        "score",
        "shot",
        "clock",
        "period",
        "half_length",
        "final",
        "empty",
    ],
)
def test_invalid_xml_rejected(tmp_path: Path, case: str) -> None:
    root = ET.fromstring(xml_game())
    venue, status = root.find("venue"), root.find("status")
    play = root.find("plays/period/play")
    if case == "live":
        status.set("complete", "N")
    elif case == "women":
        venue.set("gender", "F")
    elif case == "naia":
        root.find("team").set("code", "NAIA123")
    elif case == "event_id":
        venue.set("sbid", "different")
    elif case == "team":
        venue.set("homeid", "other")
    elif case == "side":
        play.attrib.pop("vh")
    elif case == "play_team":
        play.set("team", "H")
    elif case == "score":
        play.set("vscore", "7")
    elif case == "shot":
        play.set("type", "3PTR")
    elif case == "clock":
        play.set("time", "21:00")
    elif case == "period":
        root.findall("plays/period")[1].set("number", "1")
    elif case == "half_length":
        root.find("plays/period").set("time", "10:00")
    elif case == "final":
        root.find("team/linescore").set("score", "100")
    elif case == "empty":
        root.find("plays/period").clear()
    with pytest.raises(ValueError):
        load_game(tmp_path, ET.tostring(root))


# Purpose: ensure data integrity, competition classification, deduplication and cutoff are
# mandatory.
# Callers: pytest.
@pytest.mark.parametrize(
    "case", ["hash", "same_day", "future", "duplicate", "classification", "same_team", "empty"]
)
def test_invalid_manifest_rejected(tmp_path: Path, case: str) -> None:
    path = manifest_file(tmp_path)
    manifest = json.loads(path.read_text())
    row = manifest["games"][0]
    if case == "hash":
        row["sha256"] = "0" * 64
    elif case in {"same_day", "future"}:
        row["date"] = "2026-02-02" if case == "same_day" else "2026-02-03"
    elif case == "duplicate":
        manifest["games"].append(dict(row))
    elif case == "classification":
        manifest["competition"] = "unknown"
    elif case == "same_team":
        row["home_team"] = row["away_team"]
    elif case == "empty":
        manifest["games"] = []
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        StatBroadcastSource(path, date(2026, 2, 2)).parse()


# Purpose: block XML declarations and manifest paths that escape the reviewed source directory.
# Callers: pytest.
def test_entity_and_path_escape(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="declarations"):
        load_game(tmp_path, b'<!DOCTYPE bbgame [<!ENTITY x "text">]>' + xml_game())
    nested = tmp_path / "sources"
    nested.mkdir()
    path = manifest_file(nested)
    manifest = json.loads(path.read_text())
    manifest["games"][0]["xml"] = "../game.xml"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="bounded local"):
        StatBroadcastSource(path, date(2026, 2, 2)).parse()


# Purpose: retain actual full-game results while excluding overtime from regulation training.
# Callers: pytest.
def test_overtime_and_untyped_made_basket(tmp_path: Path) -> None:
    root = ET.fromstring(xml_game(overtime=True))
    root.find("plays/period/play").attrib.pop("type")
    game = load_game(tmp_path, ET.tostring(root))
    assert game.overtime_periods == 1
    assert game.home_score == game.away_score == 36
    assert len(game.periods) == 2
    model = build_model([game], "away")
    np.testing.assert_array_equal(model.events_per_half, [28, 28])
    assert model.sources[0]["overtime_periods_excluded"] == 1


# Purpose: prove venue inversion rotates both axes and that halftime does not create a transition.
# Callers: pytest; expected counts are computed from explicit events, not implementation helpers.
def test_orientation_and_halftime(tmp_path: Path) -> None:
    game = load_game(tmp_path)
    away, home = build_model([game], "away"), build_model([game], "home")
    assert isinstance(away, TransitionModel)
    expected = np.zeros((14, 14))
    for events in game.periods:
        for before, after in zip(events, events[1:]):
            expected[before, after] += 1
    np.testing.assert_array_equal(away.counts, expected)
    for i in range(14):
        for j in range(14):
            assert home.counts[(i + 7) % 14, (j + 7) % 14] == expected[i, j]
    assert away.counts.sum() == 54
    assert away.counts[13, 0] == 2  # no transition from the first half into the second
    np.testing.assert_array_equal(home.starts[:, 7], [1, 1])


# Purpose: verify date ordering, exact team membership, weighting and portable numeric state.
# Callers: pytest.
def test_history_and_roundtrip(tmp_path: Path) -> None:
    game = load_game(tmp_path)
    older = NCAAGame(
        game_id="older",
        home_team="home",
        away_team="away",
        date=datetime(2026, 1, 1),
        periods=((7, 7, 7), (7, 7, 7)),
    )
    model = build_model([older, game], "away")
    assert [item["game_id"] for item in model.sources] == ["sample", "older"]
    np.testing.assert_allclose([item["weight"] for item in model.sources], [2 / 3, 1 / 3])
    restored = NCAATransitionModel.from_flat(model.flatten(), "away", "history", "opponents")
    np.testing.assert_array_equal(model.flatten(), restored.flatten())
    assert restored.sources == []
    assert restored.state_space_size == 14 and restored.context_count == 1
    assert restored.validity_mask.shape == (1, 14, 14)
    assert sum(restored.get_probabilities(0, 0).values()) == pytest.approx(1)
    restored.extract_from_game(game)
    np.testing.assert_array_equal(restored.flatten(), build_model([game], "away").flatten())
    with pytest.raises(ValueError, match="exact team"):
        build_model([game], "aw")
    with pytest.raises(ValueError):
        NCAATransitionModel.from_flat(np.full(226, np.nan), "a", "g", "o")


# Purpose: independently score a deterministic observed chain, including first events in both
# halves.
# Callers: pytest.
def test_simulator_scores_and_reproducibility(tmp_path: Path) -> None:
    game = load_game(tmp_path)
    home, away = build_model([game], "home"), build_model([game], "away")
    engine = NCAAEngine()
    assert isinstance(engine, Simulator)
    outcomes = engine.simulate(home, away, n_sims=100, seed=8)
    assert outcomes == engine.simulate(home, away, n_sims=100, seed=8)
    assert {(o.home_score, o.away_score, o.total, o.margin) for o in outcomes} == {(24, 24, 48, 0)}
    assert all(o.metadata["scope"] == "regulation_only" for o in outcomes)
    away.counts[2] = 0
    assert away.get_probabilities(0, 2) == {}
    with pytest.raises(ValueError, match="Missing empirical transitions"):
        engine.simulate(home, away, n_sims=100)


# Purpose: reject malformed controls and models instead of emitting empty/unsupported simulations.
# Callers: pytest.
@pytest.mark.parametrize(
    "case", ["zero", "negative", "same_team", "empty", "nan", "negative_counts", "length"]
)
def test_invalid_simulation_rejected(tmp_path: Path, case: str) -> None:
    game = load_game(tmp_path)
    home, away = build_model([game], "home"), build_model([game], "away")
    count = 10
    if case == "zero":
        count = 0
    elif case == "negative":
        count = -1
    elif case == "same_team":
        home = away
    elif case == "empty":
        home = NCAATransitionModel("home")
    elif case == "nan":
        home.counts[0, 0] = np.nan
    elif case == "negative_counts":
        home.counts[0, 0] = -1
    elif case == "length":
        home.events_per_half[:] = 10000
    with pytest.raises(ValueError):
        NCAAEngine().simulate(home, away, count)


# Purpose: exercise the public CLI end-to-end and prevent overwriting an earlier research run.
# Callers: pytest; only synthetic local files are read/written.
def test_cli_export(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = manifest_file(tmp_path)
    output = tmp_path / "run"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ncaab",
            "--manifest",
            str(path),
            "--as-of",
            "2026-02-02",
            "--home",
            "home",
            "--away",
            "away",
            "--n-sims",
            "50",
            "--output",
            str(output),
        ],
    )
    assert main() == 0
    report = json.loads((output / "report.json").read_text())
    assert report["attempted"] == report["retained"] == 50
    assert report["model_version"] == MODEL_VERSION
    assert report["tie_probability"] == 1
    assert report["actual_contract_ev"] is None and report["actual_contract_kelly"] is None
    with np.load(output / "scores.npz", allow_pickle=False) as scores:
        np.testing.assert_array_equal(
            scores["totals"], scores["home_scores"] + scores["away_scores"]
        )
        np.testing.assert_array_equal(
            scores["margins"], scores["home_scores"] - scores["away_scores"]
        )
    original = (output / "report.json").read_bytes()
    with pytest.raises(SystemExit) as exit_info:
        main()
    assert exit_info.value.code == 1
    assert (output / "report.json").read_bytes() == original


# Purpose: accept reviewed per-game play aliases; reject overlap and automatic matching.
# Callers: pytest; reproduces the WM venue ID versus W&M play ID found in Ryan's cached export.
def test_reviewed_play_alias(tmp_path: Path) -> None:
    root = ET.fromstring(xml_game())
    for play in root.iter("play"):
        if play.get("vh") == "V":
            play.set("team", "A&X")
    path = manifest_file(tmp_path, ET.tostring(root))
    source = StatBroadcastSource(path, date(2026, 2, 2))
    with pytest.raises(ValueError, match="Play team"):
        source.parse()
    manifest = json.loads(path.read_text())
    manifest["games"][0]["away_play_id"] = "A&X"
    path.write_text(json.dumps(manifest))
    assert source.parse()[0].away_score == 24
    manifest["games"][0]["home_play_id"] = "A&X"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="overlap"):
        source.parse()


# Purpose: reject unknown misses instead of the legacy two-point assumption.
# Callers: pytest; reproduces the untyped miss in the committed UAB/Temple export.
def test_untyped_miss_fails(tmp_path: Path) -> None:
    root = ET.fromstring(xml_game())
    root.findall("plays/period/play")[3].attrib.pop("type")
    with pytest.raises(ValueError, match="Unrecognized missed shot"):
        load_game(tmp_path, ET.tostring(root))


# Purpose: report malformed manifest structure without leaking an internal traceback.
# Callers: pytest.
@pytest.mark.parametrize(
    "value", [[], {"schema_version": 1, "competition": "ncaa-men", "games": [None]}]
)
def test_manifest_structure(tmp_path: Path, value: object) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="object"):
        StatBroadcastSource(path, date(2026, 2, 2)).parse()


# Purpose: check stochastic scoring against an independent binomial distribution with known moments.
# Callers: pytest; every empirical state here is a synthetic test fixture.
def test_stochastic_chain_matches_known_distribution() -> None:
    away = NCAATransitionModel("away")
    away.counts[:, 0] = 1
    away.counts[:, 7] = 3
    away.starts[:, 0] = 1
    away.starts[:, 7] = 3
    away.events_per_half[:] = 10
    home = NCAATransitionModel("home")
    home.counts[:, 0] = 3
    home.counts[:, 7] = 1
    home.starts[:, 0] = 3
    home.starts[:, 7] = 1
    home.events_per_half[:] = 10
    outcomes = NCAAEngine().simulate(home, away, 5000, 29)
    visitor = np.array([game.away_score for game in outcomes])
    assert all(game.total == 40 for game in outcomes)
    assert visitor.mean() == pytest.approx(20 * 0.25 * 2, abs=0.2)
    assert visitor.var() == pytest.approx(20 * 0.25 * 0.75 * 4, abs=0.8)
