"""Test NFL data pipeline: fetch, parse, extract, transitions."""

import pytest
from goalpost.data.nflverse_source import NFLVerseSource
from goalpost.data.nfl_drive_extractor import NFLDriveExtractor
from goalpost.domain.models import Game, Possession, Play, Transition, GameState


class TestNFLVerseSource:
    """Test fetching and parsing NFL data from nflverse."""

    def test_fetch_single_season(self):
        """Test fetching a single season of play-by-play data."""
        source = NFLVerseSource()
        source.fetch(seasons=[2023])

        assert source._raw_data is not None
        assert len(source._raw_data) == 1
        assert source._seasons == [2023]

    def test_parse_returns_games(self):
        """Test parsing fetched data into Game domain objects."""
        source = NFLVerseSource()
        source.fetch(seasons=[2023])
        games = source.parse()

        assert len(games) > 0
        assert all(isinstance(g, Game) for g in games)

        # Check first game structure
        game = games[0]
        assert game.sport == "nfl"
        assert game.game_id
        assert game.home_team
        assert game.away_team
        assert game.season == 2023

    def test_games_have_possessions(self):
        """Test that parsed games contain possessions (drives)."""
        source = NFLVerseSource()
        source.fetch(seasons=[2023])
        games = source.parse()

        game = games[0]
        assert len(game.possessions) > 0
        assert all(isinstance(p, Possession) for p in game.possessions)

    def test_possessions_have_plays(self):
        """Test that possessions contain plays."""
        source = NFLVerseSource()
        source.fetch(seasons=[2023])
        games = source.parse()

        game = games[0]
        possession = game.possessions[0]
        assert len(possession.plays) > 0
        assert all(isinstance(p, Play) for p in possession.plays)

    def test_play_fields_populated(self):
        """Test that play fields are correctly mapped from nflverse."""
        source = NFLVerseSource()
        source.fetch(seasons=[2023])
        games = source.parse()

        game = games[0]
        possession = game.possessions[0]
        play = possession.plays[0]

        assert play.play_id
        assert play.play_type
        # NFL-specific fields should be populated
        assert play.down is not None
        assert play.distance is not None
        assert play.yardline is not None

    def test_drive_results_inferred(self):
        """Test that drive end results are inferred correctly."""
        source = NFLVerseSource()
        source.fetch(seasons=[2023])
        games = source.parse()

        # Find a possession with a result
        for game in games[:5]:  # Check first 5 games
            for possession in game.possessions:
                if possession.result is not None:
                    # Should be a valid PossessionResult
                    assert possession.result.value in [
                        "td", "fg", "punt", "turnover",
                        "turnover_on_downs", "safety",
                        "end_of_half", "end_of_game"
                    ]
                    return

        pytest.fail("No possession with inferred result found in first 5 games")


class TestNFLDriveExtractor:
    """Test extracting transitions from NFL possessions."""

    def test_extract_from_games(self):
        """Test extracting possessions from parsed games."""
        source = NFLVerseSource()
        source.fetch(seasons=[2023])
        games = source.parse()

        extractor = NFLDriveExtractor()
        possessions = extractor.extract(games)

        # Should flatten all possessions from all games
        total_possessions = sum(len(g.possessions) for g in games)
        assert len(possessions) == total_possessions

    def test_compute_transitions(self):
        """Test computing state transitions from possessions."""
        source = NFLVerseSource()
        source.fetch(seasons=[2023])
        games = source.parse()

        extractor = NFLDriveExtractor()
        possessions = extractor.extract(games)
        transitions = extractor.compute_state_transitions(possessions)

        assert len(transitions) > 0
        assert all(isinstance(t, Transition) for t in transitions)

    def test_transition_structure(self):
        """Test that transitions have correct structure."""
        source = NFLVerseSource()
        source.fetch(seasons=[2023])
        games = source.parse()

        extractor = NFLDriveExtractor()
        possessions = extractor.extract(games)
        transitions = extractor.compute_state_transitions(possessions)

        transition = transitions[0]
        assert isinstance(transition.state, GameState)
        assert isinstance(transition.next_state, GameState)
        assert transition.action
        assert transition.team_id
        assert transition.sport == "nfl"

    def test_process_pipeline(self):
        """Test full process: extract + transitions in one call."""
        source = NFLVerseSource()
        source.fetch(seasons=[2023])
        games = source.parse()

        extractor = NFLDriveExtractor()
        transitions = extractor.process(games)

        assert len(transitions) > 0


class TestEndToEndPipeline:
    """Test the complete NFL data pipeline end-to-end."""

    def test_full_pipeline(self):
        """Test complete pipeline: fetch → parse → extract → transitions."""
        # Fetch data
        source = NFLVerseSource()
        source.fetch(seasons=[2023])
        games = source.parse()

        # Extract transitions
        extractor = NFLDriveExtractor()
        transitions = extractor.process(games)

        # Validate pipeline output
        assert len(games) > 0
        assert len(transitions) > 0

        # Print summary statistics
        total_possessions = sum(len(g.possessions) for g in games)
        total_plays = sum(
            sum(len(p.plays) for p in g.possessions)
            for g in games
        )

        print(f"\n=== NFL 2023 Pipeline Summary ===")
        print(f"Games: {len(games)}")
        print(f"Total possessions (drives): {total_possessions}")
        print(f"Total plays: {total_plays}")
        print(f"State transitions: {len(transitions)}")
        print(f"Avg plays per possession: {total_plays / total_possessions:.1f}")
        print(f"Avg possessions per game: {total_possessions / len(games):.1f}")

    def test_small_sample_validation(self):
        """Test with a small sample for quick validation."""
        source = NFLVerseSource()
        source.fetch(seasons=[2023])
        games = source.parse()

        # Only process first game for quick validation
        first_game = [games[0]]
        extractor = NFLDriveExtractor()
        transitions = extractor.process(first_game)

        assert len(transitions) > 0
        assert all(t.sport == "nfl" for t in transitions)
        assert all(t.state.down is not None for t in transitions[:10])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
