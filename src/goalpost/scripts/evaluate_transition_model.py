"""Proper holdout evaluation of transition model."""

import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from goalpost.data.nflverse_source import NFLVerseSource
from goalpost.data.nfl_drive_extractor import NFLDriveExtractor
from goalpost.representation.bayesian_updater import BayesianTeamUpdater
from goalpost.transitions.play_transition import EmpiricalPlayTransitionModel
from goalpost.transitions.drive_transition import EmpiricalDriveTransitionModel
from goalpost.domain.models import GameState, PossessionResult


def evaluate():
    print("=" * 60)
    print("Transition Model Holdout Evaluation")
    print("=" * 60)

    # Load data
    source = NFLVerseSource()
    source.fetch(seasons=[2023])
    games = source.parse()

    # Split: train on weeks 1-10, test on weeks 11-18
    train_games = [g for g in games if g.week <= 10]
    test_games = [g for g in games if g.week > 10]

    print(f"\nTrain: {len(train_games)} games (weeks 1-10)")
    print(f"Test:  {len(test_games)} games (weeks 11-18)")

    # Extract possessions
    extractor = NFLDriveExtractor()
    train_possessions = []
    for g in train_games:
        train_possessions.extend(g.possessions)

    test_possessions = []
    for g in test_games:
        test_possessions.extend(g.possessions)

    print(f"Train possessions: {len(train_possessions)}")
    print(f"Test possessions:  {len(test_possessions)}")

    # Build transitions
    train_transitions = extractor.compute_state_transitions(train_possessions)
    play_trans = [t for t in train_transitions if t.action not in [
        'td', 'fg', 'punt', 'turnover', 'turnover_on_downs', 'safety',
        'end_of_half', 'end_of_game', 'unknown'
    ]]
    drive_trans = [t for t in train_transitions if t.action in [
        'td', 'fg', 'punt', 'turnover', 'turnover_on_downs', 'safety',
        'end_of_half', 'end_of_game', 'unknown'
    ]]

    print(f"Train play transitions: {len(play_trans)}")
    print(f"Train drive transitions: {len(drive_trans)}")

    # Build team representations from train
    updater = BayesianTeamUpdater(latent_dim=8)
    updater.fit(train_possessions)
    team_vecs = {team: updater.encode(team) for team in updater.team_stats.keys()}

    # Fit models
    play_model = EmpiricalPlayTransitionModel()
    play_model.fit(team_vecs, play_trans)

    drive_model = EmpiricalDriveTransitionModel()
    drive_model.fit(team_vecs, drive_trans)

    # Evaluate on test set
    print("\n" + "=" * 60)
    print("Drive-Level Evaluation")
    print("=" * 60)

    # For each test drive, predict from start state
    correct_top1 = 0
    correct_top2 = 0
    total = 0

    log_loss = 0.0
    calibration = Counter()
    predictions_by_actual = Counter()

    for drive in test_possessions:
        if not drive.plays or not drive.team:
            continue

        # Get team vectors
        offense_vec = team_vecs.get(drive.team, np.zeros(8))
        # Defense is opponent — find from game context
        # For simplicity, use average defense
        defense_vec = np.mean([team_vecs[t] for t in team_vecs], axis=0)

        # Build start state
        start_play = drive.plays[0]
        start_state = GameState(
            down=start_play.down,
            distance=start_play.distance,
            yardline=start_play.yardline,
            quarter=drive.quarter,
            possession=drive.team,
        )

        # Predict
        pred = drive_model.predict(offense_vec, defense_vec, start_state)

        if not pred:
            continue

        # Get actual result
        actual = drive.result
        if actual is None:
            continue
        actual_key = actual.value

        # Top-1 accuracy
        predicted = max(pred, key=pred.get)
        if predicted == actual_key:
            correct_top1 += 1

        # Top-2 accuracy
        top2 = sorted(pred.items(), key=lambda x: -x[1])[:2]
        if actual_key in [k for k, _ in top2]:
            correct_top2 += 1

        # Log probability of actual outcome
        prob_actual = pred.get(actual_key, 0.01)  # Laplace smoothing
        log_loss += -np.log(max(prob_actual, 0.001))

        # Calibration bucket
        prob_bin = int(prob_actual * 10) / 10
        calibration[prob_bin] += 1
        predictions_by_actual[actual_key] += 1

        total += 1

    print(f"\nDrive Result Prediction Accuracy:")
    print(f"  Top-1: {correct_top1}/{total} = {correct_top1/total*100:.1f}%")
    print(f"  Top-2: {correct_top2}/{total} = {correct_top2/total*100:.1f}%")
    print(f"  Average Log-Loss: {log_loss/total:.3f}")
    print(f"  Random Baseline: {100/6:.1f}%")

    print(f"\nActual Result Distribution (test set):")
    for result, count in predictions_by_actual.most_common():
        print(f"  {result}: {count} ({count/total*100:.1f}%)")

    # Detailed calibration check
    print("\n" + "=" * 60)
    print("Calibration Check")
    print("=" * 60)
    print("When model predicts X% probability, how often does it happen?")

    # Group by predicted probability buckets
    from collections import defaultdict
    bucket_actuals = defaultdict(list)

    for drive in test_possessions:
        if not drive.plays or not drive.team or drive.result is None:
            continue

        offense_vec = team_vecs.get(drive.team, np.zeros(8))
        defense_vec = np.mean([team_vecs[t] for t in team_vecs], axis=0)

        start_play = drive.plays[0]
        start_state = GameState(
            down=start_play.down,
            distance=start_play.distance,
            yardline=start_play.yardline,
            quarter=drive.quarter,
            possession=drive.team,
        )

        pred = drive_model.predict(offense_vec, defense_vec, start_state)
        if not pred:
            continue

        actual_key = drive.result.value
        for result, prob in pred.items():
            bucket = int(prob * 5) / 5  # 0.0, 0.2, 0.4, 0.6, 0.8, 1.0
            is_actual = 1 if result == actual_key else 0
            bucket_actuals[bucket].append(is_actual)

    for bucket in sorted(bucket_actuals.keys()):
        actuals = bucket_actuals[bucket]
        observed_rate = np.mean(actuals)
        print(f"  Predicted prob ~{bucket:.1f}: observed rate {observed_rate:.2f} (n={len(actuals)})")

    print("\n" + "=" * 60)
    print("Sample Predictions")
    print("=" * 60)

    # Show a few actual predictions
    for drive in test_possessions[:5]:
        if not drive.plays or not drive.team:
            continue

        offense_vec = team_vecs.get(drive.team, np.zeros(8))
        defense_vec = np.mean([team_vecs[t] for t in team_vecs], axis=0)

        start_play = drive.plays[0]
        start_state = GameState(
            down=start_play.down,
            distance=start_play.distance,
            yardline=start_play.yardline,
            quarter=drive.quarter,
            possession=drive.team,
        )

        pred = drive_model.predict(offense_vec, defense_vec, start_state)

        print(f"\nDrive: {drive.team} starting at {start_play.yardline} yard line")
        print(f"Actual: {drive.result}")
        print("Predicted:")
        for result, prob in sorted(pred.items(), key=lambda x: -x[1]):
            marker = " ***" if result == drive.result.value else ""
            print(f"  {result}: {prob:.3f}{marker}")


if __name__ == "__main__":
    evaluate()
