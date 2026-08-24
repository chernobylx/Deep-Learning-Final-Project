"""Tests for the data-processing and feature-engineering logic.

These tests cover the pandas-based pipeline with small synthetic inputs;
model training is exercised via the notebook, not here.
"""

import numpy as np
import pandas as pd
import pytest

from mlb_prediction.data import extract_wins_losses
from mlb_prediction.features import (
    calculate_running_averages,
    create_ratio_features,
    prepare_modeling_data,
)


def test_extract_wins_losses_parses_record_strings():
    df = pd.DataFrame(
        {
            "home-record": ["51-32", "10-5"],
            "homehome-record": ["30-12", "6-2"],
            "away-record": ["40-43", "7-8"],
            "awayaway-record": ["18-25", "3-4"],
        }
    )
    result = extract_wins_losses(df)
    assert result["home_wins"].tolist() == [51, 10]
    assert result["home_losses"].tolist() == [32, 5]
    assert result["away_wins"].tolist() == [40, 7]
    assert result["away_losses"].tolist() == [43, 8]


def _make_merged_df():
    """Three consecutive games between two teams, home stats increasing."""
    dates = pd.to_datetime(["2021-05-01", "2021-05-02", "2021-05-03"])
    return pd.DataFrame(
        {
            "Date": dates,
            "home": ["NYY", "NYY", "NYY"],
            "away": ["BOS", "BOS", "BOS"],
            "home_score": [4, 6, 2],
            "away_score": [3, 1, 5],
            "home_hitters_H": [8.0, 10.0, 12.0],
            "away_hitters_H": [7.0, 5.0, 9.0],
        },
        index=pd.Index([1, 2, 3], name="Game"),
    )


def test_running_averages_exclude_current_game():
    merged_df = _make_merged_df()
    result = calculate_running_averages(merged_df, windows=[2])

    col = "home_hitters_H_avg_2"
    assert col in result.columns
    # Game 1: no prior games -> NaN
    assert np.isnan(result.loc[1, col])
    # Game 2: only game 1 in history -> 8.0 (current game excluded)
    assert result.loc[2, col] == pytest.approx(8.0)
    # Game 3: mean of games 1 and 2 -> 9.0
    assert result.loc[3, col] == pytest.approx(9.0)


def test_prepare_modeling_data_targets_and_columns():
    merged_df = _make_merged_df()
    running_df = calculate_running_averages(merged_df, windows=[2])
    modeling_df = prepare_modeling_data(running_df)

    # Derived targets
    assert (modeling_df["total"] == modeling_df["home_score"] + modeling_df["away_score"]).all()
    assert (
        modeling_df["margin"] == (modeling_df["home_score"] - modeling_df["away_score"]).abs()
    ).all()

    # Only targets and rolling-average features remain; NaN rows dropped
    non_target = [c for c in modeling_df.columns if c not in
                  ["home_score", "away_score", "total", "margin"]]
    assert all("_avg_" in c for c in non_target)
    assert not modeling_df.isna().any().any()


def test_create_ratio_features_clipping_and_log():
    df = pd.DataFrame(
        {
            "home_score": [4, 6],
            "away_score": [3, 1],
            "total": [7, 7],
            "margin": [1, 5],
            "home_hitters_H_avg_2": [100.0, 8.0],
            "away_hitters_H_avg_2": [0.0, 8.0],
        }
    )
    enhanced, new_features = create_ratio_features(df, max_ratio=5.0)

    assert "ratio_hitters_H_avg_2" in enhanced.columns
    # Extreme ratio (101/1) is clipped to max_ratio
    assert enhanced["ratio_hitters_H_avg_2"].iloc[0] == pytest.approx(5.0)
    # Equal stats -> ratio 1, log-ratio 0
    assert enhanced["ratio_hitters_H_avg_2"].iloc[1] == pytest.approx(1.0)
    assert enhanced["log_ratio_hitters_H_avg_2"].iloc[1] == pytest.approx(0.0)
    # Difference feature present
    assert enhanced["diff_hitters_H_avg_2"].iloc[0] == pytest.approx(100.0)
    assert len(new_features) == 3
