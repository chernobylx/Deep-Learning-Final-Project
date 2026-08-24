"""Loading, cleaning, and merging of the raw MLB game data.

The raw data is the "MLB Game Data" Kaggle dataset
(https://www.kaggle.com/datasets/josephvm/mlb-game-data), of which three
files are used:

- ``games.csv``          -- one row per game: scores, records, metadata
- ``hittersByGame.csv``  -- one row per hitter per game
- ``pitchersByGame.csv`` -- one row per pitcher per game

The public entry point is :func:`build_game_dataset`, which runs the full
load -> clean -> merge pipeline and returns a single game-level dataframe
with home/away team statistics side by side.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# Game-level metadata that carries no predictive signal for this project.
GAMES_COLS_TO_DROP = [
    "Stadium",
    "Location",
    "Odds",
    "O/U",
    "Umpires",
    "Duration",
    "Capacity",
    "Attendance",
]

# Columns duplicated between the games file and the per-team stat files.
REDUNDANT_MERGED_COLS = [
    "Walks Issued - Away",
    "Walks Issued - Home",
    "Strikeouts Thrown - Away",
    "Strikeouts Thrown - Home",
    "home_pitchers_R",
    "away_pitchers_R",
]

COLUMN_RENAMES = {
    "away-score": "away_score",
    "home-score": "home_score",
    "Stolen Bases - Home": "home_sb",
    "Stolen Bases - Away": "away_sb",
    "Total Bases - Home": "home_tb",
    "Total Bases - Away": "away_tb",
}


def load_raw_data(data_dir):
    """Load the three raw CSV files, indexed by game ID.

    Parameters
    ----------
    data_dir : path-like
        Directory containing ``games.csv``, ``hittersByGame.csv``, and
        ``pitchersByGame.csv``.

    Returns
    -------
    (games_df, hitters_df, pitchers_df) : tuple of pd.DataFrame
    """
    data_dir = Path(data_dir)
    games_df = pd.read_csv(data_dir / "games.csv", low_memory=False).set_index("Game")
    hitters_df = pd.read_csv(data_dir / "hittersByGame.csv", low_memory=False).set_index("Game")
    pitchers_df = pd.read_csv(data_dir / "pitchersByGame.csv", low_memory=False).set_index("Game")
    return games_df, hitters_df, pitchers_df


def extract_wins_losses(df):
    """Extract wins and losses from record strings (format: 'W-L')."""
    df = df.apply(lambda records: records.str.split("-"))

    new_df = pd.DataFrame()
    new_df["home_wins"] = df["home-record"].apply(lambda x: int(x[0]))
    new_df["home_losses"] = df["home-record"].apply(lambda x: int(x[1]))
    new_df["away_wins"] = df["away-record"].apply(lambda x: int(x[0]))
    new_df["away_losses"] = df["away-record"].apply(lambda x: int(x[1]))
    new_df["home_home_wins"] = df["homehome-record"].apply(lambda x: int(x[0][0]))
    new_df["home_home_losses"] = df["homehome-record"].apply(lambda x: int(x[1][0]))
    new_df["away_away_wins"] = df["awayaway-record"].apply(lambda x: int(x[0][0]))
    new_df["away_away_losses"] = df["awayaway-record"].apply(lambda x: int(x[1][0]))

    return new_df


def clean_games(games_df):
    """Filter to regular-season 9-inning games and parse records/dates."""
    # Drop metadata and per-pitcher decision columns (WIN/LOSS/SAVE ...)
    pitcher_cols = [
        col
        for col in games_df.columns
        if any(prefix in col for prefix in ["WIN ", "LOSS ", "SAVE "])
    ]
    games_df = games_df.drop(GAMES_COLS_TO_DROP + pitcher_cols, axis=1)

    # Keep only regular season games with no extra innings
    games_df = games_df[games_df["postseason info"].isna()].drop(["postseason info"], axis=1)
    games_df = games_df[games_df["Extra Innings"].isna()].drop(["Extra Innings"], axis=1)

    games_df["Date"] = pd.to_datetime(games_df["Date"])
    games_df = games_df.dropna()

    # Split "51-32"-style record strings into numeric win/loss columns
    record_cols = ["home-record", "homehome-record", "away-record", "awayaway-record"]
    record_df = extract_wins_losses(games_df[record_cols])
    games_df = pd.concat([games_df.drop(record_cols, axis=1), record_df], axis=1)

    return games_df


def calculate_team_batting(group):
    """Calculate team batting statistics from individual player rows."""
    stats = {}
    stats["AB"] = group["AB"].sum()
    stats["H"] = group["H"].sum()
    stats["RBI"] = group["RBI"].sum()
    stats["K"] = group["K"].sum()
    stats["#P"] = group["#P"].sum()
    stats["AVG"] = stats["H"] / stats["AB"] if stats["AB"] > 0 else 0
    stats["OBP"] = (group["OBP"] * group["AB"]).sum() / stats["AB"] if stats["AB"] > 0 else 0
    stats["SLG"] = (group["SLG"] * group["AB"]).sum() / stats["AB"] if stats["AB"] > 0 else 0
    return pd.Series(stats)


def clean_hitters(hitters_df):
    """Aggregate individual hitter rows to team-level batting totals."""
    hitters_df = hitters_df.replace("--", np.nan)
    hitters_df = hitters_df[hitters_df.Position != "TEAM"].drop(
        ["Position", "Hitters", "H-AB", "Hitter Id"], axis=1
    )

    for col in hitters_df.columns:
        if col not in ["Team"]:
            hitters_df[col] = pd.to_numeric(hitters_df[col], errors="coerce")

    return hitters_df.groupby(["Game", "Team"]).apply(calculate_team_batting)


def clean_pitchers(pitchers_df):
    """Keep team-total pitching rows and parse pitch counts."""
    pitchers_df = pitchers_df.drop(["Pitcher Id", "ERA", "Extra"], axis=1)
    pitchers_df = pitchers_df[pitchers_df.Pitchers == "TEAM"].drop(["Pitchers"], axis=1)

    pitchers_df["PC"] = pitchers_df["PC"].astype(int)
    pitchers_df["ST"] = (
        pitchers_df["PC-ST"].str.split("-").apply(lambda x: int(x[-1]) if len(x) == 2 else None)
    )
    pitchers_df = pitchers_df.drop(["PC-ST"], axis=1).dropna()

    return pitchers_df


def align_dataframes(games_df, hitters_df, pitchers_df):
    """Restrict all three dataframes to the games present in each."""
    common_games = games_df.index.intersection(pitchers_df.index)
    games_df = games_df.loc[common_games]
    hitters_df = hitters_df.loc[common_games]
    pitchers_df = pitchers_df.loc[common_games].reset_index().set_index(["Game", "Team"])
    return games_df, hitters_df, pitchers_df


def merge_team_stats(games_df, hitters_df, pitchers_df):
    """Merge games_df with team statistics from hitters_df and pitchers_df.

    Parameters
    ----------
    games_df : pd.DataFrame
        Indexed by Game, with 'home' and 'away' team-name columns.
    hitters_df, pitchers_df : pd.DataFrame
        Multi-indexed by (Game, Team).

    Returns
    -------
    pd.DataFrame
        games_df with ``home_hitters_*``, ``home_pitchers_*``,
        ``away_hitters_*``, and ``away_pitchers_*`` columns appended.
    """
    result = games_df.copy()

    hitters_reset = hitters_df.reset_index()
    pitchers_reset = pitchers_df.reset_index()

    def merge_team_type_stats(base_df, stats_df, team_type, stat_type):
        """Merge stats for either the home or away team."""
        merged = base_df.reset_index().merge(
            stats_df,
            left_on=["Game", team_type],
            right_on=["Game", "Team"],
            how="left",
            suffixes=("", "_drop"),
        )

        merged = merged.drop(
            columns=["Team"] + [col for col in merged.columns if col.endswith("_drop")]
        )

        stat_cols = [col for col in stats_df.columns if col not in ["Game", "Team"]]
        rename_dict = {col: f"{team_type}_{stat_type}_{col}" for col in stat_cols}
        merged = merged.rename(columns=rename_dict)

        return merged.set_index("Game")

    result = merge_team_type_stats(result, hitters_reset, "home", "hitters")
    result = merge_team_type_stats(result, pitchers_reset, "home", "pitchers")
    result = merge_team_type_stats(result, hitters_reset, "away", "hitters")
    result = merge_team_type_stats(result, pitchers_reset, "away", "pitchers")

    return result.drop_duplicates()


def build_game_dataset(data_dir):
    """Run the full load -> clean -> merge pipeline.

    Returns a game-level dataframe with standardized column names,
    one row per regular-season 9-inning game.
    """
    games_df, hitters_df, pitchers_df = load_raw_data(data_dir)

    games_df = clean_games(games_df)
    hitters_df = clean_hitters(hitters_df)
    pitchers_df = clean_pitchers(pitchers_df)
    games_df, hitters_df, pitchers_df = align_dataframes(games_df, hitters_df, pitchers_df)

    merged_df = merge_team_stats(games_df, hitters_df, pitchers_df)
    merged_df = merged_df.rename(columns=COLUMN_RENAMES)
    merged_df = merged_df.drop(REDUNDANT_MERGED_COLS, axis=1)

    return merged_df
