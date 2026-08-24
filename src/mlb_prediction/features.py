"""Feature engineering: rolling averages, matchup ratios, and advanced metrics.

Two stages of feature engineering are applied to the merged game dataset:

1. :func:`calculate_running_averages` converts each team's raw per-game
   statistics into trailing rolling averages (excluding the current game),
   so that every feature is known *before* the game is played.
2. :func:`create_ratio_features` and :func:`create_advanced_features`
   turn those absolute averages into home-vs-away comparison features
   (ratios, log-ratios, differences, and composite domain metrics), which
   carry far more predictive signal than the raw statistics alone.
"""

import numpy as np
import pandas as pd

TARGET_COLUMNS = ["home_score", "away_score", "total", "margin"]

# Per-game performance metrics that get rolling averages
# (cumulative stats such as wins/losses are excluded).
STATS_TO_AVERAGE = [
    # Hitter stats
    "hitters_AB", "hitters_H", "hitters_RBI", "hitters_K",
    "hitters_#P", "hitters_AVG", "hitters_OBP", "hitters_SLG",
    # Pitcher stats
    "pitchers_IP", "pitchers_H", "pitchers_ER", "pitchers_BB",
    "pitchers_K", "pitchers_HR", "pitchers_PC", "pitchers_ST",
    # General stats
    "sb", "tb",
]


def calculate_running_averages(merged_df, windows=(3, 5, 10)):
    """Calculate trailing rolling averages of team statistics.

    For each team and each statistic, computes the mean over the previous
    ``window`` games (``shift(1)`` excludes the current game, preventing
    target leakage) and attaches the result to the corresponding home/away
    column of each game row.

    Parameters
    ----------
    merged_df : pd.DataFrame
        Game-level data indexed by Game, as produced by
        :func:`mlb_prediction.data.build_game_dataset`.
    windows : sequence of int
        Window sizes for the rolling averages.

    Returns
    -------
    pd.DataFrame
        ``merged_df`` with ``{home,away}_{stat}_avg_{window}`` columns added.
    """
    # Reshape to long format: one row per team per game
    records = []
    for idx, row in merged_df.iterrows():
        for venue in ("home", "away"):
            record = {
                "game_id": idx,
                "date": row["Date"],
                "team": row[venue],
                "venue": venue,
            }
            for stat in STATS_TO_AVERAGE:
                col_name = f"{venue}_{stat}"
                if col_name in row.index:
                    record[stat] = row[col_name]
            records.append(record)

    long_df = pd.DataFrame(records)
    long_df = long_df.sort_values(["team", "date"])

    # Rolling averages per team; shift(1) so only prior games are used
    for window in windows:
        for stat in STATS_TO_AVERAGE:
            if stat in long_df.columns:
                long_df[f"{stat}_avg_{window}"] = long_df.groupby("team")[stat].transform(
                    lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
                )

    # Pivot back to wide format, collected in one dict to avoid fragmentation
    new_columns = {}
    for venue in ("home", "away"):
        venue_df = long_df[long_df["venue"] == venue].set_index("game_id")
        for window in windows:
            for stat in STATS_TO_AVERAGE:
                col_name = f"{stat}_avg_{window}"
                if col_name in venue_df.columns:
                    new_columns[f"{venue}_{col_name}"] = venue_df[col_name]

    new_cols_df = pd.DataFrame(new_columns, index=merged_df.index)
    return pd.concat([merged_df, new_cols_df], axis=1)


def prepare_modeling_data(running_df):
    """Keep only historical-average features and the four targets.

    Adds the derived ``total`` and ``margin`` targets, restricts the
    dataframe to rolling-average columns (the only features known before
    a game starts), and drops early-season rows without enough history.
    """
    running_df = running_df.copy()
    running_df["total"] = running_df.home_score + running_df.away_score
    running_df["margin"] = (running_df.home_score - running_df.away_score).abs()

    avg_columns = [col for col in running_df.columns if "_avg_" in col]
    modeling_df = running_df[TARGET_COLUMNS + avg_columns].copy()

    return modeling_df.dropna()


def create_ratio_features(modeling_df, include_differences=True, clip_ratios=True, max_ratio=5.0):
    """Create ratio, log-ratio, and difference features between home and away stats.

    Counting stats use Laplace smoothing (+1) and rate stats a small
    epsilon so near-zero denominators cannot blow up; ratios are then
    clipped to ``[1/max_ratio, max_ratio]``.

    Returns
    -------
    (enhanced_df, new_features) : tuple of (pd.DataFrame, list of str)
    """
    enhanced_df = modeling_df.copy()

    home_features = [
        col for col in modeling_df.columns if col.startswith("home_") and "_avg_" in col
    ]
    new_features = []

    for home_col in home_features:
        away_col = home_col.replace("home_", "away_")
        if away_col not in modeling_df.columns:
            continue

        feature_name = home_col.replace("home_", "")
        home_vals = enhanced_df[home_col]
        away_vals = enhanced_df[away_col]

        if any(stat in feature_name for stat in ["AB", "H", "RBI", "K", "HR", "BB"]):
            # Counting stats: Laplace smoothing
            ratio = (home_vals + 1) / (away_vals + 1)
        else:
            # Rate stats (AVG, OBP, SLG): larger epsilon
            ratio = (home_vals + 0.01) / (away_vals + 0.01)

        if clip_ratios:
            ratio = np.clip(ratio, 1 / max_ratio, max_ratio)

        enhanced_df[f"ratio_{feature_name}"] = ratio
        enhanced_df[f"log_ratio_{feature_name}"] = np.log(ratio)
        new_features.extend([f"ratio_{feature_name}", f"log_ratio_{feature_name}"])

        if include_differences:
            enhanced_df[f"diff_{feature_name}"] = home_vals - away_vals
            new_features.append(f"diff_{feature_name}")

    print(f"Created {len(new_features)} new features")
    return enhanced_df, new_features


def create_advanced_features(modeling_df, windows=("10",)):
    """Create composite features based on baseball domain knowledge.

    - Offensive Power Ratio: (AVG * SLG * OBP) home vs away
    - Pitching Dominance:     K/BB ratio comparison
    - Run Efficiency:         RBI per hit comparison
    - Matchup Advantage:      each offense vs the opposing pitching staff

    Returns
    -------
    (enhanced_df, new_features) : tuple of (pd.DataFrame, list of str)
    """
    enhanced_df = modeling_df.copy()
    new_features = []

    for window in windows:
        # 1. Offensive Power Ratio (combines multiple hitting stats)
        home_offensive = (
            enhanced_df[f"home_hitters_AVG_avg_{window}"]
            * enhanced_df[f"home_hitters_SLG_avg_{window}"]
            * enhanced_df[f"home_hitters_OBP_avg_{window}"]
        )
        away_offensive = (
            enhanced_df[f"away_hitters_AVG_avg_{window}"]
            * enhanced_df[f"away_hitters_SLG_avg_{window}"]
            * enhanced_df[f"away_hitters_OBP_avg_{window}"]
        )
        feature_name = f"offensive_power_ratio_avg_{window}"
        enhanced_df[feature_name] = (home_offensive + 0.001) / (away_offensive + 0.001)
        new_features.append(feature_name)

        # 2. Pitching Dominance Score (K/BB ratio comparison)
        if f"home_pitchers_K_avg_{window}" in enhanced_df.columns:
            home_k_bb = (enhanced_df[f"home_pitchers_K_avg_{window}"] + 1) / (
                enhanced_df[f"home_pitchers_BB_avg_{window}"] + 1
            )
            away_k_bb = (enhanced_df[f"away_pitchers_K_avg_{window}"] + 1) / (
                enhanced_df[f"away_pitchers_BB_avg_{window}"] + 1
            )
            feature_name = f"pitching_dominance_ratio_avg_{window}"
            enhanced_df[feature_name] = home_k_bb / (away_k_bb + 0.1)
            new_features.append(feature_name)

        # 3. Run Production Efficiency (RBI per hit)
        if f"home_hitters_H_avg_{window}" in enhanced_df.columns:
            home_eff = (enhanced_df[f"home_hitters_RBI_avg_{window}"] + 1) / (
                enhanced_df[f"home_hitters_H_avg_{window}"] + 1
            )
            away_eff = (enhanced_df[f"away_hitters_RBI_avg_{window}"] + 1) / (
                enhanced_df[f"away_hitters_H_avg_{window}"] + 1
            )
            feature_name = f"run_efficiency_ratio_avg_{window}"
            enhanced_df[feature_name] = home_eff / (away_eff + 0.1)
            new_features.append(feature_name)

        # 4. Matchup Strength (offense vs the opposing pitching staff)
        home_vs_away = enhanced_df[f"home_hitters_AVG_avg_{window}"] / (
            enhanced_df[f"away_pitchers_H_avg_{window}"]
            / enhanced_df[f"away_pitchers_IP_avg_{window}"]
            + 0.1
        )
        away_vs_home = enhanced_df[f"away_hitters_AVG_avg_{window}"] / (
            enhanced_df[f"home_pitchers_H_avg_{window}"]
            / enhanced_df[f"home_pitchers_IP_avg_{window}"]
            + 0.1
        )

        enhanced_df[f"home_matchup_advantage_avg_{window}"] = home_vs_away
        enhanced_df[f"away_matchup_advantage_avg_{window}"] = away_vs_home
        enhanced_df[f"matchup_ratio_avg_{window}"] = home_vs_away / (away_vs_home + 0.1)
        new_features.extend(
            [
                f"home_matchup_advantage_avg_{window}",
                f"away_matchup_advantage_avg_{window}",
                f"matchup_ratio_avg_{window}",
            ]
        )

    print(f"Created {len(new_features)} advanced features")
    return enhanced_df, new_features


def build_feature_matrix(merged_df, windows=(10,)):
    """Full feature pipeline: rolling averages -> modeling frame -> engineered features."""
    running_df = calculate_running_averages(merged_df, windows=windows)
    modeling_df = prepare_modeling_data(running_df)
    enhanced_df, _ = create_ratio_features(modeling_df)
    enhanced_df, _ = create_advanced_features(enhanced_df)
    return enhanced_df
