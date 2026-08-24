"""Exploratory data analysis: summary statistics, correlations, and plots."""

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats

from .features import TARGET_COLUMNS


def perform_eda(modeling_df, figsize=(16, 12)):
    """Run a full EDA pass over the modeling dataframe.

    Prints dataset/target/feature summaries, draws a 3x3 grid of
    distribution and correlation plots, and returns key statistics.

    Returns
    -------
    dict
        EDA results (target statistics, home win rate, top correlations).
    """
    eda_results = {}

    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"Shape: {modeling_df.shape[0]} games, {modeling_df.shape[1]} columns")
    print(f"Features: {modeling_df.shape[1] - len(TARGET_COLUMNS)}")
    print(f"Targets: {len(TARGET_COLUMNS)} ({', '.join(TARGET_COLUMNS)})")
    print(f"Memory usage: {modeling_df.memory_usage().sum() / 1024**2:.2f} MB")

    print("\n" + "=" * 60)
    print("TARGET VARIABLE ANALYSIS")
    print("=" * 60)

    target_stats = modeling_df[TARGET_COLUMNS].describe()
    print("\nTarget Statistics:")
    print(target_stats)

    modeling_df = modeling_df.copy()
    modeling_df["score_diff"] = modeling_df["home_score"] - modeling_df["away_score"]
    print(f"\nHome advantage: {(modeling_df['score_diff'] > 0).mean():.1%} of games won by home team")
    print(f"Average score differential: {modeling_df['score_diff'].mean():.2f} runs (positive = home advantage)")
    print(f"Average total runs per game: {modeling_df['total'].mean():.2f}")
    print(f"Average margin of victory: {modeling_df['margin'].mean():.2f} runs")
    print(f"Blowout games (margin > 5): {(modeling_df['margin'] > 5).mean():.1%}")
    print(f"Close games (margin <= 2): {(modeling_df['margin'] <= 2).mean():.1%}")

    eda_results["target_stats"] = target_stats
    eda_results["home_win_pct"] = (modeling_df["score_diff"] > 0).mean()

    print("\n" + "=" * 60)
    print("FEATURE ANALYSIS")
    print("=" * 60)

    feature_cols = [
        col for col in modeling_df.columns if col not in TARGET_COLUMNS + ["score_diff"]
    ]
    hitter_features = [col for col in feature_cols if "hitters" in col]
    pitcher_features = [col for col in feature_cols if "pitchers" in col]
    other_features = [col for col in feature_cols if col not in hitter_features + pitcher_features]

    print("\nFeature breakdown:")
    print(f"  Hitter features: {len(hitter_features)}")
    print(f"  Pitcher features: {len(pitcher_features)}")
    print(f"  Other features: {len(other_features)}")

    _plot_eda_grid(modeling_df, feature_cols, figsize)

    print("\n" + "=" * 60)
    print("FEATURE CORRELATIONS")
    print("=" * 60)

    correlations = modeling_df[feature_cols + TARGET_COLUMNS].corr()
    for target in TARGET_COLUMNS:
        target_corr = correlations[target].drop(TARGET_COLUMNS).sort_values(ascending=False)
        print(f"\nTop 10 features correlated with {target}:")
        for feat, corr in target_corr.head(10).items():
            print(f"  {feat}: {corr:.3f}")
        eda_results[f"top_{target}_correlations"] = target_corr.head(10)

    print("\n" + "=" * 60)
    print("FEATURE VALUE RANGES")
    print("=" * 60)

    feature_stats = modeling_df[feature_cols].describe()
    low_variance_features = [
        col for col in feature_cols if feature_stats.loc["std", col] < 0.01
    ]
    if low_variance_features:
        print(f"\nWarning: {len(low_variance_features)} features have very low variance")
        print(f"Examples: {low_variance_features[:5]}")

    skewed_features = [
        (col, stats.skew(modeling_df[col]))
        for col in feature_cols
        if abs(stats.skew(modeling_df[col])) > 2
    ]
    if skewed_features:
        print(f"\n{len(skewed_features)} features are highly skewed (|skew| > 2)")
        print("Top 5 most skewed:")
        for feat, skew in sorted(skewed_features, key=lambda x: abs(x[1]), reverse=True)[:5]:
            print(f"  {feat}: {skew:.2f}")

    return eda_results


def _plot_eda_grid(modeling_df, feature_cols, figsize):
    """Draw the 3x3 grid of EDA plots."""
    plt.figure(figsize=figsize)

    plt.subplot(3, 3, 1)
    modeling_df[["home_score", "away_score"]].plot(kind="hist", bins=20, alpha=0.7, ax=plt.gca())
    plt.title("Score Distributions")
    plt.xlabel("Runs Scored")
    plt.ylabel("Frequency")
    plt.legend(["Home", "Away"])

    plt.subplot(3, 3, 2)
    modeling_df["total"].plot(kind="hist", bins=25, color="purple", alpha=0.7)
    plt.axvline(modeling_df["total"].mean(), color="red", linestyle="--",
                label=f"Mean: {modeling_df['total'].mean():.1f}")
    plt.title("Total Runs per Game Distribution")
    plt.xlabel("Total Runs")
    plt.ylabel("Frequency")
    plt.legend()

    plt.subplot(3, 3, 3)
    modeling_df["margin"].plot(kind="hist", bins=20, color="orange", alpha=0.7)
    plt.axvline(modeling_df["margin"].mean(), color="red", linestyle="--",
                label=f"Mean: {modeling_df['margin'].mean():.1f}")
    plt.title("Margin of Victory Distribution")
    plt.xlabel("Run Differential")
    plt.ylabel("Frequency")
    plt.legend()

    plt.subplot(3, 3, 4)
    modeling_df["score_diff"].plot(kind="hist", bins=30, color="green", alpha=0.7)
    plt.axvline(0, color="red", linestyle="--", label="Even")
    plt.title("Score Differential (Home - Away)")
    plt.xlabel("Score Differential")
    plt.ylabel("Frequency")
    plt.legend()

    plt.subplot(3, 3, 5)
    scatter = plt.scatter(modeling_df["home_score"], modeling_df["away_score"],
                          c=modeling_df["total"], cmap="viridis", alpha=0.5)
    plt.plot([0, 20], [0, 20], "r--", label="Equal scores")
    plt.xlabel("Home Score")
    plt.ylabel("Away Score")
    plt.title("Home vs Away Scores (colored by total)")
    plt.colorbar(scatter, label="Total Runs")
    plt.legend()

    plt.subplot(3, 3, 6)
    plt.scatter(modeling_df["total"], modeling_df["margin"], alpha=0.5)
    plt.xlabel("Total Runs")
    plt.ylabel("Margin of Victory")
    plt.title("Total Runs vs Margin of Victory")

    sample_features = feature_cols[::4][:8]

    plt.subplot(3, 3, 7)
    corr_with_total = modeling_df[sample_features + ["total"]].corr()["total"].drop("total")
    corr_with_total.plot(kind="barh")
    plt.title("Sample Feature Correlations with Total Runs")
    plt.xlabel("Correlation")

    plt.subplot(3, 3, 8)
    corr_with_margin = modeling_df[sample_features + ["margin"]].corr()["margin"].drop("margin")
    corr_with_margin.plot(kind="barh")
    plt.title("Sample Feature Correlations with Margin")
    plt.xlabel("Correlation")

    plt.subplot(3, 3, 9)
    home_scores = modeling_df["home_score"].values
    away_scores = modeling_df["away_score"].values
    plt.boxplot([home_scores, away_scores], positions=[1, 2], widths=0.6, patch_artist=True,
                boxprops=dict(facecolor="lightblue", alpha=0.7),
                medianprops=dict(color="red", linewidth=2),
                flierprops=dict(marker="o", markersize=4, alpha=0.5))
    plt.scatter([1, 2], [np.mean(home_scores), np.mean(away_scores)],
                color="green", s=100, marker="D", label="Mean", zorder=10)
    plt.xticks([1, 2], ["Home Team", "Away Team"])
    plt.ylabel("Runs Scored")
    plt.title("Score Distribution: Home vs Away")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.text(1.5, plt.ylim()[1] * 0.95,
             f"Home advantage: +{np.mean(home_scores) - np.mean(away_scores):.2f} runs",
             ha="center", fontsize=10, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    plt.show()


def create_correlation_heatmap(modeling_df, feature_subset=20, figsize=(12, 10)):
    """Heatmap of the features most correlated (on average) with the targets."""
    feature_cols = [col for col in modeling_df.columns if col not in TARGET_COLUMNS]
    correlations = modeling_df[feature_cols + TARGET_COLUMNS].corr()

    avg_target_corr = sum(abs(correlations[target]) for target in TARGET_COLUMNS) / len(
        TARGET_COLUMNS
    )
    top_features = avg_target_corr.drop(TARGET_COLUMNS).nlargest(feature_subset).index.tolist()

    subset_corr = modeling_df[top_features + TARGET_COLUMNS].corr()

    plt.figure(figsize=figsize)
    sns.heatmap(subset_corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title(f"Correlation Heatmap (Top {feature_subset} Features)")
    plt.tight_layout()
    plt.show()


def analyze_feature_importance(enhanced_df, target="total", top_n=20):
    """Rank features by correlation with a target; returns the correlation series."""
    feature_cols = [col for col in enhanced_df.columns if col not in TARGET_COLUMNS]
    correlations = enhanced_df[feature_cols + [target]].corr()[target].drop(target)

    print(f"\nTop {top_n} Features Correlated with {target}:")
    print("=" * 60)
    print("\nPositive Correlations:")
    for feat, corr in correlations.nlargest(top_n).items():
        print(f"  {feat:<50} {corr:>6.3f}")
    print("\nNegative Correlations:")
    for feat, corr in correlations.nsmallest(top_n).items():
        print(f"  {feat:<50} {corr:>6.3f}")

    ratio_features = [col for col in feature_cols if "ratio_" in col or "diff_" in col]
    if ratio_features:
        best_ratio = correlations[ratio_features].abs().nlargest(5)
        print("\nBest Ratio/Difference Features:")
        for feat, _ in best_ratio.items():
            print(f"  {feat:<50} {correlations[feat]:>6.3f}")

    return correlations


def visualize_feature_engineering_impact(original_df, enhanced_df, target="total"):
    """Compare correlation strength of original vs engineered features."""
    original_features = [col for col in original_df.columns if col not in TARGET_COLUMNS]
    original_corrs = original_df[original_features + [target]].corr()[target].drop(target)

    new_features = [col for col in enhanced_df.columns if col not in original_df.columns]
    new_corrs = enhanced_df[new_features + [target]].corr()[target].drop(target)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    axes[0].hist(original_corrs.abs(), bins=30, alpha=0.7, color="blue", edgecolor="black")
    axes[0].axvline(original_corrs.abs().mean(), color="red", linestyle="--",
                    label=f"Mean: {original_corrs.abs().mean():.3f}")
    axes[0].set_title("Original Features - Absolute Correlations")
    axes[0].set_xlabel("|Correlation|")
    axes[0].set_ylabel("Count")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if len(new_corrs) > 0:
        axes[1].hist(new_corrs.abs(), bins=30, alpha=0.7, color="green", edgecolor="black")
        axes[1].axvline(new_corrs.abs().mean(), color="red", linestyle="--",
                        label=f"Mean: {new_corrs.abs().mean():.3f}")
        axes[1].set_title("Engineered Features - Absolute Correlations")
        axes[1].set_xlabel("|Correlation|")
        axes[1].set_ylabel("Count")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        print("\nFeature Engineering Impact Summary:")
        print(f"Original features: Max correlation = {original_corrs.abs().max():.3f}")
        print(f"Engineered features: Max correlation = {new_corrs.abs().max():.3f}")
        print(f"Improvement: {new_corrs.abs().max() - original_corrs.abs().max():.3f}")

    plt.tight_layout()
    plt.show()
