#exploratory analysis analytics
import matplotlib.pyplot as plt
import pandas as pd      # ←-- make sure this is imported if you call binned_points_distribution

# ──────────────────────────────────────────────────────────────────────────────
# ▶ BOTH (location + spread)  — quantiles include the median (center) and tails
def points_percentiles(df, col='points', percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]):
    """
    Returns specified percentiles for the points column in the DataFrame.

    Statistic type
    --------------
    • 5-th/25-th/75-th/95-th  → spread
    • 50-th (median)         → location
    """
    return df[col].quantile(percentiles)

# ──────────────────────────────────────────────────────────────────────────────
# ▶ BOTH (location + spread)  — median line = location, box/IQR & whiskers = spread
def plot_points_box(df, col='points', title='Distribution of Player Points'):
    """
    Plots a box plot for the points column.

    Statistic type
    --------------
    • Median line            → location
    • IQR & whiskers         → spread
    """
    ax = df[col].plot.box()
    ax.set_ylabel('Points')
    ax.set_title(title)
    plt.show()
    return ax

# ──────────────────────────────────────────────────────────────────────────────
# ▶ SPREAD  — shape/width show dispersion; center is secondary
def plot_points_hist(df, col='points', bins=15, title='Histogram of Player Points'):
    """
    Plots a histogram for the points column.

    Statistic type
    --------------
    • Bar width & silhouette → spread (variance, skew)
    """
    ax = df[col].plot.hist(bins=bins, figsize=(5, 4),
                           alpha=0.7, edgecolor='black')
    ax.set_xlabel('Points')
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    plt.show()
    return ax

# ──────────────────────────────────────────────────────────────────────────────
# ▶ SPREAD  — same information as a histogram but returned as counts per bin
def binned_points_distribution(df, col='points', bins=10):
    """
    Returns value counts for points binned into intervals.

    Statistic type
    --------------
    • Count per interval     → spread (distribution profile)
    """
    binned = pd.cut(df[col], bins=bins)
    return binned.value_counts().sort_index()

# ──────────────────────────────────────────────────────────────────────────────
# ▶ SPREAD  — histogram + smooth KDE visualise dispersion/shape
def plot_points_hist_kde(df, col='points', bins=30,
                         title='Points Distribution (Histogram + KDE)'):
    """
    Plots a histogram with density=True and overlays a KDE for the points column.

    Statistic type
    --------------
    • Histogram & KDE curve  → spread (variance, multimodality, tails)
    """
    ax = df[col].plot.hist(density=True, bins=bins,
                           alpha=0.6, edgecolor='black',
                           figsize=(5, 4))
    df[col].plot.density(ax=ax)
    ax.set_xlabel('Points')
    ax.set_ylabel('Density')
    ax.set_title(title)
    plt.show()
    return ax

# ──────────────────────────────────────────────────────────────────────────────
# ▶ LOCATION  — bar height is the mean save-percentage per team-year
def plot_top_save_pct_teams_year(df, col_team='team', col_save='save_percentage',
                                 col_year='year', top_n=10):
    """
    Plots a bar chart of the top-N (team, year) pairs by average save percentage.

    Statistic type
    --------------
    • Mean save-percentage   → location (no spread shown unless error bars added)
    """
    subset = df[[col_team, col_save, col_year]].copy()
    subset[col_save] = pd.to_numeric(subset[col_save], errors='coerce')
    subset = subset.dropna(subset=[col_save, col_team, col_year])

    # Compute mean save % for every (team, year) combo
    group_avg = (subset
                 .groupby([col_team, col_year])[col_save]
                 .mean()
                 .sort_values(ascending=False))
    top_group = group_avg.head(top_n)

    # Human-friendly x-tick labels: "Team (Year)"
    labels = [f"{team} ({year})" for (team, year) in top_group.index]

    ax = top_group.plot.bar(figsize=(10, 5), color='slateblue', legend=False)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_xlabel('Team (Year)')
    ax.set_ylabel('Average Save Percentage')
    ax.set_title(f'Top {top_n} Team-Years by Average Save Percentage')
    plt.tight_layout()
    plt.show()
    return ax

sns.set_theme(style="whitegrid", palette="colorblind")
HIGHLIGHT = sns.color_palette("colorblind")[2]  # accent for Ovi
MUTED     = sns.color_palette("colorblind")[0]

def top_goals_chart(df, top_n=3, highlight_name="Alex Ovechkin"):
    d = df.copy()
    d["goals"] = pd.to_numeric(d["goals"], errors="coerce")
    d = d.dropna(subset=["player", "goals"])

    # aggregate if multiple rows per player
    d = d.groupby("player", as_index=False)["goals"].max()

    # get top_n sorted descending
    top = d.sort_values("goals", ascending=False).head(top_n)

    # colors: highlight Ovi
    colors = [HIGHLIGHT if p == highlight_name else MUTED for p in top["player"]]

    plt.figure(figsize=(9, 4))
    ax = sns.barplot(
        data=top,
        x="goals",
        y="player",
        orient="h",
        palette=colors,
        order=top.sort_values("goals", ascending=False)["player"]  # sort DESC so highest at top
    )
    ax.set_title(f"Top {top_n} Career Goals — Drafted Players (1963–2022 dataset)")
    ax.set_xlabel("Career goals (dataset)")
    ax.set_ylabel("")

    for c in ax.containers:
        ax.bar_label(c, fmt="%.0f", padding=3)

    ax.text(
        0.0, -0.15,
        "Note: single Kaggle dataset; not league-official and not cross-validated.",
        transform=ax.transAxes, ha="left", va="top", fontsize=9, color="gray"
    )

    plt.tight_layout()
    plt.show()

# Example call
top_goals_chart(hockey_clean, top_n=3, highlight_name="Alex Ovechkin")

# ──────────────────────────────────────────────────────────────────────────────
# Example usage (uncomment to run):
plot_top_save_pct_teams_year(hockey_clean)
points_percentiles(hockey_clean)
plot_points_box(hockey_clean)
plot_points_hist(hockey_clean)
plot_points_hist_kde(hockey_clean)
