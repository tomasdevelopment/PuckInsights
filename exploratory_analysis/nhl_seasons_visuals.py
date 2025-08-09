import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def _ensure_numeric(df, col):
    out = df.copy()
    out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.dropna(subset=[col])

def season_comparison_simple(
    df,
    *,
    cutoff_year=2000,
    top_before=10,
    top_after=5,
    bins=20
):
    """
    Season-based comparison: pre-2000 vs >=2000.
    Makes at most 4 plots (2 bars + 2 distributions) and prints mean/median per cohort.
    """
    req = {"year", "player", "team", "points"}
    miss = req - set(df.columns)
    if miss:
        raise KeyError(f"Missing columns: {miss}")

    df = _ensure_numeric(df, "points")
    df = _ensure_numeric(df, "year")

    # Split cohorts (season-based)
    pre = df[df["year"] < cutoff_year].copy()
    post = df[df["year"] >= cutoff_year].copy()

    # Top seasons per cohort
    top_pre  = (pre.sort_values("points", ascending=False)
                   .head(top_before)
                   .assign(cohort=f"<{cutoff_year} (top {top_before})"))
    top_post = (post.sort_values("points", ascending=False)
                   .head(top_after)
                   .assign(cohort=f"≥{cutoff_year} (top {top_after})"))

    # Nice labels: "Player (Year – Team)"
    top_pre["label"]  = top_pre.apply(lambda r: f"{r.player} ({int(r.year)} – {r.team})", axis=1)
    top_post["label"] = top_post.apply(lambda r: f"{r.player} ({int(r.year)} – {r.team})", axis=1)

    # ---- Stats (location only, keep it simple) ----
    stats = (
        pd.concat([
            pre.assign(cohort=f"<{cutoff_year}"),
            post.assign(cohort=f"≥{cutoff_year}")
        ])[["cohort", "points"]]
         .groupby("cohort")
         .agg(mean_points=("points", "mean"),
              median_points=("points", "median"),
              count=("points", "count"))
         .round(2)
         .reset_index()
    )
    print("Season points — location stats by cohort:")
    print(stats.to_string(index=False))

    # ---- Shared x-limits for fair distribution comparison ----
    xmin = df["points"].min()
    xmax = df["points"].max()

    sns.set_theme()

    # ========== FIGURE 1: Top seasons (2 bar charts) ==========
    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 6), sharex=False, sharey=False)
    # Pre-2000 bars
    ax = axes1[0]
    sns.barplot(data=top_pre, x="points", y="label", ax=ax)
    ax.set_title(f"Top Seasons < {cutoff_year}")
    ax.set_xlabel("Season points")
    ax.set_ylabel("")
    for c in ax.containers:
        ax.bar_label(c, fmt="%.0f")

    # Post-2000 bars
    ax = axes1[1]
    sns.barplot(data=top_post, x="points", y="label", ax=ax)
    ax.set_title(f"Top Seasons ≥ {cutoff_year}")
    ax.set_xlabel("Season points")
    ax.set_ylabel("")
    for c in ax.containers:
        ax.bar_label(c, fmt="%.0f")

    fig1.tight_layout()
    plt.show()

    # ========== FIGURE 2: Distributions (2 panels) ==========
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)
    # Pre
    ax = axes2[0]
    sns.histplot(pre["points"], bins=bins, stat="density", ax=ax)
    try:
        sns.kdeplot(pre["points"], ax=ax)
    except Exception:
        pass
    ax.set_title(f"Distribution < {cutoff_year}")
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel("Season points")
    ax.set_ylabel("Density")

    # Post
    ax = axes2[1]
    sns.histplot(post["points"], bins=bins, stat="density", ax=ax)
    try:
        sns.kdeplot(post["points"], ax=ax)
    except Exception:
        pass
    ax.set_title(f"Distribution ≥ {cutoff_year}")
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel("Season points")
    ax.set_ylabel("")

    fig2.tight_layout()
    plt.show()

    return {
        "stats": stats,
        "top_pre": top_pre[["player","team","year","points"]],
        "top_post": top_post[["player","team","year","points"]],
    }
out = season_comparison_simple(hockey_clean, cutoff_year=2000, top_before=10, top_after=5)
out["stats"]        # mean/median per cohort
out["top_pre"]      # top-10 seasons before 2000
out["top_post"]     # top-5 seasons since 2000
