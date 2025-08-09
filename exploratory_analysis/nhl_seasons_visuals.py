import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ---------- Setup: consistent, colorblind-safe theme ----------
sns.set_theme(style="whitegrid", palette="colorblind")
PRE_COLOR, POST_COLOR, FOCUS_2020 = sns.color_palette("colorblind")[0], sns.color_palette("colorblind")[3], sns.color_palette("colorblind")[2]

def ensure_points_per_season(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["points"] = pd.to_numeric(out["points"], errors="coerce")
    out["year"]   = pd.to_numeric(out["year"], errors="coerce")
    out["to_year"] = pd.to_numeric(out.get("to_year", np.nan), errors="coerce")
    out = out.dropna(subset=["points","year"])

    # Conservative handling: if to_year is NaN, treat it as the draft year → years_played = 1
    end_year = out["to_year"].fillna(out["year"])
    years_played = (end_year - out["year"] + 1).clip(lower=1)
    out["years_played"] = years_played
    out["points_per_season"] = out["points"] / out["years_played"]
    return out

def season_comparison_enhanced(
    df: pd.DataFrame,
    *,
    cutoff_year: int = 2000,
    top_before: int = 10,
    top_after: int = 5,
    bins: int = 24,
    roll: int = 5
):
    d = ensure_points_per_season(df)
    req = {"year","player","team","points","points_per_season"}
    miss = req - set(d.columns)
    if miss:
        raise KeyError(f"Missing columns: {miss}")

    pre  = d[d["year"] <  cutoff_year].copy()
    post = d[d["year"] >= cutoff_year].copy()

    # ====== stats table for cohorts (rate metric) ======
    stats = (
        pd.concat([pre.assign(cohort=f"<{cutoff_year}"),
                   post.assign(cohort=f"≥{cutoff_year}")])
          .groupby("cohort")
          .agg(mean_points_per_season=("points_per_season","mean"),
               median_points_per_season=("points_per_season","median"),
               mean_points=("points","mean"),
               median_points=("points","median"),
               n=("points","count"))
          .round(2)
          .reset_index()
    )

    # ====== era (decade) table on rate metric ======
    # eras: 1960s, 1970s, ..., 2020–2022 (or to your max year)
    max_year = int(d["year"].max())
    bins_ = [1963, 1969, 1979, 1989, 1999, 2009, 2019, max_year]
    labels_ = ["1960s","1970s","1980s","1990s","2000s","2010s",f"2020–{max_year}"]
    era_table = (
        d.assign(era=pd.cut(d["year"], bins=bins_, labels=labels_, include_lowest=True, right=True))
         .groupby("era", observed=True)
         .agg(mean_pps=("points_per_season","mean"),
              median_pps=("points_per_season","median"),
              players=("player","count"))
         .round(2)
         .reset_index()
    )

    # ====== Top seasons bars (colors cleaned up) ======
    top_pre  = pre.sort_values("points", ascending=False).head(top_before)
    top_post = post.sort_values("points", ascending=False).head(top_after)
    top_pre["label"]  = top_pre.apply(lambda r: f"{r.player} ({int(r.year)} – {r.team})", axis=1)
    top_post["label"] = top_post.apply(lambda r: f"{r.player} ({int(r.year)} – {r.team})", axis=1)

    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 6), sharex=False, sharey=False)
    sns.barplot(data=top_pre,  x="points", y="label", ax=axes1[0], color=PRE_COLOR)
    axes1[0].set_title(f"Top Seasons < {cutoff_year}")
    axes1[0].set_xlabel("Season points"); axes1[0].set_ylabel("")
    for c in axes1[0].containers: axes1[0].bar_label(c, fmt="%.0f")

    sns.barplot(data=top_post, x="points", y="label", ax=axes1[1], color=POST_COLOR)
    axes1[1].set_title(f"Top Seasons ≥ {cutoff_year}")
    axes1[1].set_xlabel("Season points"); axes1[1].set_ylabel("")
    for c in axes1[1].containers: axes1[1].bar_label(c, fmt="%.0f")
    fig1.tight_layout()
    plt.show()

    # ====== Distributions (use rate metric; comparable across eras) ======
    xmin = d["points_per_season"].min(); xmax = d["points_per_season"].max()
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)

    sns.histplot(pre["points_per_season"], bins=bins, stat="density", ax=axes2[0], alpha=.55, color=PRE_COLOR, edgecolor="black")
    sns.kdeplot(pre["points_per_season"], ax=axes2[0], lw=2, color=PRE_COLOR)
    axes2[0].set_title(f"Points/Season < {cutoff_year}"); axes2[0].set_xlim(xmin, xmax); axes2[0].set_xlabel("Points per season"); axes2[0].set_ylabel("Density")

    sns.histplot(post["points_per_season"], bins=bins, stat="density", ax=axes2[1], alpha=.55, color=POST_COLOR, edgecolor="black")
    sns.kdeplot(post["points_per_season"], ax=axes2[1], lw=2, color=POST_COLOR)
    axes2[1].set_title(f"Points/Season ≥ {cutoff_year}"); axes2[1].set_xlim(xmin, xmax); axes2[1].set_xlabel("Points per season"); axes2[1].set_ylabel("")
    fig2.tight_layout()
    plt.show()

    # ====== NEW: by-draft-year trend on points_per_season ======
    yearly = (
        d.groupby("year", as_index=False)
         .agg(median_pps=("points_per_season","median"),
              mean_pps=("points_per_season","mean"),
              n=("player","count"))
         .sort_values("year")
    )
    yearly["roll_median_pps"] = yearly["median_pps"].rolling(roll, min_periods=1, center=True).mean()

    fig3, ax3 = plt.subplots(figsize=(12, 5))
    sns.lineplot(data=yearly, x="year", y="median_pps", ax=ax3, lw=1.5, label="Median P/Season")
    sns.lineplot(data=yearly, x="year", y="roll_median_pps", ax=ax3, lw=3, label=f"{roll}-yr rolling median")

    # highlight ≥2020 for emphasis
    ax3.axvspan(2020, yearly["year"].max(), alpha=.15, color=FOCUS_2020, label="2020+")
    ax3.axvline(cutoff_year, ls="--", lw=1, color="gray", alpha=.8, label=f"{cutoff_year}")
    ax3.set_title("Draft-Year Trend: Points per Season (median & rolling)")
    ax3.set_xlabel("Draft year"); ax3.set_ylabel("Points per season")
    ax3.legend()
    plt.tight_layout()
    plt.show()

    return {"cohort_stats": stats, "era_table": era_table, "yearly_trend": yearly}
def season_median_comparison(df, cutoff_year=2000, roll=5):
    d = ensure_points_per_season(df)
    
    yearly = (
        d.groupby("year", as_index=False)
         .agg(median_pps=("points_per_season","median"))
    )

    # Split into pre/post cohorts for separate medians
    yearly["cohort"] = np.where(yearly["year"] < cutoff_year, f"<{cutoff_year}", f"≥{cutoff_year}")

    # Rolling medians per cohort
    rolled = (
        yearly.groupby("cohort", group_keys=False)
              .apply(lambda g: g.assign(roll_median_pps=g["median_pps"]
                                        .rolling(roll, min_periods=1, center=True).mean()))
    )

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=rolled, x="year", y="median_pps", hue="cohort", lw=1.5, style="cohort", alpha=0.6)
    sns.lineplot(data=rolled, x="year", y="roll_median_pps", hue="cohort", lw=3)

    plt.axvline(cutoff_year, ls="--", lw=1, color="gray", alpha=.8, label=f"{cutoff_year}")
    plt.axvspan(2020, rolled["year"].max(), alpha=.15, color="tab:blue", label="2020+")
    plt.title(f"Median Points per Season — {cutoff_year} Cohort Split (with {roll}-yr rolling)")
    plt.xlabel("Draft year")
    plt.ylabel("Points per season")
    plt.legend()
    plt.tight_layout()
    plt.show()

def season_median_two_lines(df, cutoff_year=2000, roll=5):
    d = ensure_points_per_season(df)

    yearly = (
        d.groupby("year", as_index=False)
         .agg(median_pps=("points_per_season", "median"))
    )
    yearly["cohort"] = np.where(yearly["year"] < cutoff_year,
                                f"<{cutoff_year}", f"≥{cutoff_year}")

    # one series per cohort: rolling median only
    rolled = (yearly.groupby("cohort", group_keys=False)
                    .apply(lambda g: g.assign(y=g["median_pps"]
                                              .rolling(roll, min_periods=1, center=True).mean())))

    plt.figure(figsize=(12, 5))
    sns.lineplot(data=rolled, x="year", y="y", hue="cohort",
                 lw=3, palette=[PRE_COLOR, POST_COLOR])
    plt.axvline(cutoff_year, ls="--", lw=1, color="gray", alpha=.8)
    plt.axvspan(2020, rolled["year"].max(), alpha=.15, color=FOCUS_2020)
    plt.title(f"Median Points per Season — {roll}-yr rolling (two cohorts)")
    plt.xlabel("Draft year"); plt.ylabel("Points per season")
    plt.legend(title="")
    plt.tight_layout(); plt.show()

# ---------- Run ----------
res = season_comparison_enhanced(hockey_clean, cutoff_year=2000, top_before=10, top_after=5, bins=24, roll=5)
season_median_comparison(hockey_clean)
season_median_two_lines(hockey_clean)
# View the tables
print("\n== Cohort stats (rate metric) ==")
print(res["cohort_stats"].to_string(index=False))

print("\n== Era table (decade buckets; rate metric) ==")
print(res["era_table"].to_string(index=False))
