def plot_top_by_cohort_career(df, threshold=650, top_after=5, top_before=10, first_year_col="year"):
    """
    Aggregate points by player across seasons; define player cohort by first season year.
    """
    req_cols = {"year", "player", "points"}
    missing = req_cols - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    df = ensure_numeric(df, "points")
    df = ensure_numeric(df, "year")

    # Career totals & first season (to define cohort)
    first_year = df.groupby("player")["year"].min().rename("first_year")
    career = (
        df.groupby("player", as_index=False)["points"].sum()
          .merge(first_year, on="player")
    )
    high = career[career["points"] >= threshold]
    if high.empty:
        print(f"No players with career points >= {threshold}.")
        return None

    post2000 = (
        high[high["first_year"] >= 2000]
        .sort_values("points", ascending=False)
        .head(top_after)
        .assign(cohort=f"Debut ≥2000 (top {top_after})")
    )
    pre2000 = (
        high[high["first_year"] < 2000]
        .sort_values("points", ascending=False)
        .head(top_before)
        .assign(cohort=f"Debut <2000 (top {top_before})")
    )
    compare = pd.concat([post2000, pre2000], ignore_index=True)

    summaries = cohort_summaries(compare.rename(columns={"points":"points"}))
    print("Per-cohort location & dispersion (career):")
    print(summaries)

    # Side-by-side bars
    sns.set_theme()
    g = sns.catplot(
        data=compare,
        x="points", y="player",
        col="cohort", kind="bar", sharex=True, sharey=False,
        height=5, aspect=0.9
    )
    g.set_axis_labels("Career points", "")
    g.set_titles("{col_name}")
    for ax in g.axes.flatten():
        ax.bar_label(ax.containers[0], fmt="%.0f")
    plt.tight_layout()
    plt.show()

    # Distributions
    bins = max(5, int(np.sqrt(len(compare))))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
    for i, (label, sub) in enumerate(compare.groupby("cohort")):
        sns.histplot(sub["points"], bins=bins, stat="density", ax=axes[i])
        try:
            sns.kdeplot(sub["points"], ax=axes[i])
        except Exception:
            pass
        axes[i].set_title(f"{label}: career distribution")
        axes[i].set_xlabel("Career points")
        axes[i].set_ylabel("Density" if i == 0 else "")
    plt.tight_layout()
    plt.show()

    return summaries, compare

# Example:
summaries_career, compare_career = plot_top_by_cohort_career(hockey_clean, threshold=650, top_after=5, top_before=10)


#run it
out = season_comparison_simple(hockey_clean, cutoff_year=2000, top_before=10, top_after=5)
out["stats"]        # mean/median per cohort
out["top_pre"]      # top-10 seasons before 2000
out["top_post"]     # top-5 seasons since 2000
