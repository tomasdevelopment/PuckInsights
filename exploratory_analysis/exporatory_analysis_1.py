import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple

# --------------------------------------------------------------------------- #
# 1.  Cleaning layer (unchanged from last fix)                                #
# --------------------------------------------------------------------------- #
def clean_hockey_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()
    df.dropna(how='all', inplace=True)

    # points → numeric, NaN→0, cast to int
    if 'points' not in df.columns:
        raise KeyError("'points' column not found")
    df['points'] = (
        pd.to_numeric(df['points'], errors='coerce')
        .fillna(0)
        .astype(int)
    )
    return df


# --------------------------------------------------------------------------- #
# 2A.  Summary-only helper                                                    #
# --------------------------------------------------------------------------- #
def points_summary(
    pts: pd.Series, *, trim_prop: float = 0.10
) -> Dict[str, float | int]:
    """
    Return location & dispersion stats for a points Series.
    """
    q1, q3 = pts.quantile([0.25, 0.75])
    iqr = q3 - q1
    return {
        "count"        : int(pts.count()),
        "mean"         : float(pts.mean()),
        "median"       : int(pts.median()),
        "trimmed_mean" : float(stats.trim_mean(pts, trim_prop)),
        "mad"          : float(stats.median_abs_deviation(pts, scale="normal")),
        "variance"     : float(pts.var()),
        "std_dev"      : float(pts.std()),
        "iqr"          : int(iqr),
        "lower_bound"  : int(q1 - 1.5 * iqr),
        "upper_bound"  : int(q3 + 1.5 * iqr),
    }


# --------------------------------------------------------------------------- #
# 2B.  Group-slices helper                                                    #
# --------------------------------------------------------------------------- #
def points_groups(
    df_clean: pd.DataFrame,
    summary: Dict[str, float | int],
) -> Dict[str, List[dict]]:
    """
    Return dict of interesting player slices, already sorted.
    """
    pts = df_clean["points"]

    def keep_cols(df: pd.DataFrame) -> List[dict]:
        return df[["year", "player", "team", "points"]].to_dict("records")

    upper = df_clean[pts >= summary["upper_bound"]]\
            .sort_values(["year", "points"], ascending=False)

    lower = df_clean[pts <= summary["lower_bound"]]\
            .sort_values(["points", "year"], ascending=True)

    max_pts, min_pts = int(pts.max()), int(pts.min())

    return {
        "upper_players"   : keep_cols(upper),
        "lower_players"   : keep_cols(lower),
        "top_players"     : keep_cols(df_clean[pts == max_pts]),
        "bottom_players"  : keep_cols(df_clean[pts == min_pts]),
        "bottom_nonzero"  : keep_cols(df_clean[pts == pts[pts != 0].min()])
                            if (pts != 0).any() else [],
    }


# --------------------------------------------------------------------------- #
# 3.  Public façade — returns *two* objects                                   #
# --------------------------------------------------------------------------- #
def hockey_overview(
    df: pd.DataFrame, *, trim_prop: float = 0.10
) -> Tuple[Dict[str, float | int], Dict[str, List[dict]]]:
    """
    High-level wrapper that returns (summary, groups).
    """
    df_clean = clean_hockey_df(df)
    pts      = df_clean["points"]
    summary  = points_summary(pts, trim_prop=trim_prop)
    groups   = points_groups(df_clean, summary)
    return summary, groups


# --------------------------------------------------------------------------- #
# Example usage                                                               #
# --------------------------------------------------------------------------- #
FILE_PATH = "/content/drive/My Drive/sportsanalytics/nhldraft.csv"
raw_df     = pd.read_csv(FILE_PATH)

summary_dict, groups_dict = hockey_overview(raw_df)

print("== Location & dispersion statistics ==")
for k, v in summary_dict.items():
    print(f"{k:>14}: {v}")

print("\n== Upper-outlier players ==")
for rec in groups_dict["upper_players"][:5]:  # show first 5 only
    print(rec)
