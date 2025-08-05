import pandas as pd
from scipy import stats

# ---------- 1. One-time hygiene ---------- #
hockey_df['points'] = pd.to_numeric(hockey_df['points'], errors='coerce')
points                      = hockey_df['points'].dropna()        # numeric, no NaNs
hockey_df_clean             = hockey_df.loc[points.index]         # mirror rows

# ---------- 2. Compact summary ---------- #
def summarize(series: pd.Series, trim_prop: float = 0.10) -> dict:
    """Return a dictionary of robust location / spread measures."""
    q1, q3  = series.quantile([.25, .75])
    iqr     = q3 - q1
    return {
        "count"        : int(series.count()),
        "mean"         : series.mean(),
        "median"       : series.median(),
        "trimmed_mean" : stats.trim_mean(series, trim_prop),
        "mad"          : stats.median_abs_deviation(series, scale='normal'),
        "variance"     : series.var(),
        "std_dev"      : series.std(),
        "iqr"          : iqr,
        "lower_bound"  : q1 - 1.5 * iqr,
        "upper_bound"  : q3 + 1.5 * iqr,
    }

summary = summarize(points)

# ---------- 3. Interesting slices ---------- #
upper_players  = hockey_df_clean[hockey_df_clean['points'] >= summary['upper_bound']]
lower_players  = hockey_df_clean[hockey_df_clean['points'] <= summary['lower_bound']]

max_pts        = points.max()
top_players    = hockey_df_clean[hockey_df_clean['points'] == max_pts]

min_pts        = points.min()
bottom_players = hockey_df_clean[hockey_df_clean['points'] == min_pts]

nonzero        = points[points != 0]
min_nonzero    = nonzero.min() if not nonzero.empty else None
bottom_no_zero = hockey_df_clean[hockey_df_clean['points'] == min_nonzero] if min_nonzero is not None else pd.DataFrame()

# ---------- 4. Quick pulse-check ---------- #
print("Basic points summary:\n", summary)
print(f"\nTop performer(s) ({max_pts} pts):", top_players[['year','player','team','points']].to_dict('records'))
print(f"Low performer(s) ({min_pts} pts incl. zeros):", bottom_players[['year','player','team','points']].to_dict('records'))
if not bottom_no_zero.empty:
    print(f"Lowest non-zero performer(s) ({min_nonzero} pts):", bottom_no_zero[['year','player','team','points']].to_dict('records'))
