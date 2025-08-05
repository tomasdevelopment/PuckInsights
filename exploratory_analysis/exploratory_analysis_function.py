import pandas as pd
from scipy import stats

def hockey_overview(df: pd.DataFrame, trim_prop: float = 0.10) -> dict:
    """Clean `points`, compute robust stats, create slices, and report memory."""

    # ---------- 1. Hygiene --------------------------------------------------- #
    df = df.copy()                               # avoid mutating caller’s frame
    df['points'] = pd.to_numeric(df['points'], errors='coerce')
    df_clean     = df.dropna(subset=['points'])

    pts = df_clean['points']

    # ---------- 2. Robust summary ------------------------------------------- #
    q1, q3 = pts.quantile([.25, .75])
    iqr    = q3 - q1
    summary = {
        "count"        : int(pts.count()),
        "mean"         : pts.mean(),
        "median"       : pts.median(),
        "trimmed_mean" : stats.trim_mean(pts, trim_prop),
        "mad"          : stats.median_abs_deviation(pts, scale='normal'),
        "variance"     : pts.var(),
        "std_dev"      : pts.std(),
        "iqr"          : iqr,
        "lower_bound"  : q1 - 1.5 * iqr,
        "upper_bound"  : q3 + 1.5 * iqr,
    }

    # ---------- 3. Interesting slices --------------------------------------- #
    upper = df_clean[pts >= summary['upper_bound']].sort_values(['year','points'], ascending=False)
    lower = df_clean[pts <= summary['lower_bound']].sort_values(['points','year'])

    max_pts  = pts.max()
    min_pts  = pts.min()

    top_same    = df_clean[pts == max_pts]
    bottom_same = df_clean[pts == min_pts]

    nonzero        = pts[pts != 0]
    min_nonzero    = nonzero.min() if not nonzero.empty else None
    bottom_no_zero = df_clean[pts == min_nonzero] if min_nonzero is not None else pd.DataFrame()

    # ---------- 4. Generic overview ----------------------------------------- #
    mem_bytes   = df_clean.memory_usage(deep=True).sum()
    mem_mb      = mem_bytes / 1024 ** 2
    row_size_b  = mem_bytes / len(df_clean) if len(df_clean) else 0

    overview = {
        # data size
        'num_rows'       : len(df_clean),
        'num_cols'       : df_clean.shape[1],
        'memory_MB'      : round(mem_mb, 3),
        'bytes_per_row'  : round(row_size_b, 1),

        # dtypes / schema
        'dtypes'         : df_clean.dtypes.to_dict(),
        'null_counts'    : df_clean.isnull().sum().to_dict(),

        # categorical quick-look (defensive for missing cols)
        'top_teams'      : df_clean['team'].value_counts().head(5).to_dict() \
                           if 'team' in df_clean.columns else {},
        'top_players'    : df_clean['player'].value_counts().head(5).to_dict() \
                           if 'player' in df_clean.columns else {},

        # numerical high-level stats
        'points_summary' : summary,

        # outlier collections
        'upper_players'  : upper[['year','player','team','points']].to_dict('records'),
        'lower_players'  : lower[['year','player','team','points']].to_dict('records'),
        'top_players'    : top_same[['year','player','team','points']].to_dict('records'),
        'bottom_players' : bottom_same[['year','player','team','points']].to_dict('records'),
        'bottom_nonzero' : bottom_no_zero[['year','player','team','points']].to_dict('records') \
                           if not bottom_no_zero.empty else []
    }

    return overview
