# Pre-correlation diagnostics (linear vs monotonic) — matplotlib-only
# Produces: (1) Scatter+LS, (2) Residuals vs Fitted, (3) Correlation heatmap
# Prints a Pearson vs Spearman recommendation.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Optional
from scipy.stats import pearsonr, spearmanr
from scipy.stats import probplot


def _simple_corr_heatmap(df: pd.DataFrame, cols: Optional[List[str]] = None, title: str = None):
    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cols = [c for c in cols if c in df.columns]
    if len(cols) < 2:
        print("[heatmap] Need at least 2 numeric columns.")
        return
    cmat = df[cols].corr(method="pearson")
    plt.figure(figsize=(6, 5), dpi=140)
    im = plt.imshow(cmat.values, vmin=-1, vmax=1)
    plt.colorbar(im, label="pearson r")
    plt.xticks(range(len(cols)), cols, rotation=45, ha="right")
    plt.yticks(range(len(cols)), cols)
    plt.title(title or "Correlation heatmap (Pearson)")
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = cmat.values[i, j]
            if not np.isnan(v):
                plt.text(j, i, f"{v:.2f}", ha="center", va="center")
    plt.tight_layout()
    plt.show()

def pre_correlation_analysis(df: pd.DataFrame, x: str, y: str, heatmap_cols: Optional[List[str]] = None, min_n: int = 30):
    sub = df[[x, y]].dropna()
    n = len(sub)
    if n < 3:
        print(f"[pre_correlation_analysis] Not enough data for plots: n={n}")
        return

    xv = sub[x].to_numpy()
    yv = sub[y].to_numpy()

    # (1) Scatter + least-squares line
    # (1) Scatter + least-squares line and optional quadratic
    plt.figure(figsize=(5, 4), dpi=140)
    plt.scatter(xv, yv, alpha=0.5)

    xs = np.linspace(np.nanmin(xv), np.nanmax(xv), 100)
    try:
        m, b = np.polyfit(xv, yv, 1)
        plt.plot(xs, m * xs + b, linewidth=2, label="Linear fit")
    except Exception as e:
        print(f"[scatter] Fit failed: {e}")
        m, b = np.nan, np.nan

    try:
        a2, a1, a0 = np.polyfit(xv, yv, 2)
        plt.plot(xs, a2 * xs**2 + a1 * xs + a0, linestyle="--", label="Quadratic fit")
    except Exception as e:
        print(f"[scatter] Quadratic fit failed: {e}")

    r, p = pearsonr(xv, yv)
    rho, ps = spearmanr(xv, yv)
    plt.xlabel(x); plt.ylabel(y)
    plt.title(f"Scatter {x} vs {y} (r={r:.2f}, ρ={rho:.2f}, n={n})")
    plt.legend()
    plt.tight_layout(); plt.show()

    # (2) Residuals vs fitted from linear model
    if not np.isnan(m):
        yhat = m * xv + b
        resid = yv - yhat
        # pattern & nonlinearity checks
        r_res_fit = pearsonr(resid, yhat)[0] if n >= 3 else np.nan
        try:
            a2, a1, a0 = np.polyfit(xv, yv, 2)
            yhat2 = a2 * xv**2 + a1 * xv + a0
            sse1 = np.sum((yv - yhat)**2)
            sse2 = np.sum((yv - yhat2)**2)
            tss  = np.sum((yv - np.mean(yv))**2)
            r2_1 = 1 - sse1 / tss if tss > 0 else np.nan
            r2_2 = 1 - sse2 / tss if tss > 0 else np.nan
            delta_r2 = (r2_2 - r2_1) if (not np.isnan(r2_1) and not np.isnan(r2_2)) else np.nan
        except Exception:
            delta_r2 = np.nan
        plt.figure(figsize=(5, 4), dpi=140)
        plt.scatter(yhat, resid, alpha=0.5)
        plt.axhline(0, color="black", linewidth=1)

        # Optional annotation when thresholds crossed
        if abs(r_res_fit) > 0.2:
            plt.axhline(0, color="red", linestyle="--", linewidth=1, label="|resid-fitted| > 0.2")
        if delta_r2 > 0.03:
            plt.axhline(0, color="green", linestyle="--", linewidth=1, label="ΔR² > 0.03")

        plt.xlabel("Fitted (linear)"); plt.ylabel("Residuals")
        plt.title(f"Residuals vs Fitted (r(resid, fitted)={r_res_fit:.2f}, ΔR²_quad={0 if np.isnan(delta_r2) else delta_r2:.3f})")
        plt.legend()
        plt.tight_layout(); plt.show()
        
        ##qqplot
        plt.figure(figsize=(5, 4), dpi=140)
        probplot(resid, dist="norm", plot=plt)
        plt.title("QQ Plot of Residuals (Linear Model)")
        plt.tight_layout(); plt.show()

    else:
        r_res_fit = np.nan
        delta_r2 = np.nan

    # (3) Correlation heatmap (choose columns or default numeric)
    _simple_corr_heatmap(df, cols=heatmap_cols, title="Correlation heatmap (Pearson)")
    # (3b) Spearman correlation heatmap
    if heatmap_cols is None:
        heatmap_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    spearman_corr = df[heatmap_cols].corr(method="spearman")
    plt.figure(figsize=(6, 5), dpi=140)
    im = plt.imshow(spearman_corr.values, vmin=-1, vmax=1)
    plt.colorbar(im, label="spearman ρ")
    plt.xticks(range(len(heatmap_cols)), heatmap_cols, rotation=45, ha="right")
    plt.yticks(range(len(heatmap_cols)), heatmap_cols)
    plt.title("Correlation heatmap (Spearman)")
    for i in range(len(heatmap_cols)):
        for j in range(len(heatmap_cols)):
            v = spearman_corr.values[i, j]
            if not np.isnan(v):
                plt.text(j, i, f"{v:.2f}", ha="center", va="center")
    plt.tight_layout(); plt.show()

    # Heuristic recommendation
    recommend = "pearson"
    reasons = []
    if not (np.isnan(rho) or np.isnan(r)) and (abs(rho) - abs(r) >= 0.10):
        recommend = "spearman"; reasons.append("|ρ| noticeably > |r| (≥ 0.10)")
    if not np.isnan(r_res_fit) and abs(r_res_fit) > 0.20:
        recommend = "spearman"; reasons.append("|corr(resid, fitted)| > 0.20 (pattern)")
    if not np.isnan(delta_r2) and delta_r2 > 0.03:
        recommend = "spearman"; reasons.append("quadratic adds > 0.03 R² (nonlinearity)")
    warn = " (low n)" if n < min_n else ""
    print(f"Recommendation: use **{recommend.upper()}**{warn}.  "
          f"Stats: r={r:.3f} (p={p:.3g}), ρ={rho:.3f} (p={ps:.3g}).  "
          f"Reasons: {', '.join(reasons) if reasons else 'linear looks ok'}")
    return {
    "recommendation": recommend,
    "n": n,
    "pearson_r": r,
    "spearman_rho": rho,
    "pearson_p": p,
    "spearman_p": ps,
    "resid_corr": r_res_fit,
    "delta_r2_quad": delta_r2,
    "reasons": reasons
}


pre_correlation_analysis(
    hockey_df,
    "assists",
    "goalie_wins",
    heatmap_cols=["assists","goalie_wins","goals_against_average","save_percentage"]
)
