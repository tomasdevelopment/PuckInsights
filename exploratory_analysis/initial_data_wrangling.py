import pandas as pd

# ------------------------------------------------------------------ #
# 0.  Load raw data                                                  #
# ------------------------------------------------------------------ #
FILE_PATH = "/content/drive/My Drive/sportsanalytics/nhldraft.csv"
hockey_df = pd.read_csv(FILE_PATH)

# ------------------------------------------------------------------ #
# 1.  Clean function (must exist BEFORE using it)                    #
# ------------------------------------------------------------------ #
def clean_hockey_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform basic cleaning on hockey draft DataFrame.
    Replace this logic with your custom cleaning.
    """
    df = df.copy()
    # Example cleaning steps
    df.columns = df.columns.str.strip()  # Strip whitespace from column names
    df.dropna(how='all', inplace=True)   # Drop rows where all values are NaN
    # Add your real cleaning steps here
    return df

# ------------------------------------------------------------------ #
# 2.  Clean, but DO NOT overwrite the function!                      #
# ------------------------------------------------------------------ #
hockey_clean = clean_hockey_df(hockey_df)   # ← keep object & function distinct

# ------------------------------------------------------------------ #
# 3.  One-liner footprint helper                                     #
# ------------------------------------------------------------------ #
def footprint(df: pd.DataFrame) -> dict:
    """Return shape & memory stats for any DataFrame."""
    mem_bytes = df.memory_usage(deep=True).sum()
    return {
        "rows"          : len(df),
        "cols"          : df.shape[1],
        "memory_MB"     : round(mem_bytes / 1024 ** 2, 3),
        "bytes_per_row" : round(mem_bytes / len(df), 1) if len(df) else 0,
    }

print("After cleaning:")
for k, v in footprint(hockey_clean).items():
    print(f"  {k:>13}: {v}")

# ------------------------------------------------------------------ #
# 4.  Optional peek                                                  #
# ------------------------------------------------------------------ #
print(hockey_clean.head())
