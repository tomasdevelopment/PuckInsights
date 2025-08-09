#Mcdavid Analytics
# Filter recent draft years
recent_players = hockey_clean[hockey_clean["year"] >= 2010].copy()

# Sort by career points in dataset
top_recent = recent_players.sort_values("points", ascending=False).head(10)

import seaborn as sns
import matplotlib.pyplot as plt

sns.barplot(data=top_recent, x="points", y="player")
plt.title("Top Career Points — Drafted Since 2010")
plt.xlabel("Points")
plt.ylabel("")
plt.show()


hockey_clean["years_played"] = hockey_clean["to_year"] - hockey_clean["year"] + 1
hockey_clean["points_per_season"] = hockey_clean["points"] / hockey_clean["years_played"]

top_rate = (
    hockey_clean[hockey_clean["points"] > 0]
    .sort_values("points_per_season", ascending=False)
    .head(10)
)

sns.barplot(data=top_rate, x="points_per_season", y="player")
plt.title("Points per Season — All-Time (in dataset)")
plt.xlabel("Points / Season")
plt.ylabel("")
plt.show()


