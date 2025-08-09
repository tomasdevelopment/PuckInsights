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
