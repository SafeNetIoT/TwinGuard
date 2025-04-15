import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import pearsonr
from matplotlib.lines import Line2D

# Setup
base_path = "../data/dt"
window_sizes = [2, 3, 4, 5, 6]
file_template = "digital_twin_log_win{}.csv"

# Load data
all_data = []
for win in window_sizes:
    file_path = os.path.join(base_path, file_template.format(win))
    df = pd.read_csv(file_path)
    for _, row in df.iterrows():
        for model in ["RF", "XGB"]:
            all_data.append({
                "Model": model,
                "Window Size": f"w{win}",
                "Unknown Rate (%)": row["Unknown Rate"],
                "Accuracy Drop (%)": -row[f"{model} Accuracy Drop"]
            })

df_all = pd.DataFrame(all_data)

# Corrected column name usage
r, p = pearsonr(df_all["Unknown Rate (%)"], df_all["Accuracy Drop (%)"])
print(f"Pearson correlation: r = {r:.4f}, p-value = {p:.4e}")

plot_df = df_all.copy()

# Plot setup
plt.figure(figsize=(12, 6.4))
#sns.set(style="whitegrid")

# Color for each window
palette = sns.color_palette("Spectral", len(window_sizes))
window_colors = {f"w{w}": palette[i] for i, w in enumerate(window_sizes)}
model_shapes = {"RF": "o", "XGB": "X"}

# Draw scatter points
for _, row in plot_df.iterrows():
    plt.scatter(
        row["Unknown Rate (%)"],
        row["Accuracy Drop (%)"],
        marker=model_shapes[row["Model"]],
        color=window_colors[row["Window Size"]],
        edgecolor="black",
        s=120,
        alpha=0.9
    )

# Add regression line with label
sns.regplot(
    data=plot_df,
    x="Unknown Rate (%)",
    y="Accuracy Drop (%)",
    scatter=False,
    color="gray",
    line_kws={"label": f"Pearson r = {r:.2f}"}
)

# Combined legend
legend_elements = []

# Window size color blocks
for w, color in window_colors.items():
    legend_elements.append(Line2D([0], [0], marker='s', linestyle='None',
                                  markerfacecolor=color, markeredgecolor='black',
                                  label=w, markersize=10))

# Model shape legend
legend_elements += [
    Line2D([0], [0], marker='o', linestyle='None', color='black',
           label='RF', markersize=10),
    Line2D([0], [0], marker='X', linestyle='None', color='black',
           label='XGB', markersize=10)
]

plt.legend(
    handles=legend_elements,
    title="Window Size / Model",
    loc='lower right',
    fontsize=18,
    title_fontsize=18,
    frameon=True
)

# Final touches
plt.title("Unknown Rate vs. Accuracy Drop", fontsize=24)
plt.xlabel("Unknown Rate (%)", fontsize=22)
plt.ylabel("Accuracy Drop (%)", fontsize=22)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.tight_layout()
plt.savefig("../figs/scatter_pearson.png", dpi=300)
