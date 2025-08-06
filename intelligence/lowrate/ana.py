import os
import pandas as pd
import glob
import numpy as np

# --- Set your result dir here ---
RESULTS_DIR = "../../data/lowrate"

all_results = []

for fname in sorted(glob.glob(os.path.join(RESULTS_DIR, "metrics_*.csv"))):
    # Extract interval from file name
    basename = os.path.basename(fname)
    interval = int(basename.split("_")[1].split(".")[0])
    
    df = pd.read_csv(fname)
    
    summary_row = {
        "interval": interval,
        "n_blocks": len(df),
        "accuracy_mean": df["accuracy"].mean(),
        "accuracy_median": df["accuracy"].median(),
        "accuracy_std": df["accuracy"].std(),
        "n_samples": df["n_samples"].median(),
    }
    # If FP columns exist, summarize them
    for col in df.columns:
        if col.startswith("fp_"):
            summary_row[f"{col}_mean"] = df[col].mean()
            #summary_row[f"{col}_median"] = df[col].median()
    all_results.append(summary_row)

summary_df = pd.DataFrame(all_results)
summary_df = summary_df.sort_values("interval").reset_index(drop=True)

print("\n==== Aggregated Results Across Intervals ====\n")
print(summary_df)

# --- Save as CSV ---
summary_df.to_csv("metrics_summary_by_interval.csv", index=False)
print("\nSaved to metrics_summary_by_interval.csv")


'''
visulization
'''
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = "../../data/lowrate"
dfs = []
for fname in sorted(glob.glob(os.path.join(RESULTS_DIR, "metrics_*.csv"))):
    interval = int(os.path.basename(fname).split("_")[1].split(".")[0])
    df = pd.read_csv(fname)
    df["interval"] = interval
    dfs.append(df)
all_df = pd.concat(dfs, ignore_index=True)

intervals = sorted(all_df["interval"].unique())
n = len(intervals)
ncols = 4
nrows = (n + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(20, 4*nrows), sharex=True, sharey=True)
axes = axes.flatten()

bin_edges = np.arange(75, 100.5, 0.5)
global_max = 0
for interval in intervals:
    data = all_df[all_df["interval"] == interval]["accuracy"] * 100
    counts, _ = np.histogram(data, bins=bin_edges)
    global_max = max(global_max, counts.max())

for i, interval in enumerate(intervals):
    data = all_df[all_df["interval"] == interval]["accuracy"] * 100
    axes[i].hist(
        data,
        bins=bin_edges,
        color="royalblue",
        edgecolor="black",
        linewidth=0.6,
        alpha=0.85
    )
    axes[i].set_title(f"Interval: {interval}s", fontsize=34)
    axes[i].set_xlim(75, 100)
    axes[i].set_ylim(0, global_max * 1.05)
    axes[i].tick_params(axis='both', which='major', labelsize=28)



xticks = np.arange(75, 101, 5)
yticks = np.linspace(0, int(global_max * 1.05), 5).astype(int)
#yticks = np.arange(0, int(global_max * 1.05) + 1, 100)  # use 100 step for block count

for i, ax in enumerate(axes):
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    ax.tick_params(axis='x', labelsize=28)
    ax.tick_params(axis='y', labelsize=28)
    # Only show x tick labels on the bottom row
    #if i < ncols * (nrows - 1):
    #    ax.set_xticklabels([])
    # Only show y tick labels on the leftmost column
    #if i % ncols != 0:
    #    ax.set_yticklabels([])

# Hide unused subplots if nrows*ncols > n
for j in range(n, nrows*ncols):
    axes[j].axis('off')

fig.text(0.5, 0.04, 'Accuracy (%)', ha='center', va='center', fontsize=40)
fig.text(0.07, 0.5, 'Number of Blocks', ha='center', va='center', rotation='vertical', fontsize=40)

plt.tight_layout(rect=[0.08, 0.06, 1, 0.96])

plt.savefig("../../data/lowrate/acc_distribution.pdf", bbox_inches='tight')
plt.close()
