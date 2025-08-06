import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os

model_files = {
    "Betarte et al.": "static_model_metrics_sc2_w8.csv",
    "Estepa et al.": "static_model_metrics_2gram_w8.csv",
    "Eunaicy et al.": "static_model_metrics_rnn_w8.csv",
    "Gniewkowski et al.": "static_model_metrics_sec2vec_w8.csv",
    "LIVID": "stable_model_metrics_rf_w8.csv"
}
metric_keywords = ["accuracy", "precision", "recall", "f1_score"]

# ----- Numerical summary -----
summary = {}
daily_means = {}

def extract_date(filepath):
    match = re.search(r'(\d{4}-\d{2}-\d{2})', str(filepath))
    return match.group(1) if match else None


for model, fname in model_files.items():
    if not os.path.exists(fname):
        print(f"File {fname} not found, skipping.")
        continue
    df = pd.read_csv(fname)
    # Numerical summary
    stat = {}
    for key in metric_keywords:
        col = [c for c in df.columns if re.search(key, c, re.IGNORECASE)]
        if col:
            vals = pd.to_numeric(df[col[0]], errors='coerce').dropna()
            stat[key] = vals.mean()
        else:
            stat[key] = None
    summary[model] = stat
    # Daily means for line plot
    df["date"] = df["file"].apply(extract_date)
    col_acc = [c for c in df.columns if re.search("accuracy", c, re.IGNORECASE)]
    if col_acc:
        vals = pd.to_numeric(df[col_acc[0]], errors='coerce')
        df_metric = pd.DataFrame({"date": df["date"], model: vals})
        daily_mean = df_metric.groupby("date")[model].mean()
        daily_means[model] = daily_mean

plot_df = pd.DataFrame(summary).T.round(4)
print("\nNumerical summary (mean values):")
print(plot_df)

'''
# ----- Grouped bar plot (all 4 metrics per model) -----
plt.figure(figsize=(14, 10))
bar_width = 0.18
index = np.arange(len(plot_df))
colors = ['#274753', '#e7c66b', '#f3a361', '#e66d50']  # For metrics

for i, metric in enumerate(metric_keywords):
    plt.bar(index + i * bar_width, plot_df[metric], bar_width, label=metric.title(), color=colors[i])

plt.xlabel("Model", fontsize=26)
plt.ylabel("Mean Score", fontsize=26)
plt.title("Mean Metrics by Model", fontsize=24)
plt.xticks(index + 1.5*bar_width, plot_df.index, fontsize=20)
plt.yticks(fontsize=20)
plt.ylim(0, 1)
plt.legend(fontsize=20)
plt.tight_layout()
plt.savefig("mean_metrics_grouped_barplot.png")
plt.close()
'''
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Assume daily_means is already created as {model_name: pd.Series(date, accuracy)}
line_df = pd.DataFrame(daily_means)

custom_colors = {
    "Betarte et al.": "#4fc3f7",        # Blue
    "Estepa et al.": "#f3a361",         # Orange
    "Eunaicy et al.": "#299d8f",        # Green
    "Gniewkowski et al.": "#e66d50",    # Red-Orange
    "LIVID": "#9467bd",                # Purple
}
custom_styles = {
    "Betarte et al.": (0, (3, 1, 1, 1)),
    "Estepa et al.": "--",
    "Eunaicy et al.": "-.",
    "Gniewkowski et al.": ":",
    "LiveID": "-"  # solid line
}

# Ensure index is DatetimeIndex and sorted
line_df.index = pd.to_datetime(line_df.index)
line_df = line_df.sort_index()

fig, ax = plt.subplots(figsize=(14, 8))

for col in line_df.columns:
    ax.plot(
        line_df.index, line_df[col],
        label=col,
        color=custom_colors.get(col, None),
        linestyle=custom_styles.get(col, "-"),
        linewidth=3
    )

ax.set_xlabel("Date", fontsize=26)
ax.set_ylabel("Daily Mean Accuracy", fontsize=26)

ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 1.24),
    ncol=3,
    frameon=False,
    fontsize=26
)
plt.subplots_adjust(top=0.83)

# ------ Professional date axis (month-day, no rotation, auto spacing) ------
ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=8))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax.tick_params(axis='x', labelsize=24)
ax.tick_params(axis='y', labelsize=24)
plt.setp(ax.get_xticklabels(), rotation=0, ha='center')
# ax.set_xlim(line_df.index.min(), line_df.index.max())
plt.tight_layout()

plt.savefig("comline2.pdf", bbox_inches='tight')
plt.close()
