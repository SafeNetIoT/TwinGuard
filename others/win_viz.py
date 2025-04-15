import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.lines as mlines

# === CONFIG ===
win = 6
df = pd.read_csv(f"../data/dt/digital_twin_log_win{win}.csv")

# === Preprocess Dates ===
df["Test Date"] = pd.to_datetime(df["Test Date"]).dt.floor("D")
df["Train Window Start"] = pd.to_datetime(df["Train Window Start"]).dt.floor("D")
df["Train Window End"] = pd.to_datetime(df["Train Window End"]).dt.floor("D")
df = df.sort_values("Test Date")  # Ensure chronological x-axis

# === Compute Start of Testing Line ===
testing_start = df["Train Window End"].min() + pd.Timedelta(days=1)

# === Plot Setup ===
fig, ax1 = plt.subplots(figsize=(12, 6))

# === Plot RF and XGB Accuracy (Left Y-Axis) ===
ax1.plot(df["Test Date"], df["RF Test Accuracy"], marker='o', color='#A9082C', label="RF Accuracy (%)", linewidth=2)
ax1.plot(df["Test Date"], df["XGB Test Accuracy"], marker='s', color='#117733', label="XGB Accuracy (%)", linewidth=2)
ax1.set_ylabel("Accuracy (%)", fontsize=22)
ax1.set_ylim(30, 100)
ax1.tick_params(axis='y', labelsize=18)
ax1.tick_params(axis='x', labelsize=18)

# === Plot Unknown Rate (Right Y-Axis) ===
ax2 = ax1.twinx()
ax2.plot(df["Test Date"], df["Unknown Rate"], marker='^', linestyle='--', color='#4B61A8', label="Unknown Rate (%)", linewidth=2)
ax2.set_ylabel("Unknown Rate (%)", color='#4B61A8', fontsize=22)
ax2.set_ylim(0, 25)
ax2.tick_params(axis='y', labelcolor='#4B61A8', labelsize=18)

# === Mark Model Update Dates (Vertical Dashed Lines) ===
for idx, row in df.iterrows():
    if row["Model Updated"] == "Yes":
        ax1.axvline(x=row["Test Date"], color='#FDB76D', linestyle='--', linewidth=1.2)

# === Add Start Testing Line ===
ax1.axvline(x=testing_start, color='#7E57C2', linestyle='-.', linewidth=1.5)
ax1.text(testing_start, ax1.get_ylim()[1] - 15, 'Start Testing', color='#7E57C2',
         ha='center', fontsize=22, fontweight='bold')

# === Legend Setup ===
update_line = mlines.Line2D([], [], color='#FDB76D', linestyle='--', linewidth=1.5, label='Model Update')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

ax1.legend(handles=lines1 + lines2 + [update_line],
           labels=labels1 + labels2 + ['Model Update'],
           loc="lower left", fontsize=16)

# === Format X-Axis ===
ax1.set_xlim(left=df["Test Date"].min())
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
fig.autofmt_xdate(rotation=0)

# === Title and Save ===
plt.title(f"Accuracy & Unknown Rate Over Time (Window = {win})", fontsize=24)
plt.subplots_adjust(top=1) 
plt.tight_layout()
plt.savefig(f"../figs/win{win}.png", dpi=300)
plt.show()
