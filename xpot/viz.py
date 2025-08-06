import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
import re

'''
1. General visualization comparison
2. Retrain Analysis
3. Burst Aware
'''

# --- Load Data ---
stable = pd.read_csv("../data/xpot/stable_model_metrics_rf_w8.csv")
static = pd.read_csv("../data/xpot/static_model_metrics_rf_w8.csv")

# --- Extract date from file path ---
def extract_date(filepath):
    match = re.search(r'/(\d{4}-\d{2}-\d{2})/', str(filepath))
    return match.group(1) if match else None

stable["date"] = stable["file"].apply(extract_date)
static["date"] = static["file"].apply(extract_date)

# --- Group by date, take daily mean ---
stable_daily = stable.groupby("date").mean(numeric_only=True).reset_index()
static_daily = static.groupby("date").mean(numeric_only=True).reset_index()

# --- Convert date columns to datetime for plotting ---
stable_daily["date"] = pd.to_datetime(stable_daily["date"])
static_daily["date"] = pd.to_datetime(static_daily["date"])

# --- Prepare retrain event daily marking ---
stable["retrain_event_bool"] = stable["retrain_event"].astype(bool)
retrain_days = stable.groupby("date")["retrain_event_bool"].any().reset_index()
retrain_days = retrain_days[retrain_days["retrain_event_bool"]]["date"].tolist()

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

# --- Plot ---
fig, ax1 = plt.subplots(figsize=(14, 8))  # consistent size

# Accuracy lines
ln1 = ax1.plot(stable_daily["date"], stable_daily["accuracy"], label="Adaptive Accuracy", color="royalblue", linewidth=3)
ln2 = ax1.plot(static_daily["date"], static_daily["accuracy"], label="Static Accuracy", color="tomato", linestyle="--", alpha=0.6, linewidth=3)

ax1.set_ylabel("Accuracy", fontsize=26)
ax1.set_xlabel("Date", fontsize=26)
ax1.tick_params(axis='both', labelsize=24)

# Sparse x-ticks
every_n = 14
xticks = stable_daily["date"].iloc[::every_n]
ax1.set_xticks(xticks)
ax1.set_xticklabels([d.strftime("%m-%d") for d in xticks], ha='center')

# Shaded region
highlight_start = pd.to_datetime("2025-05-01")
highlight_end = pd.to_datetime("2025-05-30")
ax1.axvspan(highlight_start, highlight_end, color='gray', alpha=0.12, label="May Window")

# Drift axis (right)
ax2 = ax1.twinx()
ln3 = ax2.plot(stable_daily["date"], stable_daily["drift"], label="Data Drift", color="#299d8f", linewidth=2, linestyle=":")
ax2.set_ylabel("Data Drift", color="#299d8f", fontsize=26, labelpad=30, rotation=0)
ax2.yaxis.set_label_coords(1.04, -0.07)
ax2.tick_params(axis='y', labelcolor="#299d8f", labelsize=24)

# Unknown rate (far right)
ax3 = ax1.twinx()
ax3.spines["right"].set_position(("axes", 1.07))
ln4 = ax3.plot(stable_daily["date"], stable_daily["unknown_rate"], label="Feature Drift", color="#f3a361", linewidth=2, linestyle="-.")
ax3.set_ylabel("Feature Drift", color="#f3a361", fontsize=26, labelpad=30, rotation=0)
ax3.yaxis.set_label_coords(1.04, -0.14)
ax3.tick_params(axis='y', labelcolor="#f3a361", labelsize=24)
ax3.set_ylim(0, 1)

# --- Legend (top, one line, no border) ---
lns = ln1 + ln2 + ln3 + ln4
labels = [l.get_label() for l in lns]
ax1.legend(
    lns,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.24),
    ncol=2,
    frameon=False,
    fontsize=26
)

# Final layout
plt.tight_layout(rect=[0, 0, 0.95, 1])
plt.subplots_adjust(right=0.88)
plt.savefig("../data/xpot/model_comparison_withxpot.pdf", bbox_inches='tight')



# ==============================================================
# --- Retrain analysis to CSV ---
n_retrains = stable["retrain_event"].sum()
pd.DataFrame({"total_retrains": [n_retrains]}).to_csv("../data/xpot/total_retrain_events.csv", index=False)

retrain_per_day = stable.groupby("date")["retrain_event"].sum()
retrain_per_day.to_csv("../data/xpot/retrain_events_per_day.csv", header=["retrain_events"])

if "retrain_reason" in stable.columns:
    reason_counts = stable["retrain_reason"].value_counts()
    reason_counts.to_csv("../data/xpot/retrain_reasons_counts.csv", header=["count"])

    reason_by_day = stable.groupby(["date", "retrain_reason"])["retrain_event"].sum().unstack(fill_value=0)
    reason_by_day.to_csv("../data/xpot/retrain_reason_by_day.csv")





# ==================================================
# ---Burst Aware
# Align on block_index for fair comparison
df = pd.merge(stable, static, on="block_index", suffixes=("_stable", "_static"))

# --- Mark retrain (burst) blocks in the stable model ---
df["is_retrain"] = df["retrain_event"].astype(bool) if "retrain_event" in df.columns else df["retrain_event"].astype(bool)

# === 1. OVERALL MEAN PERFORMANCE ===
overall = {
    "Static": {
        "Accuracy": df["accuracy_static"].mean(),
        "Precision": df["precision_static"].mean(),
        "Recall": df["recall_static"].mean(),
        "F1": df["f1_score_static"].mean(),
    },
    "Stable (Adaptive)": {
        "Accuracy": df["accuracy_stable"].mean(),
        "Precision": df["precision_stable"].mean(),
        "Recall": df["recall_stable"].mean(),
        "F1": df["f1_score_stable"].mean(),
    }
}

# === 2. RETRAIN-ONLY (BURST) MEAN PERFORMANCE ===
retrain = df[df["is_retrain"]]
retrain_only = {
    "Static": {
        "Accuracy": retrain["accuracy_static"].mean(),
        "Precision": retrain["precision_static"].mean(),
        "Recall": retrain["recall_static"].mean(),
        "F1": retrain["f1_score_static"].mean(),
    },
    "Stable (Adaptive)": {
        "Accuracy": retrain["accuracy_stable"].mean(),
        "Precision": retrain["precision_stable"].mean(),
        "Recall": retrain["recall_stable"].mean(),
        "F1": retrain["f1_score_stable"].mean(),
    }
}

# === 3. "Rescue" Rate (blocks where Stable outperforms Static during retrain) ===
rescued_blocks = retrain[retrain["accuracy_stable"] > retrain["accuracy_static"]]
n_retrains = len(retrain)
n_rescued = len(rescued_blocks)
if n_retrains > 0:
    rescue_rate = 100 * n_rescued / n_retrains
    rescue_str = f"Stable rescued {n_rescued}/{n_retrains} retrain blocks ({rescue_rate:.1f}%)"
else:
    rescue_str = "No retrain blocks for rescue rate calculation."

# === 4. Print results ===
print("\n=== OVERALL ===")
print(pd.DataFrame(overall))

print("\n=== RETRAIN-ONLY (BURST) ===")
print(pd.DataFrame(retrain_only))

print(f"\n{rescue_str}")

# === 5. Save as CSVs for easy reporting ===
pd.DataFrame(overall).T.to_csv("../data/xpot/highlight_overall_static_vs_stable.csv")
pd.DataFrame(retrain_only).T.to_csv("../data/xpot/highlight_retrain_static_vs_stable.csv")
with open("../data/xpot/highlight_rescue_rate.txt", "w") as f:
    f.write(rescue_str)
