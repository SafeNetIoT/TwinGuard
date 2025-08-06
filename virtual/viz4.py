import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import re

# --- Load Data ---
stablerf = pd.read_csv("../data/vir/stable_model_metrics_rf_w8.csv")
staticrf = pd.read_csv("../data/vir/static_model_metrics_rf_w8.csv")
stablemlp = pd.read_csv("../data/vir/stable_model_metrics_mlp_w8.csv")
staticmlp = pd.read_csv("../data/vir/static_model_metrics_mlp_w8.csv")

def extract_date(filepath):
    match = re.search(r'/(\d{4}-\d{2}-\d{2})/', str(filepath))
    return match.group(1) if match else None

for df in [stablerf, staticrf, stablemlp, staticmlp]:
    df["date"] = df["file"].apply(extract_date)

# --- Group by date, take daily mean ---
stablerf_daily = stablerf.groupby("date").mean(numeric_only=True).reset_index()
staticrf_daily = staticrf.groupby("date").mean(numeric_only=True).reset_index()
stablemlp_daily = stablemlp.groupby("date").mean(numeric_only=True).reset_index()
staticmlp_daily = staticmlp.groupby("date").mean(numeric_only=True).reset_index()

# --- Prepare retrain event daily marking (using RF stable as example) ---
stablerf["retrain_event_bool"] = stablerf["retrain_event"].astype(bool)
retrain_days = stablerf.groupby("date")["retrain_event_bool"].any().reset_index()
retrain_days = retrain_days[retrain_days["retrain_event_bool"]]["date"].tolist()

# --- Plot: Accuracy Comparison ---
fig, ax1 = plt.subplots(figsize=(20, 8))

ln1 = ax1.plot(stablerf_daily["date"], stablerf_daily["accuracy"], label="Stable RF", color="b", linewidth=2)
ln2 = ax1.plot(staticrf_daily["date"], staticrf_daily["accuracy"], label="Static RF", color="gray", linestyle="--", alpha=0.6)
ln3 = ax1.plot(stablemlp_daily["date"], stablemlp_daily["accuracy"], label="Stable MLP", color="g", linewidth=2)
ln4 = ax1.plot(staticmlp_daily["date"], staticmlp_daily["accuracy"], label="Static MLP", color="orange", linestyle="--", alpha=0.6)

ax1.set_ylabel("Accuracy", fontsize=22)
ax1.set_xlabel("Date", fontsize=22)
ax1.tick_params(axis='both', labelsize=18)

# --- X-axis ticks: Show fewer ticks for clarity ---
every_n = 7
xticks = stablerf_daily["date"].iloc[::every_n]
ax1.set_xticks(xticks)
ax1.set_xticklabels(xticks, rotation=45, ha='right')

# --- Second Y-axis: Drift (from RF only for simplicity) ---
ax2 = ax1.twinx()
ln5 = ax2.plot(stablerf_daily["date"], stablerf_daily["drift"], label="Data Drift", color="#FF6347", linewidth=1)
ax2.set_ylabel("Data Drift", color="#FF6347", fontsize=22)
ax2.tick_params(axis='y', labelcolor="#FF6347", labelsize=18)

# --- Third Y-axis: Unknown Rate (from RF only for simplicity) ---
ax3 = ax1.twinx()
ax3.spines["right"].set_position(("axes", 1.07))
ln6 = ax3.plot(stablerf_daily["date"], stablerf_daily["unknown_rate"], label="Unknown Rate", color="#FFD700", linewidth=1.5)
ax3.set_ylabel("Unknown Rate", color="#FFD700", fontsize=22)
ax3.tick_params(axis='y', labelcolor="#FFD700", labelsize=18)
ax3.set_ylim(0, 1)

# --- Add retrain event days as vertical lines (optional) ---
# for d in retrain_days:
#     ax1.axvline(d, color="red", linestyle="--", alpha=0.6, ymax=0.93, linewidth=1.5)

# --- Legend setup ---
lns = ln1 + ln2 + ln3 + ln4 + ln5 + ln6
labels = [l.get_label() for l in lns]
#custom_lines = [
#    Line2D([0], [0], color="red", linestyle="--", linewidth=2, label="Retrain (any trigger)"),
#]
#lns += custom_lines
#labels += [l.get_label() for l in custom_lines]
ax1.legend(lns, labels, loc="upper left", fontsize=18)

#plt.title("Daily Aggregated Accuracy: Static vs Stable (RF/MLP)", fontsize=28)
plt.tight_layout(rect=[0, 0, 0.98, 1])
plt.subplots_adjust(right=0.85)
plt.savefig("../data/vir/model_comparison_daily_rf_mlp.png", dpi=300)

# ==============================================================
# --- Retrain analysis to CSV (for RF stable) ---
n_retrains = stablerf["retrain_event"].sum()
pd.DataFrame({"total_retrains": [n_retrains]}).to_csv("../data/vir/total_retrain_events.csv", index=False)
retrain_per_day = stablerf.groupby("date")["retrain_event"].sum()
retrain_per_day.to_csv("../data/vir/retrain_events_per_day.csv", header=["retrain_events"])
if "retrain_reason" in stablerf.columns:
    reason_counts = stablerf["retrain_reason"].value_counts()
    reason_counts.to_csv("../data/vir/retrain_reasons_counts.csv", header=["count"])
    reason_by_day = stablerf.groupby(["date", "retrain_reason"])["retrain_event"].sum().unstack(fill_value=0)
    reason_by_day.to_csv("../data/vir/retrain_reason_by_day.csv")

# ==================================================
# --- Burst Aware: align on block_index for fair comparison ---
def burst_analysis(stable, static, label):
    df = pd.merge(stable, static, on="block_index", suffixes=("_stable", "_static"))
    df["is_retrain"] = df["retrain_event_stable"].astype(bool) if "retrain_event_stable" in df.columns else df["retrain_event"].astype(bool)
    overall = {
        f"{label} Static": {
            "Accuracy": df["accuracy_static"].mean(),
            "Precision": df["precision_static"].mean(),
            "Recall": df["recall_static"].mean(),
            "F1": df["f1_score_static"].mean(),
        },
        f"{label} Stable (Adaptive)": {
            "Accuracy": df["accuracy_stable"].mean(),
            "Precision": df["precision_stable"].mean(),
            "Recall": df["recall_stable"].mean(),
            "F1": df["f1_score_stable"].mean(),
        }
    }
    retrain = df[df["is_retrain"]]
    retrain_only = {
        f"{label} Static": {
            "Accuracy": retrain["accuracy_static"].mean(),
            "Precision": retrain["precision_static"].mean(),
            "Recall": retrain["recall_static"].mean(),
            "F1": retrain["f1_score_static"].mean(),
        },
        f"{label} Stable (Adaptive)": {
            "Accuracy": retrain["accuracy_stable"].mean(),
            "Precision": retrain["precision_stable"].mean(),
            "Recall": retrain["recall_stable"].mean(),
            "F1": retrain["f1_score_stable"].mean(),
        }
    }
    rescued_blocks = retrain[retrain["accuracy_stable"] > retrain["accuracy_static"]]
    n_retrains = len(retrain)
    n_rescued = len(rescued_blocks)
    if n_retrains > 0:
        rescue_rate = 100 * n_rescued / n_retrains
        rescue_str = f"{label} Stable rescued {n_rescued}/{n_retrains} retrain blocks ({rescue_rate:.1f}%)"
    else:
        rescue_str = f"No retrain blocks for rescue rate calculation in {label}."
    return overall, retrain_only, rescue_str

overall_rf, retrain_only_rf, rescue_rf = burst_analysis(stablerf, staticrf, "RF")
overall_mlp, retrain_only_mlp, rescue_mlp = burst_analysis(stablemlp, staticmlp, "MLP")

print("\n=== OVERALL ===")
print(pd.DataFrame({**overall_rf, **overall_mlp}))

print("\n=== RETRAIN-ONLY (BURST) ===")
print(pd.DataFrame({**retrain_only_rf, **retrain_only_mlp}))

print(f"\n{rescue_rf}")
print(f"{rescue_mlp}")

# Save to CSVs for easy reporting
pd.DataFrame({**overall_rf, **overall_mlp}).T.to_csv("../data/vir/highlight_overall_static_vs_stable_rf_mlp.csv")
pd.DataFrame({**retrain_only_rf, **retrain_only_mlp}).T.to_csv("../data/vir/highlight_retrain_static_vs_stable_rf_mlp.csv")
with open("../data/vir/highlight_rescue_rate_rf_mlp.txt", "w") as f:
    f.write(rescue_rf + "\n" + rescue_mlp)
