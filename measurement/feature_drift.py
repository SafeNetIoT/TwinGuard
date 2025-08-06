import os
import pandas as pd
import re
from collections import Counter
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = "../data/daily"
FREQUENCY_THRESHOLD = 20

def tokenize(text):
    return re.findall(r"\b\w+\b", str(text).lower())

# Data containers
word_counter = Counter()
sensitive_set = set()
daily_new_sensitive = []
daily_overlapped_sensitive = []
daily_inactive_sensitive = []

seen_sequences = set()
daily_new_sequences = []
daily_overlapped_sequences = []
daily_inactive_sequences = []

daily_labels = []

files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.csv')])

for fname in tqdm(files, desc="Processing daily CSVs"):
    path = os.path.join(DATA_DIR, fname)
    df = pd.read_csv(path)

    # --- Sensitive word tracking (for this day) ---
    new_this_day = set()
    sensitive_today = set()
    for col in ["http_uri", "http_body"]:
        if col not in df.columns:
            continue
        for text in df[col].dropna():
            for word in tokenize(text):
                word_counter[word] += 1
                if word_counter[word] >= FREQUENCY_THRESHOLD:
                    sensitive_today.add(word)
                if word_counter[word] == FREQUENCY_THRESHOLD and word not in sensitive_set:
                    new_this_day.add(word)
    overlapped_this_day = (sensitive_today - new_this_day) & sensitive_set
    inactive_this_day = sensitive_set - sensitive_today
    sensitive_set.update(new_this_day)
    daily_new_sensitive.append(len(new_this_day))
    daily_overlapped_sensitive.append(len(overlapped_this_day))
    daily_inactive_sensitive.append(len(inactive_this_day))

    # --- Sequence tracking (based on current sensitive_set) ---
    sequences_today = set()
    def extract_keywords(path, sensitive_words):
        tokens = re.split(r"[/._\-]", str(path).lower())
        matched_keywords = {token for token in tokens if token in sensitive_words}
        if not matched_keywords:
            return "<no-matching>"
        return ",".join(sorted(matched_keywords))

    for row in df.itertuples(index=False):
        try:
            method = getattr(row, "http_method")
            status = getattr(row, "http_status")
            uri = getattr(row, "http_uri")
        except AttributeError:
            continue
        sens_key = extract_keywords(uri, sensitive_set)
        seq = (method, status, sens_key)
        sequences_today.add(seq)
    new_seq_today = sequences_today - seen_sequences
    overlapped_seq_today = sequences_today & seen_sequences
    inactive_seq_today = seen_sequences - sequences_today
    seen_sequences.update(new_seq_today)
    daily_new_sequences.append(len(new_seq_today))
    daily_overlapped_sequences.append(len(overlapped_seq_today))
    daily_inactive_sequences.append(len(inactive_seq_today))

    day_label = fname.replace("http_day_", "").replace(".csv", "")
    daily_labels.append(day_label)

# --- Burst day helper
def burst_days(new_counts):
    mean = np.mean(new_counts)
    std = np.std(new_counts)
    threshold = mean + 2 * std
    return np.where(np.array(new_counts) > threshold)[0]

burst_idx_sens = burst_days(daily_new_sensitive)
burst_idx_seq = burst_days(daily_new_sequences)

# Print burst days to console (in order)
print("\n=== Sensitive Word Burst Days ===")
for idx in burst_idx_sens:
    print(f"{daily_labels[idx]} ({daily_new_sensitive[idx]} new words)")

print("\n=== Unique Sequence Burst Days ===")
for idx in burst_idx_seq:
    print(f"{daily_labels[idx]} ({daily_new_sequences[idx]} new sequences)")

# --- Plotting ---
fig, axs = plt.subplots(1, 2, figsize=(28, 8), sharex=True)

# Sensitive words subplot
axs[0].bar(daily_labels, daily_inactive_sensitive, color='lightgray', label="Inactive Sensitive Words")
axs[0].bar(daily_labels, daily_overlapped_sensitive, bottom=daily_inactive_sensitive, color='royalblue', label="Overlapped Sensitive Words")
axs[0].bar(
    daily_labels,
    daily_new_sensitive,
    bottom=np.array(daily_inactive_sensitive) + np.array(daily_overlapped_sensitive),
    color='tomato',
    label="New Sensitive Words"
)
axs[0].set_xlabel("Day", fontsize=24)
axs[0].set_ylabel("Cumulative Sensitive Words", fontsize=24)
#axs[0].set_title("Sensitive Words Drift", fontsize=28)
axs[0].legend(fontsize=20)
axs[0].tick_params(axis='x', rotation=45, labelsize=22)
axs[0].tick_params(axis='y', labelsize=22)

# Unique sequences subplot
axs[1].bar(daily_labels, daily_inactive_sequences, color='lightgray', label="Inactive Sequences")
axs[1].bar(daily_labels, daily_overlapped_sequences, bottom=daily_inactive_sequences, color='royalblue', label="Overlapped Sequences")
axs[1].bar(
    daily_labels,
    daily_new_sequences,
    bottom=np.array(daily_inactive_sequences) + np.array(daily_overlapped_sequences),
    color='tomato',
    label="New Sequences"
)
axs[1].set_xlabel("Day", fontsize=24)
axs[1].set_ylabel("Cumulative Unique Sequences", fontsize=24)
#axs[1].set_title("Unique Attack Sequence Drift", fontsize=28)
axs[1].legend(fontsize=20)
axs[1].tick_params(axis='x', rotation=45, labelsize=22)
axs[1].tick_params(axis='y', labelsize=22)

# Annotate bursts with only arrows (no text)
def annotate_arrows(ax, burst_indices, cumulative_height, color):
    y_offsets = [10 * i for i in range(len(burst_indices))]
    for idx, offset in zip(burst_indices, y_offsets):
        ax.annotate(
            '⬆',
            (idx, cumulative_height[idx] + offset),
            textcoords="offset points", xytext=(0, 0), ha='center',
            fontsize=18, color=color
        )


sens_cumulative = np.array(daily_inactive_sensitive) + np.array(daily_overlapped_sensitive) + np.array(daily_new_sensitive)
seq_cumulative = np.array(daily_inactive_sequences) + np.array(daily_overlapped_sequences) + np.array(daily_new_sequences)
annotate_arrows(axs[0], burst_idx_sens, sens_cumulative, color='#299df8')
annotate_arrows(axs[1], burst_idx_seq, seq_cumulative, color='#299df8')

# X-axis: show only every Nth tick label
N = 14
xtick_positions = np.arange(0, len(daily_labels), N)
axs[0].set_xticks(xtick_positions)
axs[0].set_xticklabels([daily_labels[i] for i in xtick_positions], rotation=45, ha='right', fontsize=22)
axs[1].set_xticks(xtick_positions)
axs[1].set_xticklabels([daily_labels[i] for i in xtick_positions], rotation=45, ha='right', fontsize=22)

plt.tight_layout()
plt.savefig("../data/plots/feature_drift.pdf", bbox_inches="tight")
plt.close()
print("✅ Annotated burst plot (arrows only) saved to ../data/plots/feature_and_sequence_drift_burst_arrows.pdf")
