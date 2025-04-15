import pandas as pd
import matplotlib.pyplot as plt
import os

# Define paths and window sizes
base_path = "../data/dt"
file_template = "digital_twin_log_win{}.csv"
window_sizes = [2, 3, 4, 5, 6]

# Collect total new sequences per window
new_sequences_by_window = {}

for win in window_sizes:
    file_path = os.path.join(base_path, file_template.format(win))
    df = pd.read_csv(file_path)
    total_new_sequences = df["New Sequences"].sum()
    new_sequences_by_window[win] = total_new_sequences

# Plotting
plt.figure(figsize=(12, 6))
bars = plt.bar(new_sequences_by_window.keys(), new_sequences_by_window.values(), color='#4B61A8', alpha=0.7)
plt.xlabel("Sliding Window Size (Days)", fontsize=22)
plt.ylabel("Total New Sequences", fontsize=22)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.title("Total New Sequences vs. Sliding Window Size", fontsize=24)
plt.xticks(window_sizes)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add numbers on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, f"{int(height)}", ha='center', va='bottom', fontsize=12)

plt.tight_layout()
plt.savefig("../figs/new_sequences_by_window.png", dpi=300)
