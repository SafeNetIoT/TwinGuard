import os
import pandas as pd
import numpy as np
import re
from collections import Counter, deque
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- CONFIG ---
ENCODED_SEQUENCE_DIR = "../data/sequences_encoded"
WINDOW_BLOCKS = 192           # rolling window size
FREQUENCY_THRESHOLD = 100     # frequency threshold for sensitive tokens

BLOCK_START = 192             # start block for visualization
BLOCK_END = 624               # end block for both processing and plotting

# --- Tokenization and keyword extraction ---
def tokenize(text):
    return re.findall(r"\b\w+\b", str(text).lower())

def extract_keywords(path, sensitive_words):
    tokens = re.split(r"[/._\-]", str(path).lower())
    matched_keywords = {token for token in tokens if token in sensitive_words}
    if not matched_keywords:
        return "<no-matching>"
    return ",".join(sorted(matched_keywords))

def stream_chronological_files(sequence_dir):
    files = []
    for day in sorted(os.listdir(sequence_dir)):
        day_path = os.path.join(sequence_dir, day)
        if not os.path.isdir(day_path):
            continue
        for fname in sorted(os.listdir(day_path)):
            if fname.endswith("_encoded.csv"):
                files.append(os.path.join(day_path, fname))
    return files

# --- Rolling Window Unknown Sequence Rate Calculation ---
all_files = stream_chronological_files(ENCODED_SEQUENCE_DIR)

# Cap the number of processed files to BLOCK_END
if BLOCK_END >= len(all_files):
    raise ValueError(f"BLOCK_END ({BLOCK_END}) exceeds total number of available blocks ({len(all_files)}).")
files_to_process = all_files[:BLOCK_END + 1]

word_counter = Counter()
sensitive_set = set()
rolling_deque = deque(maxlen=WINDOW_BLOCKS)
rows = []

for block_index, seq_file in enumerate(tqdm(files_to_process, desc="Processing blocks (rolling window)")):
    df = pd.read_csv(seq_file)
    
    for col in ["http_uri", "http_body"]:
        if col in df.columns:
            for text in df[col].dropna():
                for word in tokenize(text):
                    word_counter[word] += 1
                    if word_counter[word] >= FREQUENCY_THRESHOLD:
                        sensitive_set.add(word)

    sequences_this_block = set()
    for row in df.itertuples(index=False):
        try:
            method = getattr(row, "http_method")
            status = getattr(row, "http_status")
            uri = getattr(row, "http_uri")
        except AttributeError:
            continue
        sens_key = extract_keywords(uri, sensitive_set)
        seq = (method, status, sens_key)
        sequences_this_block.add(seq)

    rolling_deque.append(sequences_this_block)

    if block_index >= WINDOW_BLOCKS:
        seen_set = set().union(*list(rolling_deque)[:-1])
        unknown_seqs = [seq for seq in sequences_this_block if seq not in seen_set]
        unknown_rate = len(unknown_seqs) / (len(sequences_this_block) + 1e-8)
        rows.append({
            "block_index": block_index,
            "file": seq_file,
            "unique_sequences": len(sequences_this_block),
            "unknown_sequences": len(unknown_seqs),
            "unknown_rate": unknown_rate,
        })

# Save stats
df_stats = pd.DataFrame(rows)
df_stats.to_csv("../data/plots/unknown_rate_per_block_rollingwindow.csv", index=False)
print("✅ Rolling window unknown rates saved to ../data/plots/unknown_rate_per_block_rollingwindow.csv")

# --- Visualization with Formatting ---
df = pd.read_csv("../data/plots/unknown_rate_per_block_rollingwindow.csv")
df = df[(df["block_index"] >= BLOCK_START) & (df["block_index"] <= BLOCK_END)]

blocks = df["block_index"]
unknown_rates = df["unknown_rate"]

plt.figure(figsize=(14, 8))
plt.plot(blocks, unknown_rates, marker="o", linestyle="-", label="Unknown Rate", alpha=0.85, color="royalblue")

mean_ur = np.mean(unknown_rates)
std_ur = np.std(unknown_rates)
p95_ur = np.percentile(unknown_rates, 95)

plt.axhline(mean_ur, color="#f3a361", linestyle="--", linewidth=1.8, label=f"Mean ({mean_ur:.3f})")
plt.axhline(mean_ur + 3 * std_ur, color="#e66d50", linestyle=":", linewidth=1.8, label=f"Mean + 3 STD ({mean_ur+ 3 * std_ur:.3f})")
plt.axhline(p95_ur, color="#299d8f", linestyle="-.", linewidth=1.8, label=f"95th Percentile ({p95_ur:.3f})")

plt.xlabel("Block Index", fontsize=26)
plt.ylabel("Unknown Rate", fontsize=26)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.legend(
    loc="upper center", 
    bbox_to_anchor=(0.5, 1.24), 
    ncol=2, 
    frameon=False, fontsize=26)
plt.tight_layout()
plt.savefig("../data/plots/unknown_rate_stats.pdf", bbox_inches='tight')
plt.show()

print(f"Mean: {mean_ur:.4f}, Std: {std_ur:.4f}, 95th percentile: {p95_ur:.4f}")
