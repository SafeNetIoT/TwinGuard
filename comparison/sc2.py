import os
import pandas as pd
import numpy as np
from collections import deque
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# --- Expert-selected 64 tokens from Table I in the paper (example, fill as needed) ---
TOKENS = [
    "<", ">", "alert", "exec", "password", "alter", "from", "path", "child", "'", '"',
    "and", "href", "script", "=", "(", ")", "bash", "history", "#include", "select", ";", "$",
    "/c", "insert", "shell", "--", "*", "cmd", "javascript:", "union", "*/", "cn=", "mail=",
    "upper", "-->", "&", "commit", "objectclass", "url=", "+", "count", "onmouseover",
    "User-Agent:", "%00", "-craw", "or", "where", "%0a", "document.cookie", "order", "winnt",
    "/*", "Accept:", "etc/passwd", "passwd"
    # ... (add any missing ones from Table I, up to 64)
]

def count_tokens(request_str):
    import re
    return [len(re.findall(re.escape(tok), str(request_str))) for tok in TOKENS]

def extract_sc2_features(df):
    # Combine http_uri + http_body (could add headers too)
    request_text = df["http_uri"].astype(str) + " " + df.get("http_body", "").astype(str)
    return np.vstack([count_tokens(r) for r in request_text])

# --- Config ---
SEQUENCE_DIR = "../data/sequences"  # or "../data/sequences_encoded"
SEQUENCE_BLOCKS_PER_DAY = 24
WINDOW_DAYS = 8
N_BLOCKS = WINDOW_DAYS * SEQUENCE_BLOCKS_PER_DAY

import utils  # For your stream_chronological_files function

def run_static_experiment_sc2(window_days=WINDOW_DAYS, blocks_per_day=SEQUENCE_BLOCKS_PER_DAY):
    metric_csv = f"../data/com/static_model_metrics_sc2_w{window_days}.csv"
    window_files = deque(maxlen=N_BLOCKS)
    file_stream = utils.stream_chronological_files(SEQUENCE_DIR)

    print(f"[sc2] Filling initial window of {N_BLOCKS} blocks for training...")
    for i in tqdm(range(N_BLOCKS), desc="Initializing window"):
        try:
            f = next(file_stream)
        except StopIteration:
            raise RuntimeError(f"Not enough blocks: needed {N_BLOCKS}")
        window_files.append(f)

    # --- TRAIN STATIC MODEL ---
    print(f"[sc2] Training on {N_BLOCKS} blocks...")
    train_dfs = []
    for f in window_files:
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"[Train] Skipping broken file: {f}, error: {e}")
            continue
        if "category" not in df.columns or df["category"].dropna().empty:
            continue
        df = df.dropna(subset=['category'])
        df = df[df['category'].astype(str).str.strip() != ""]
        df['category'] = df['category'].astype(str)
        if df.empty: continue
        train_dfs.append(df)
    if not train_dfs:
        raise RuntimeError("No valid training data found!")
    df_train = pd.concat(train_dfs, ignore_index=True)
    X_train = extract_sc2_features(df_train)
    y_train = df_train['category'].values
    clf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    # --- EVALUATE ON REMAINING BLOCKS ---
    metric_log = []
    block_index = N_BLOCKS + 1
    for seq_file in tqdm(file_stream, desc="[sc2] Testing on session blocks"):
        try:
            df = pd.read_csv(seq_file, low_memory=False)
        except Exception as e:
            print(f"[Test] Skipping broken file: {seq_file}, error: {e}")
            continue
        if "category" not in df.columns or df["category"].dropna().empty:
            continue
        df = df.dropna(subset=['category'])
        df = df[df['category'].astype(str).str.strip() != ""]
        df['category'] = df['category'].astype(str)
        if df.empty: continue

        X_block = extract_sc2_features(df)
        y_block = df['category'].values
        y_pred = clf.predict(X_block)
        acc = accuracy_score(y_block, y_pred)
        prec = precision_score(y_block, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_block, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_block, y_pred, average='macro', zero_division=0)
        metric_entry = {
            "block_index": block_index,
            "file": seq_file,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1
        }
        metric_log.append(metric_entry)
        block_index += 1

    metric_df = pd.DataFrame(metric_log)
    os.makedirs(os.path.dirname(metric_csv), exist_ok=True)
    metric_df.to_csv(metric_csv, index=False)
    print(f"[sc2] Metrics saved: {metric_csv}")

if __name__ == "__main__":
    run_static_experiment_sc2()
