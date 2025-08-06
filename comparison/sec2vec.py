import os
import pandas as pd
import numpy as np
from collections import deque
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import utils

# ======= CONFIG =======
SEQUENCE_DIR = "../data/sequences"
SEQUENCE_BLOCKS_PER_DAY = 24
WINDOW_DAYS = 8
N_BLOCKS = WINDOW_DAYS * SEQUENCE_BLOCKS_PER_DAY

FIELDS_FOR_RAW_HTTP = [
    'protocol', 'http_method', 'http_status', 'http_uri'
]

def build_full_http_text(row):
    out = []
    for f in FIELDS_FOR_RAW_HTTP:
        if f in row and pd.notna(row[f]) and str(row[f]).strip():
            val = str(row[f])
            out.append(f"{f}:{val}")
    return '\n'.join(out)

def run_static_experiment_sec2vec(window_days=WINDOW_DAYS, blocks_per_day=SEQUENCE_BLOCKS_PER_DAY):
    metric_csv = f"../data/com/static_model_metrics_sec2vec_w{window_days}.csv"
    window_files = deque(maxlen=N_BLOCKS)
    file_stream = utils.stream_chronological_files(SEQUENCE_DIR)

    print(f"[Sec2vec] Filling initial window of {N_BLOCKS} blocks for training...")
    for i in tqdm(range(N_BLOCKS), desc="Initializing window"):
        try:
            f = next(file_stream)
        except StopIteration:
            raise RuntimeError(f"Not enough blocks: needed {N_BLOCKS}")
        window_files.append(f)

    # --- TRAIN STATIC MODEL ---
    print(f"[Sec2vec] Training on {N_BLOCKS} blocks...")
    train_dfs = []
    for f in window_files:
        try:
            df = pd.read_csv(f)
        except (pd.errors.ParserError, UnicodeDecodeError, OSError) as e:
            print(f"[Train] Skipping broken file: {f}, error: {e}")
            continue
        if "category" not in df.columns or df["category"].dropna().empty:
            continue
        # Robust label filtering
        df = df.dropna(subset=['category'])
        df = df[df['category'].astype(str).str.strip() != ""]
        df['category'] = df['category'].astype(str)
        if df.empty: continue
        df['http_fulltext'] = df.apply(build_full_http_text, axis=1)
        train_dfs.append(df)
    if not train_dfs:
        raise RuntimeError("No valid training data found!")
    df_train = pd.concat(train_dfs, ignore_index=True)
    tfidf = TfidfVectorizer(max_features=2000)  # You can adjust the size
    X_train = tfidf.fit_transform(df_train['http_fulltext']).toarray()
    y_train = df_train['category'].values
    clf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    # --- EVALUATE ON REMAINING BLOCKS ---
    metric_log = []
    block_index = N_BLOCKS + 1
    for seq_file in tqdm(file_stream, desc="[Sec2vec] Testing on session blocks"):
        try:
            df = pd.read_csv(seq_file, low_memory=False)
        except (pd.errors.ParserError, UnicodeDecodeError, OSError) as e:
            print(f"[Test] Skipping broken file: {seq_file}, error: {e}")
            continue
        if "category" not in df.columns or df["category"].dropna().empty:
            continue
        # Robust label filtering
        df = df.dropna(subset=['category'])
        df = df[df['category'].astype(str).str.strip() != ""]
        df['category'] = df['category'].astype(str)
        if df.empty: continue

        df['http_fulltext'] = df.apply(build_full_http_text, axis=1)
        X_block = tfidf.transform(df['http_fulltext']).toarray()
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
    print(f"[Sec2vec] Metrics saved: {metric_csv}")

if __name__ == "__main__":
    run_static_experiment_sec2vec()
