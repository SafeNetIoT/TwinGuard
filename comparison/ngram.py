import os
import pandas as pd
import numpy as np
from collections import deque
from tqdm import tqdm
import utils

# ======= CONFIG =======
SEQUENCE_DIR = "../data/sequences"
SEQUENCE_BLOCKS_PER_DAY = 24
WINDOW_DAYS = 8
N_BLOCKS = WINDOW_DAYS * SEQUENCE_BLOCKS_PER_DAY
NGRAM_N = 3   # Set to 1, 2, or 3 as needed

def run_static_experiment_ngram(ngram_n=NGRAM_N, window_days=WINDOW_DAYS, blocks_per_day=SEQUENCE_BLOCKS_PER_DAY):
    metric_csv = f"../data/com/static_model_metrics_{ngram_n}gram_w{window_days}.csv"
    window_files = deque(maxlen=N_BLOCKS)
    file_stream = utils.stream_chronological_files(SEQUENCE_DIR)

    print(f"[Static] ({ngram_n}-gram) Filling initial window of {N_BLOCKS} blocks for training...")
    for i in tqdm(range(N_BLOCKS), desc="Initializing window"):
        try:
            f = next(file_stream)
        except StopIteration:
            raise RuntimeError(f"Not enough blocks: needed {N_BLOCKS}")
        window_files.append(f)

    # --- TRAIN STATIC MODEL ---
    print(f"[Static] ({ngram_n}-gram) Training on {N_BLOCKS} blocks...")
    train_dfs = []
    for f in window_files:
        try:
            df = pd.read_csv(f)
        except (pd.errors.ParserError, UnicodeDecodeError, OSError) as e:
            print(f"[Train] Skipping broken file: {f}, error: {e}")
            continue
        if "category" not in df.columns or df["category"].dropna().empty:
            continue
        train_dfs.append(df)
    if not train_dfs:
        raise RuntimeError("No valid training data found!")
    df_train = pd.concat(train_dfs, ignore_index=True)
    normal_label = "normal"
    df_train_norm = df_train[df_train['category'] == normal_label] if (df_train['category'] == normal_label).any() else df_train
    dictionary, p_oov = utils.train_ngram(df_train_norm, n=ngram_n)

    # --- Threshold tuning on the first validation block (block after train window) ---
    try:
        val_file = next(file_stream)
        try:
            df_val = pd.read_csv(val_file)
        except (pd.errors.ParserError, UnicodeDecodeError, OSError) as e:
            print(f"[Validation] Skipping broken file: {val_file}, error: {e}")
            raise StopIteration
        val_scores = df_val['http_uri'].apply(lambda uri: utils.score_ngram(dictionary, p_oov, uri, n=ngram_n)).values
        val_true = (df_val['category'].astype(str) != normal_label).values
        best_eta, best_thr = -1, None
        for thr in pd.Series(val_scores).quantile(np.linspace(0.8, 0.99, 20)):
            pred = val_scores > thr
            TP = ((pred == 1) & (val_true == 1)).sum()
            FP = ((pred == 1) & (val_true == 0)).sum()
            TN = ((pred == 0) & (val_true == 0)).sum()
            FN = ((pred == 0) & (val_true == 1)).sum()
            DR = TP / (TP + FN) if (TP + FN) > 0 else 0
            FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
            eta = np.sqrt(DR * (1 - FPR))
            if eta > best_eta:
                best_eta, best_thr = eta, thr
        print(f"[Static] ({ngram_n}-gram) Tuned threshold: {best_thr:.4f} (eta={best_eta:.4f})")
    except StopIteration:
        print("No validation block for threshold tuning. Using default threshold 2.0")
        best_thr = 2.0

    # --- EVALUATE ON REMAINING BLOCKS ---
    metric_log = []
    block_index = N_BLOCKS + 1  # first test block after val
    for seq_file in tqdm(file_stream, desc=f"Testing ({ngram_n}-gram) on session blocks"):
        try:
            df = pd.read_csv(seq_file, low_memory=False)
        except (pd.errors.ParserError, UnicodeDecodeError, OSError) as e:
            print(f"[Test] Skipping broken file: {seq_file}, error: {e}")
            continue
        metrics = utils.evaluate_ngram_block(dictionary, p_oov, df, best_thr, n=ngram_n, normal_label=normal_label)
        metric_entry = {
            "block_index": block_index,
            "file": seq_file,
            "eta": metrics["eta"],
            "DR": metrics["DR"],
            "FPR": metrics["FPR"],
            "TP": metrics["TP"],
            "FP": metrics["FP"],
            "TN": metrics["TN"],
            "FN": metrics["FN"],
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"]
        }
        metric_log.append(metric_entry)
        block_index += 1

    metric_df = pd.DataFrame(metric_log)
    os.makedirs(os.path.dirname(metric_csv), exist_ok=True)
    metric_df.to_csv(metric_csv, index=False)
    print(f"[Static] ({ngram_n}-gram) Metrics saved: {metric_csv}")

if __name__ == "__main__":
    run_static_experiment_ngram()
