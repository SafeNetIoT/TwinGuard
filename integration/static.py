import os
import pandas as pd
from collections import deque
from tqdm import tqdm

import utils

# ======= CONFIG =======
ENCODED_SEQUENCE_DIR_MAIN = "../data/sequences_encoded"
ENCODED_SEQUENCE_DIR_XPOT = "../data/xpot/sequences_encoded"
SEQUENCE_BLOCKS_PER_DAY = 24
WINDOW_DAYS = 8
N_BLOCKS = WINDOW_DAYS * SEQUENCE_BLOCKS_PER_DAY
SELECTED_MODELS = ["rf", "mlp"]

def load_block_files(main_path, xpot_path):
    dfs = []
    if os.path.exists(main_path):
        dfs.append(pd.read_csv(main_path))
    if os.path.exists(xpot_path):
        dfs.append(pd.read_csv(xpot_path))
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return None

def run_static_experiment(model_type, window_days=WINDOW_DAYS):
    metric_csv = f"../data/xpot/static_model_metrics_{model_type}_w{window_days}.csv"
    window_files = deque(maxlen=N_BLOCKS)
    # Use the main pot as the anchor for time order
    file_stream = utils.stream_chronological_files(ENCODED_SEQUENCE_DIR_MAIN)

    print(f"[Static] ({model_type}) Filling initial window of {N_BLOCKS} blocks for training...")
    for i in tqdm(range(N_BLOCKS), desc="Initializing window"):
        f = next(file_stream)
        window_files.append(f)

    # --- TRAIN STATIC MODEL ---
    print(f"[Static] ({model_type}) Training on {N_BLOCKS} blocks (integrated data)...")
    train_dfs = []
    for f in window_files:
        block_file = os.path.basename(f)
        block_dir = os.path.basename(os.path.dirname(f))
        main_path = f
        xpot_path = os.path.join(ENCODED_SEQUENCE_DIR_XPOT, block_dir, block_file)
        df = load_block_files(main_path, xpot_path)
        if df is None or "category" not in df.columns or df["category"].dropna().empty:
            continue
        train_dfs.append(df)
    if not train_dfs:
        raise RuntimeError("No valid training data found!")
    df_train = pd.concat(train_dfs, ignore_index=True)
    X_train = utils.feature_engineering(df_train)
    y_train = df_train["category"].astype(str)
    if model_type == "rf":
        model, aux = utils.train_rf(X_train, y_train)
    elif model_type == "mlp":
        model, aux = utils.train_mlp(X_train, y_train)
    else:
        raise ValueError("Model not supported in static selection.")
    print(f"[Static] ({model_type}) Model trained.")

    # --- EVALUATE ON REMAINING BLOCKS ---
    metric_log = []
    block_index = N_BLOCKS
    for seq_file in tqdm(file_stream, desc=f"Testing ({model_type}) on session blocks"):
        block_file = os.path.basename(seq_file)
        block_dir = os.path.basename(os.path.dirname(seq_file))
        main_path = seq_file
        xpot_path = os.path.join(ENCODED_SEQUENCE_DIR_XPOT, block_dir, block_file)
        df = load_block_files(main_path, xpot_path)
        if df is None or "category" not in df.columns or df["category"].dropna().empty:
            block_index += 1
            continue
        acc, prec, rec, f1 = utils.evaluate_model(model, aux, df, model_type)
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
    print(f"[Static] ({model_type}) Metrics saved: {metric_csv}")

if __name__ == "__main__":
    for model_type in SELECTED_MODELS:
        run_static_experiment(model_type)
