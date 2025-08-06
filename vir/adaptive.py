import os
import pandas as pd
import numpy as np
from collections import Counter, deque
from tqdm import tqdm
import joblib
import logging
import sys
import re
from utils import (
    stream_chronological_files, filter_valid_categories, feature_engineering,
    compute_feature_mean, compute_data_drift, block_sequences, compute_unknown_rate,
    train_rf, train_mlp
)

# ===== USER SELECTION =====
SELECTED_MODELS = ["rf", "mlp"]

# ==== CONFIG ====
ENCODED_SEQUENCE_DIR = "../data/sequences_encoded"
MODEL_SAVE_FOLDER = "stable_model"
METRIC_CSV_BASE = "../data/vir/stable_model_metrics"
STABLE_WINDOW_DAYS = 8
SEQUENCE_BLOCKS_PER_DAY = 24
N_BLOCKS = STABLE_WINDOW_DAYS * SEQUENCE_BLOCKS_PER_DAY

FREQUENCY_THRESHOLD = 100
UNKNOWN_RATE_THRESH = 0.3
COOLDOWN_BLOCKS = 5
UNKNOWN_RATE_FOR_DRIFT_TRIGGER = 0.2

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model(window_files, model_type="rf"):
    dfs = []
    for f in window_files:
        df = pd.read_csv(f)
        df = filter_valid_categories(df)
        if len(df) == 0:
            continue
        dfs.append(df)
    if not dfs:
        raise ValueError("No valid data for training!")
    df_full = pd.concat(dfs, ignore_index=True)
    X = feature_engineering(df_full)
    y = df_full["category"].astype(str)
    if model_type == "rf":
        return train_rf(X, y)
    elif model_type == "mlp":
        return train_mlp(X, y)
    else:
        raise ValueError("Unsupported model type!")

def evaluate_model(model, scaler, df, model_type="rf"):
    df = filter_valid_categories(df)
    X = feature_engineering(df)
    y = df["category"].astype(str)
    if len(X) == 0:
        return [np.nan]*4
    try:
        if model_type == "mlp" and scaler is not None:
            X = scaler.transform(X)
        y_pred = model.predict(X)
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred, average='macro', zero_division=0)
        rec = recall_score(y, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y, y_pred, average='macro', zero_division=0)
        return acc, prec, rec, f1
    except Exception as e:
        logger.error(f"Model evaluation error: {e}")
        return [np.nan]*4

def detect_data_drift(drift_buffer, drift, mad_mult=3, iqr_mult=3.0):
    arr = np.array(drift_buffer)
    if len(arr) < 20:
        return False
    median = np.median(arr)
    mad = np.median(np.abs(arr - median))
    mad_thresh = median + mad_mult * mad
    Q1 = np.percentile(arr, 25)
    Q3 = np.percentile(arr, 75)
    iqr = Q3 - Q1
    iqr_thresh = Q3 + iqr_mult * iqr
    return (drift > mad_thresh) and (drift > iqr_thresh)

def run_for_model(model_type):
    METRIC_CSV = f"{METRIC_CSV_BASE}_{model_type}_w{STABLE_WINDOW_DAYS}.csv"
    LOG_PATH = f"{METRIC_CSV_BASE}_{model_type}_retrain.log"

    # --- Open log file for retrain events ---
    log_f = open(LOG_PATH, "w")

    all_files = list(stream_chronological_files(ENCODED_SEQUENCE_DIR))
    baseline_files = all_files[:N_BLOCKS]
    eval_files = all_files[N_BLOCKS:]

    # --- Build sensitive set for sequence encoding ---
    word_counter = Counter()
    sensitive_set = set()
    for f in baseline_files:
        df = pd.read_csv(f)
        df = filter_valid_categories(df)
        for col in ["http_uri"]:
            if col in df.columns:
                for text in df[col].dropna():
                    for word in re.findall(r"\b\w+\b", str(text).lower()):
                        word_counter[word] += 1
                        if word_counter[word] >= FREQUENCY_THRESHOLD:
                            sensitive_set.add(word)

    window_files = deque(maxlen=N_BLOCKS)
    window_means = deque(maxlen=N_BLOCKS)
    drift_buffer = deque(maxlen=30)
    model_sequences = set()

    last_model_path = None

    for f in baseline_files:
        df = pd.read_csv(f)
        df = filter_valid_categories(df)
        df_feat = feature_engineering(df)
        window_means.append(compute_feature_mean(df_feat, feature_list=df_feat.columns))
        seqs = block_sequences(df, sensitive_set)
        model_sequences.update(seqs)
        window_files.append(f)
    baseline_vec = np.mean(np.stack(window_means), axis=0)

    # --- Train Initial Model ---
    model, scaler = train_model(list(window_files), model_type=model_type)
    model_version = "init"
    os.makedirs(MODEL_SAVE_FOLDER, exist_ok=True)
    model_path = os.path.join(MODEL_SAVE_FOLDER, f"stable_{model_type}_{model_version}.joblib")
    joblib.dump(model, model_path)
    last_model_path = model_path

    # ========== MAIN EVAL/ADAPTIVE LOOP ==========
    block_index = N_BLOCKS
    last_retrain_block = block_index
    metric_log = []

    for seq_file in tqdm(eval_files, desc=f"Processing test blocks [{model_type}]"):
        df = pd.read_csv(seq_file, low_memory=False)
        df = filter_valid_categories(df)
        df_feat = feature_engineering(df)

        # ---- DATA DRIFT: L2 NORM ----
        current_mean = compute_feature_mean(df_feat, feature_list=df_feat.columns)
        drift = compute_data_drift(current_mean, baseline_vec)
        drift_buffer.append(drift)

        # ---- FEATURE DRIFT: UNKNOWN RATE ----
        seqs_this_block = block_sequences(df, sensitive_set)
        unknown_rate, total_sequences, unknown_sequences = compute_unknown_rate(seqs_this_block, model_sequences)

        # ---- METRICS ----
        acc, prec, rec, f1 = evaluate_model(model, scaler, df, model_type=model_type)

        # ---- RETRAIN LOGIC ----
        retrain = False
        retrain_reason = ""
        cooldown_ok = (block_index - last_retrain_block) > COOLDOWN_BLOCKS
        data_drift_flag = detect_data_drift(drift_buffer, drift)

        feature_drift_flag = (unknown_rate > UNKNOWN_RATE_THRESH)
        if (data_drift_flag or feature_drift_flag) and cooldown_ok:
            retrain = True
            retrain_reason = []
            if data_drift_flag: retrain_reason.append("data_drift")
            if feature_drift_flag: retrain_reason.append("unknown_rate")
            retrain_reason = ",".join(retrain_reason)


        '''
        feature_drift_flag = (unknown_rate > UNKNOWN_RATE_THRESH)
        # Combined retrain logic
        if data_drift_flag and unknown_rate > UNKNOWN_RATE_FOR_DRIFT_TRIGGER and cooldown_ok:
            retrain = True
            retrain_reason = []
            if data_drift_flag: retrain_reason.append("data_drift")
            if feature_drift_flag: retrain_reason.append("unknown_rate")
            retrain_reason = ",".join(retrain_reason)
        '''    


        '''
        # Multiple reasons for retraining
        if data_drift_flag and unknown_rate > UNKNOWN_RATE_FOR_DRIFT_TRIGGER and cooldown_ok:
            retrain = True
            retrain_reason = []
            retrain_reason.append("data_drift")
        elif feature_drift_flag and cooldown_ok:
            retrain = True
            retrain_reason = []
            retrain_reason.append("unknown_rate")
        retrain_reason = ",".join(retrain_reason)
        '''


        metric_entry = {
            "block_index": block_index,
            "file": seq_file,
            "drift": drift,
            "unknown_rate": unknown_rate,
            "total_sequences": total_sequences,
            "unknown_sequences": unknown_sequences,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "model_info": model_version,
            "retrain_event": retrain,
            "retrain_reason": retrain_reason,
        }

        if retrain:
            # Delete previous model file before saving new one
            #if last_model_path and os.path.exists(last_model_path):
            #    os.remove(last_model_path)

            window_files.append(seq_file)
            window_means.append(current_mean)
            model_sequences.update(seqs_this_block)
            while len(window_files) > N_BLOCKS:
                window_files.popleft()
            while len(window_means) > N_BLOCKS:
                window_means.popleft()
            baseline_vec = np.mean(np.stack(window_means), axis=0)

            model, scaler = train_model(list(window_files), model_type=model_type)
            model_version = f"retrain_{block_index}"
            model_path = os.path.join(MODEL_SAVE_FOLDER, f"stable_{model_type}_{model_version}.joblib")
            joblib.dump(model, model_path)
            last_model_path = model_path
            last_retrain_block = block_index
            acc_post, prec_post, rec_post, f1_post = evaluate_model(model, scaler, df, model_type=model_type)
            metric_entry.update({
                "accuracy_post_retrain": acc_post,
                "precision_post_retrain": prec_post,
                "recall_post_retrain": rec_post,
                "f1_score_post_retrain": f1_post,
            })
            # ---- Log retrain event to log file ----
            log_f.write(
                f"[RETRAIN] Block {block_index} | {model_type} | Reason(s): {retrain_reason} | drift={drift:.3f}, unknown_rate={unknown_rate:.3f} "
                f"| acc={acc_post:.3f}, prec={prec_post:.3f}, rec={rec_post:.3f}, f1={f1_post:.3f}\n"
            )
            log_f.flush()
            print(f"[RETRAIN] Block {block_index}: Model retrain! Reason(s): {retrain_reason} drift={drift:.3f}, unknown_rate={unknown_rate:.3f}. "
                  f"acc={acc_post:.3f}, prec={prec_post:.3f}, rec={rec_post:.3f}, f1={f1_post:.3f}")
        else:
            metric_entry.update({
                "accuracy_post_retrain": np.nan,
                "precision_post_retrain": np.nan,
                "recall_post_retrain": np.nan,
                "f1_score_post_retrain": np.nan,
            })

        metric_log.append(metric_entry)
        block_index += 1

    metric_df = pd.DataFrame(metric_log)
    os.makedirs(os.path.dirname(METRIC_CSV), exist_ok=True)
    metric_df.to_csv(METRIC_CSV, index=False)
    log_f.close()
    print(f"✅ Metrics log saved: {METRIC_CSV}")
    print(f"✅ Retrain events logged: {LOG_PATH}")

if __name__ == "__main__":
    for model_type in SELECTED_MODELS:
        print(f"==== Running stable pool with retrain for model: {model_type} ====")
        run_for_model(model_type)

