import os
import pandas as pd
import numpy as np
import gensim.downloader as api
import logging
import re
import joblib
import utils

# ==== LOGGING SETUP ====
logging.basicConfig(
    filename="run_eval.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
console.setFormatter(formatter)
logging.getLogger().addHandler(console)
logger = logging.getLogger(__name__)

# ==== CONFIG ====
CSV_PATH = "top100_ips_all_requests.csv"
SENSITIVE_WORDS_PATH = "../data/base/sensitive_words_with_counts.txt"
BLOCK_SIZE_SECONDS = 3600
intervals = [10, 20, 30, 50, 60, 100, 200, 300] 
max_blocks = 3000
model_dir = '../vir2/stable_model'
category_labels = {0: "attempt", 1: "intrusion-control", 2: "scan"}  # edit if needed

# ==== Load and set embedding resources globally ====
logger.info("Loading GloVe vectors...")
glove = api.load("glove-wiki-gigaword-50")
logger.info("Loaded glove.")

with open(SENSITIVE_WORDS_PATH, "r") as f:
    sensitive_words = {line.split(":")[0].strip() for line in f.readlines()}
logger.info(f"Loaded {len(sensitive_words)} sensitive words.")

utils.set_embedding_resources(glove, sensitive_words)

# ==== Load Data ====
df = pd.read_csv(CSV_PATH)
df = df.sort_values('startTime').reset_index(drop=True)
logger.info(f"Dataset shape: {df.shape}")

rate2events = {interval: BLOCK_SIZE_SECONDS // interval for interval in intervals}
logger.info("Events per block for each interval: %s", rate2events)

# ==== Model checkpoint loader ====
def get_rf_model_checkpoints(model_dir):
    checkpoints = []
    for fname in os.listdir(model_dir):
        if fname.startswith('stable_rf_init'):
            checkpoints.append((0, fname))
        else:
            m = re.match(r'stable_rf_retrain_(\d+)\.joblib', fname)
            if m:
                checkpoints.append((int(m.group(1)), fname))
    checkpoints.sort()
    return checkpoints

# ==== Process Each Interval ====
for interval in intervals:
    logger.info(f"=== [Interval: {interval}s] ===")
    events_per_block = rate2events[interval]
    max_rows = max_blocks * events_per_block
    df_limited = df.iloc[:max_rows].copy()
    df_limited['block_idx'] = [i // events_per_block for i in range(len(df_limited))]

    logger.info(f"[{interval}s] Number of blocks: {df_limited['block_idx'].nunique()}")
    logger.info(f"[{interval}s] Shape: {df_limited.shape}")

    # --- Feature Engineering ---
    out_csv = f"../data/lowrate/features_{interval}.csv"
    if os.path.exists(out_csv):
        logger.info(f"[{interval}s] Feature file exists, loading: {out_csv}")
        df_features = pd.read_csv(out_csv)
    else:
        logger.info(f"[{interval}s] Feature file not found, running feature engineering.")
        df_features = utils.feature_engineer_df(df_limited)
        df_features.to_csv(out_csv, index=False)
        # ALWAYS reload for consistent types
        df_features = pd.read_csv(out_csv)
        logger.info(f"[{interval}s] Saved features: {out_csv}")


    # --- Model Checkpoints ---
    rf_checkpoints = get_rf_model_checkpoints(model_dir)
    results = []
    cur_model = None
    cur_model_point = None
    model_iter = iter(rf_checkpoints)
    next_model_point, next_model_fname = next(model_iter, (float('inf'), None))

    for block_idx, block_df in df_features.groupby('block_idx'):
        # Load correct model checkpoint if needed
        while block_idx >= next_model_point:
            logger.info(f"Loading model for block {next_model_point}: {next_model_fname}")
            cur_model = joblib.load(os.path.join(model_dir, next_model_fname))
            cur_model_point = next_model_point
            next_model_point, next_model_fname = next(model_iter, (float('inf'), None))

        # --- Evaluate with FP ---
        metrics, fp_dict = utils.evaluate_model(cur_model, None, block_df, model_type='rf', logger=logger, return_fp=True)
        acc, prec, rec, f1 = metrics

        # Fill missing FPs as 0 for all known classes
        fp_per_class = {k: fp_dict.get(k, 0) for k in category_labels.keys()} if fp_dict else {k: np.nan for k in category_labels.keys()}
        
        logger.info(
            f"[Interval {interval}s][Block {block_idx}] n={len(block_df)} | "
            f"acc={acc:.3f} prec={prec:.3f} rec={rec:.3f} f1={f1:.3f} | "
            f"FPs={fp_per_class}"
        )
        row = {
            'block_idx': block_idx,
            'model_point': cur_model_point,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'n_samples': len(block_df)
        }
        # Add FP per class (e.g., fp_attempt, fp_intrusion_control, fp_scan)
        for cat_idx, cat_name in category_labels.items():
            row[f'fp_{cat_name}'] = fp_per_class.get(cat_idx, np.nan)
        results.append(row)

    metrics_df = pd.DataFrame(results)
    metrics_out_csv = f"../data/lowrate/metrics_{interval}.csv"
    metrics_df.to_csv(metrics_out_csv, index=False)
    logger.info(f"[{interval}s] Saved metrics: {metrics_out_csv}")

    # --- Summary ---
    summary = metrics_df[['accuracy', 'precision', 'recall', 'f1']].agg(['mean', 'median', 'std', 'min', 'max'])
    fp_cols = [f'fp_{name}' for name in category_labels.values()]
    fp_summary = metrics_df[fp_cols].agg(['mean', 'median', 'std', 'min', 'max']) if all(col in metrics_df.columns for col in fp_cols) else None
    logger.info(f"[{interval}s] Metrics Summary:\n{summary}")
    if fp_summary is not None:
        logger.info(f"[{interval}s] False Positive Summary:\n{fp_summary}")

logger.info("All intervals processed and evaluated.")
