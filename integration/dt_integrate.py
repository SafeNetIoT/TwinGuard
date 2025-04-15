import os
import pandas as pd
import joblib
import json
import re
import collections
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from utils import feature_engineering, feature_engineering_xgb
import warnings

# === CONFIG ===
ENCODED_FOLDER = "../data/encoded"
XPOT_FOLDER = "../data/integration/encoded"
TRIE_PATH = "../data/integration/trie/aggregated_probabilistic_attack_tree.json"
SENSITIVE_PATH = "../data/integration/trie/sensitive_words_with_counts.txt"
LABEL_MAPPING_PATH = "../data/integration/encoded/label_mapping.json"
DAILY_RAW_FOLDER = "../data/daily"
WINDOW_SIZE = 6
ACCURACY_THRESHOLD = 6.0
UNKNOWN_RATE_THRESHOLD = 3.0
FREQUENCY_THRESHOLD = 20
LABEL_COL = "category"
MODEL_SAVE_FOLDER = "adaptive_models"
start_date = "2025-03-20"
end_date = "2025-03-29"
RESULTS_FOLDER = f"../data/df/digital_twin_integration_log_win{WINDOW_SIZE}.csv"

os.makedirs(MODEL_SAVE_FOLDER, exist_ok=True)

with open(LABEL_MAPPING_PATH) as f:
    label_map = json.load(f)
METHOD_ID_TO_NAME = {int(k): v for k, v in label_map["http_method"].items()}
CATEGORY_ID_TO_NAME = {int(k): v for k, v in label_map["category"].items()}


# Load combined data (main + xpot) by date
def load_day_data(day_str):
    paths = [
        (os.path.join(ENCODED_FOLDER, f"http_day_{day_str}_encoded.csv"), "original"),
        (os.path.join(XPOT_FOLDER, f"test_day_{day_str}_encoded.csv"), "xpot"),
    ]
    dfs = []
    for path, source in paths:
        if os.path.exists(path):
            with warnings.catch_warnings():
                warnings.simplefilter(action='ignore', category=pd.errors.DtypeWarning)
                df = pd.read_csv(path)
            df = df[df[LABEL_COL].isin(CATEGORY_ID_TO_NAME.keys())]
            df["source"] = source  # tag where it came from
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else None


def train_pipeline(train_days, model_type):
    dfs = [load_day_data(day) for day in train_days]
    dfs = [df for df in dfs if df is not None and not df.empty]
    df_full = pd.concat(dfs, ignore_index=True)
    X = df_full.copy()
    y = X[LABEL_COL]
    if model_type == "xgb":
        y = y.astype(int)
    else:
        y = y.astype(str)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    if model_type == "rf":
        classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        transformer = FunctionTransformer(feature_engineering, validate=False)
    elif model_type == "xgb":
        classifier = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, n_jobs=-1, 
                                   eval_metric="mlogloss", random_state=42)
        transformer = FunctionTransformer(feature_engineering_xgb, validate=False)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    pipeline = Pipeline([
        ("feature_engineering", transformer),
        ("classifier", classifier)
    ])
    pipeline.fit(X_train, y_train)
    y_val_pred = pipeline.predict(X_val)
    acc = accuracy_score(y_val, y_val_pred)
    return pipeline, acc


def tokenize(text):
    return re.findall(r"\b\w+\b", str(text).lower())

def load_sensitive_words(path):
    with open(path) as f:
        return {line.split(":")[0].strip() for line in f.readlines()}

def load_trie(path):
    with open(path) as f:
        return json.load(f)

def extract_keywords(path, sensitive_words):
    tokens = re.split(r"[/._\-]", str(path).lower())
    matched_keywords = {token for token in tokens if token in sensitive_words}
    return ",".join(sorted(matched_keywords)) if matched_keywords else "<no-matching>"

def extract_sensitive_words(df, threshold=50):
    uri_words = Counter()
    body_words = Counter()
    for uri in df["http_uri"].dropna():
        uri_words.update(tokenize(uri))
    for body in df["http_body"].dropna():
        body_words.update(tokenize(body))
    sensitive_words = {word: count for word, count in uri_words.items() if count >= threshold}
    sensitive_words.update({word: count for word, count in body_words.items() if count >= threshold})
    return sensitive_words

def save_sensitive_words(sensitive_words, output_path):
    with open(output_path, "w") as f:
        for word, count in sorted(sensitive_words.items(), key=lambda x: x[1], reverse=True):
            f.write(f"{word}: {count}\n")
    print(f" Sensitive dictionary updated: {len(sensitive_words)} words")

def insert_probabilistic_attack_path(tree, sequence, attack_type, count):
    node = tree
    for i, step in enumerate(sequence):
        if step not in node:
            node[step] = {"count": 0, "probabilities": defaultdict(float), "next_step": defaultdict(int)}
        node[step]["count"] += count
        if i < len(sequence) - 1:
            next_step = sequence[i + 1]
            node[step]["next_step"][next_step] += count
        node = node[step]
    node["probabilities"][attack_type] += count

def normalize_probabilities(tree):
    for key, value in tree.items():
        if isinstance(value, dict):
            if "probabilities" in value:
                total = sum(value["probabilities"].values())
                if total > 0:
                    value["probabilities"] = {k: v / total for k, v in value["probabilities"].items()}
            if "next_step" in value:
                total_next = sum(value["next_step"].values())
                if total_next > 0:
                    value["next_step"] = {k: v / total_next for k, v in value["next_step"].items()}
            normalize_probabilities(value)

def build_probabilistic_trie(df, sensitive_words):
    df = df.copy()
    def normalize_status(s):
        try:
            return str(int(float(s)))
        except:
            return "Unknown"

    df["http_status_str"] = df["http_status"].apply(normalize_status)


    # Normalize all methods to their string label using the mapping
    def normalize_method(m):
        try:
            if isinstance(m, str) and m.upper() in METHOD_ID_TO_NAME.values():
                return m.upper()
            return METHOD_ID_TO_NAME.get(int(float(m)), str(m)).upper()
        except:
            return str(m).upper()

    df["http_method_str"] = df["http_method"].apply(normalize_method)
    df["uri_keywords"] = df["http_uri"].apply(lambda uri: extract_keywords(uri, sensitive_words))
    df["attack_sequence"] = list(zip(df["http_method_str"], df["http_status_str"], df["uri_keywords"]))


    attack_counter = Counter(zip(df["attack_sequence"], df["category"]))
    tree = {}
    for (seq, label), count in attack_counter.items():
        insert_probabilistic_attack_path(tree, seq, label, count)

    normalize_probabilities(tree)
    return tree



def jsonify_trie(node):
    if isinstance(node, collections.defaultdict):
        node = dict(node)
    if isinstance(node, dict):
        return {k: jsonify_trie(v) for k, v in node.items()}
    return node    

def build_sequence(row, sensitive_words):
    method_raw = row["http_method"]
    try:
        if isinstance(method_raw, str) and method_raw.upper() in METHOD_ID_TO_NAME.values():
            method_str = method_raw.upper()
        else:
            method_str = METHOD_ID_TO_NAME.get(int(float(method_raw)), str(method_raw)).upper()
    except:
        method_str = str(method_raw).upper()

    # Normalize status
    status_raw = row["http_status"]
    try:
        status = str(int(float(status_raw)))
    except:
        status = "Unknown"

    uri_keywords = extract_keywords(row.get("http_uri", ""), sensitive_words)
    return (method_str, status, uri_keywords)


def is_known_sequence(seq, trie):
    node = trie
    for step in seq:
        if step in node:
            node = node[step]
        else:
            return False
    return True

def check_daily_pattern(df, base_sensitive_words, base_trie):
    df["http_status"] = df["http_status"].fillna("Unknown")
    uri_words = Counter()
    body_words = Counter()
    for uri in df["http_uri"].dropna():
        uri_words.update(tokenize(uri))
    for body in df["http_body"].dropna():
        body_words.update(tokenize(body))
    all_words = set(uri_words.keys()).union(set(body_words.keys()))
    new_sensitive_words = sorted(
        word for word in all_words - base_sensitive_words
        if uri_words[word] + body_words[word] >= FREQUENCY_THRESHOLD
    )
    new_sequences = []
    for _, row in df.iterrows():
        seq = build_sequence(row, base_sensitive_words)
        if not is_known_sequence(seq, base_trie):
            new_sequences.append(seq)
    return len(df), len(new_sequences), new_sensitive_words, new_sequences

def run_adaptive_loop(start_date, end_date):
    log = []
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")

    train_start = start_date
    train_days = [(train_start + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(WINDOW_SIZE)]

    pipeline_rf, base_acc_rf = train_pipeline(train_days, model_type="rf")
    pipeline_xgb, base_acc_xgb = train_pipeline(train_days, model_type="xgb")

    print(f"📚 Initial RF model trained on: {train_days} | Acc: {base_acc_rf:.2f}")
    print(f"📚 Initial XGB model trained on: {train_days} | Acc: {base_acc_xgb:.2f}")

    window_df = []
    original_total, xpot_total = 0, 0

    for day in train_days:
        df_day = load_day_data(day)
        if df_day is not None:
            window_df.append(df_day)
            original_count = df_day[df_day["source"] == "original"].shape[0]
            xpot_count = df_day[df_day["source"] == "xpot"].shape[0]
            original_total += original_count
            xpot_total += xpot_count
            print(f" Loaded for {day} → Original: {original_count}, XPOT: {xpot_count}")
        else:
            print(f"⚠️ No data available for {day} from either source.")

    if not window_df:
        print("No training data available. Exiting.")
        return

    df_window = pd.concat(window_df, ignore_index=True)
    print("Unique status values in merged training set:", df_window["http_status"].unique())

    # Debug: show sources
    original_total, xpot_total = 0, 0
    for df_day in window_df:
        original_total += df_day[df_day["source"] == "original"].shape[0]
        xpot_total += df_day[df_day["source"] == "xpot"].shape[0]

    print(f" Training Window Source Breakdown:")
    print(f"  • Original Samples: {original_total}")
    print(f"  • XPOT Samples:     {xpot_total}")
    print(f"  • Total Merged:     {original_total + xpot_total}")



    updated_words = extract_sensitive_words(df_window, threshold=FREQUENCY_THRESHOLD)
    save_sensitive_words(updated_words, SENSITIVE_PATH)
    updated_trie = build_probabilistic_trie(df_window, updated_words)
    print("Top methods in Trie:", list(updated_trie.keys())[:5])


    # Debug
    print("Sample XPOT training sequences:")
    df_xpot = df_window[df_window["source"] == "xpot"]
    if not df_xpot.empty:
        print("Sample XPOT training sequences:")
        for _, row in df_xpot.sample(min(3, len(df_xpot))).iterrows():
            print("  →", build_sequence(row, set(updated_words.keys())))
    else:
        print("⚠️ No XPOT data found in current training window.")

    # debug
    print("Sample path in updated trie:")
    for method in updated_trie:
        print("  ↪ Method:", method)
        if isinstance(updated_trie[method], dict):
            for status in updated_trie[method].get("next_step", {}):
                print("     ↪ Status:", status)
                break
        break

    with open(TRIE_PATH, "w") as f:
        json.dump(jsonify_trie(updated_trie), f, indent=2)
    sensitive_words = set(updated_words.keys())
    base_trie = load_trie(TRIE_PATH)

    # debug
    print("\n🔍 Checking XPOT sample sequence presence in Trie:")
    sample_xpot = df_window[df_window["source"] == "xpot"].head(5)
    for _, row in sample_xpot.iterrows():
        seq = build_sequence(row, sensitive_words)
        print("Sequence:", seq)
        match = is_known_sequence(seq, base_trie)
        print(" Matched in Trie" if match else "  ❌ Not matched in Trie")


    test_day = train_start
    while test_day <= end_date:
        day_str = test_day.strftime("%Y-%m-%d")
        df_test = load_day_data(day_str)
        if df_test is None or df_test.empty:
            test_day += timedelta(days=1)
            continue

        y_test_str = df_test[LABEL_COL].astype(str)
        y_test_int = df_test[LABEL_COL].astype(int)

        acc_rf = accuracy_score(y_test_str, pipeline_rf.predict(df_test))
        acc_xgb = accuracy_score(y_test_int, pipeline_xgb.predict(df_test))

        acc_drop_rf = (base_acc_rf - acc_rf) * 100
        acc_drop_xgb = (base_acc_xgb - acc_xgb) * 100

        total_count, unknown_count, new_words, new_sequences = check_daily_pattern(df_test, sensitive_words, base_trie)
        unknown_rate = (unknown_count / total_count) * 100 if total_count > 0 else 0

        print(f"\n{day_str}")
        print(f"  • RF Acc: {acc_rf*100:.2f}% (↓ {acc_drop_rf:.2f}%)")
        print(f"  • XGB Acc: {acc_xgb*100:.2f}% (↓ {acc_drop_xgb:.2f}%)")
        print(f"  • New Words: {len(new_words)} | New Sequences: {len(new_sequences)} | Unknown Rate: {unknown_rate:.2f}%")
        for seq in new_sequences[:3]:
            print("  → Unmatched:", seq)

        needs_update = (acc_drop_rf > ACCURACY_THRESHOLD and acc_drop_xgb > ACCURACY_THRESHOLD) or unknown_rate > UNKNOWN_RATE_THRESHOLD

        log.append({
            "Train Window Start": train_days[0],
            "Train Window End": train_days[-1],
            "Test Date": day_str,
            "RF Baseline Accuracy": base_acc_rf * 100,
            "RF Test Accuracy": acc_rf * 100,
            "RF Accuracy Drop": acc_drop_rf,
            "XGB Baseline Accuracy": base_acc_xgb * 100,
            "XGB Test Accuracy": acc_xgb * 100,
            "XGB Accuracy Drop": acc_drop_xgb,
            "New Sensitive Words": len(new_words),
            "New Sequences": len(new_sequences),
            "Unknown Rate": unknown_rate,
            "Model Updated": "Yes" if needs_update else "No"
        })

        if needs_update:
            new_end = test_day
            new_start = new_end - timedelta(days=WINDOW_SIZE - 1)
            if new_start < start_date:
                print(f"⚠️ Not enough past data to retrain at {day_str}, skipping update.")
                break

            train_days = [(new_start + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(WINDOW_SIZE)]
            pipeline_rf, base_acc_rf = train_pipeline(train_days, model_type="rf")
            pipeline_xgb, base_acc_xgb = train_pipeline(train_days, model_type="xgb")
            joblib.dump(pipeline_rf, os.path.join(MODEL_SAVE_FOLDER, f"rf_pipeline_updated_{day_str}.joblib"))
            joblib.dump(pipeline_xgb, os.path.join(MODEL_SAVE_FOLDER, f"xgb_pipeline_updated_{day_str}.joblib"))

            print(f"Models retrained and saved for {day_str}")

            # standard loader that includes XPOT
            window_df = []
            for day in train_days:
                df_day = load_day_data(day)
                if df_day is not None:
                    window_df.append(df_day)

            df_window = pd.concat(window_df, ignore_index=True)
            updated_words = extract_sensitive_words(df_window, threshold=FREQUENCY_THRESHOLD)
            save_sensitive_words(updated_words, SENSITIVE_PATH)
            updated_trie = build_probabilistic_trie(df_window, updated_words)
            with open(TRIE_PATH, "w") as f:
                json.dump(jsonify_trie(updated_trie), f, indent=2)
            sensitive_words = set(updated_words.keys())
            base_trie = load_trie(TRIE_PATH)

        test_day += timedelta(days=1)

    df_log = pd.DataFrame(log)
    df_log.to_csv(RESULTS_FOLDER, index=False)
    print("Evaluation log saved as", RESULTS_FOLDER)

if __name__ == "__main__":
    run_adaptive_loop(start_date, end_date)