import numpy as np
import pandas as pd
import re
from collections import Counter
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


VALID_CATEGORIES = ["scan", "attempt", "intrusion-control"]

# -------------------- DATA LOADER --------------------
def stream_chronological_files(sequence_dir):
    for day in sorted(os.listdir(sequence_dir)):
        day_path = os.path.join(sequence_dir, day)
        if not os.path.isdir(day_path):
            continue
        for fname in sorted(os.listdir(day_path)):
            if fname.endswith(".csv"):
                yield os.path.join(day_path, fname)

def filter_valid_categories(df, valid_categories=VALID_CATEGORIES):
    return df[df["category"].astype(str).isin(valid_categories)]

def extract_ngrams(tokens, n=1):
    """Return list of n-grams (as tuple) from a list of tokens."""
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)] if len(tokens) >= n else []

def tokenize_uri(uri):
    if pd.isna(uri):
        return []
    # Use your RFC3986 split
    return [w for w in re.split(r'[:\/?#\[\]@!$&\'()*+,;=]', str(uri)) if w]


def train_1gram(df_train, uri_col="http_uri"):
    words = []
    for uri in df_train[uri_col]:
        words.extend(tokenize_uri(uri))
    counts = Counter(words)
    total = sum(counts.values())
    if total == 0:
        raise ValueError("No tokens found in training data!")
    dictionary = {w: c / total for w, c in counts.items()}
    pmin = min(dictionary.values())
    p_oov = pmin ** 3
    return dictionary, p_oov

def score_1gram(dictionary, p_oov, uri):
    tokens = tokenize_uri(uri)
    if not tokens:
        return 0.0
    log_probs = [np.log(dictionary.get(t, p_oov)) for t in tokens]
    return -np.mean(log_probs)

def evaluate_1gram_block(dictionary, p_oov, df, threshold, normal_label="normal"):
    if "http_uri" not in df.columns or "category" not in df.columns or len(df) == 0:
        return {
            "DR": np.nan, "FPR": np.nan, "eta": np.nan, "TP": 0, "FP": 0, "TN": 0, "FN": 0,
            "accuracy": np.nan, "precision": np.nan, "recall": np.nan, "f1_score": np.nan
        }
    scores = df['http_uri'].apply(lambda uri: score_1gram(dictionary, p_oov, uri)).values
    true = df["category"].astype(str)
    pred = (scores > threshold)
    # anomaly label = True, normal = False
    y_true = (true != normal_label)
    TP = np.sum((pred == 1) & (y_true == 1))
    FP = np.sum((pred == 1) & (y_true == 0))
    TN = np.sum((pred == 0) & (y_true == 0))
    FN = np.sum((pred == 0) & (y_true == 1))
    DR = TP / (TP + FN) if (TP + FN) > 0 else 0
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
    eta = np.sqrt(DR * (1 - FPR))
    # Compute sklearn metrics
    try:
        acc = accuracy_score(y_true, pred)
        prec = precision_score(y_true, pred, zero_division=0)
        rec = recall_score(y_true, pred, zero_division=0)
        f1 = f1_score(y_true, pred, zero_division=0)
    except Exception:
        acc = prec = rec = f1 = np.nan
    return {
        "DR": DR, "FPR": FPR, "eta": eta, "TP": TP, "FP": FP, "TN": TN, "FN": FN,
        "accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1
    }


from collections import Counter

def train_ngram(df_train, n=1, uri_col="http_uri"):
    ngram_list = []
    for uri in df_train[uri_col]:
        tokens = tokenize_uri(uri)
        ngram_list.extend(extract_ngrams(tokens, n=n))
    counts = Counter(ngram_list)
    total = sum(counts.values())
    if total == 0:
        raise ValueError("No tokens found in training data!")
    dictionary = {w: c/total for w, c in counts.items()}
    pmin = min(dictionary.values())
    p_oov = pmin ** 3
    return dictionary, p_oov

def score_ngram(dictionary, p_oov, uri, n=1):
    tokens = tokenize_uri(uri)
    ngrams = extract_ngrams(tokens, n=n)
    if not ngrams:
        return 0.0
    log_probs = [np.log(dictionary.get(ng, p_oov)) for ng in ngrams]
    return -np.mean(log_probs)

def evaluate_ngram_block(dictionary, p_oov, df, threshold, n=1, normal_label="normal"):
    if "http_uri" not in df.columns or "category" not in df.columns or len(df) == 0:
        return {
            "DR": np.nan, "FPR": np.nan, "eta": np.nan, "TP": 0, "FP": 0, "TN": 0, "FN": 0,
            "accuracy": np.nan, "precision": np.nan, "recall": np.nan, "f1_score": np.nan
        }
    scores = df['http_uri'].apply(lambda uri: score_ngram(dictionary, p_oov, uri, n=n)).values
    true = df["category"].astype(str)
    pred = (scores > threshold)
    y_true = (true != normal_label)
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    TP = np.sum((pred == 1) & (y_true == 1))
    FP = np.sum((pred == 1) & (y_true == 0))
    TN = np.sum((pred == 0) & (y_true == 0))
    FN = np.sum((pred == 0) & (y_true == 1))
    DR = TP / (TP + FN) if (TP + FN) > 0 else 0
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
    eta = np.sqrt(DR * (1 - FPR))
    try:
        acc = accuracy_score(y_true, pred)
        prec = precision_score(y_true, pred, zero_division=0)
        rec = recall_score(y_true, pred, zero_division=0)
        f1 = f1_score(y_true, pred, zero_division=0)
    except Exception:
        acc = prec = rec = f1 = np.nan
    return {
        "DR": DR, "FPR": FPR, "eta": eta, "TP": TP, "FP": FP, "TN": TN, "FN": FN,
        "accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1
    }

'''
Test for RNN preprocessing Eunaicy et al.(2022)
'''


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def preprocess_for_dl(df, return_encoders=False):
    """
    Preprocesses your web log dataframe for deep learning models.
    - Selects columns similar to those in benchmark deep learning papers.
    - Encodes categoricals and normalizes numerics.
    - Returns X, y, and optionally encoders/scaler.
    """

    # --- 1. Columns to keep ---
    cols_to_keep = [
        "protocol", "http_method", "http_status", "http_uri", "http_body", "http_accept",
        "http_accept_encoding", "http_user_agent", "http_content_type", "http_connection",
        "hostIP", "host_as_org", "category"
    ]
    df = df[[c for c in cols_to_keep if c in df.columns]].copy()

    # --- 2. Flatten/clean complex fields ---
    def flatten_col(x):
        if isinstance(x, str) and (x.startswith('[') or x.startswith('{')):
            try:
                return ','.join(eval(x)) if isinstance(eval(x), (list, tuple)) else str(eval(x))
            except:
                return str(x)
        return str(x)
    for c in ["http_accept", "http_accept_encoding", "http_user_agent", "http_connection"]:
        if c in df.columns:
            df[c] = df[c].astype(str).apply(flatten_col)

    # --- 3. Identify columns ---
    categorical_cols = [c for c in [
        "protocol", "http_method", "http_user_agent", "http_content_type", "http_connection", "host_as_org", "hostIP"
    ] if c in df.columns]
    text_cols = [c for c in ["http_uri", "http_body"] if c in df.columns]
    numeric_cols = [c for c in ["http_status"] if c in df.columns]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # --- 4. Label encode categoricals ---
    encoders = {}
    for c in categorical_cols + text_cols:
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c].astype(str).fillna("UNK"))
        encoders[c] = le

    # --- 5. Normalize numeric fields ---
    scaler = MinMaxScaler()
    if numeric_cols:
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    else:
        scaler = None

    # --- 6. Prepare features and label ---
    feature_cols = [c for c in df.columns if c != "category"]
    # Force all features numeric and float32
    X = df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.float32).values

    # Encode target for multi-class
    le_target = LabelEncoder()
    y = le_target.fit_transform(df["category"].astype(str))
    encoders['category'] = le_target

    if return_encoders:
        return X, y, encoders, scaler
    else:
        return X, y

def build_full_http_text(row, fields=['protocol', 'http_method', 'http_status', 'http_uri']):
    """Builds a single string for TF-IDF from various HTTP fields."""
    out = []
    for f in fields:
        if f in row and pd.notna(row[f]) and str(row[f]).strip():
            val = str(row[f])
            out.append(f"{f}:{val}")
    return ' '.join(out)  # or '\n'.join(out) if you want linebreaks