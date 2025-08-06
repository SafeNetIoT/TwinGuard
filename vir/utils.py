import os
import numpy as np
import pandas as pd
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense, Conv1D, Flatten, LSTM, Dropout

# -------------------- CONFIG --------------------
VALID_CATEGORIES = ["0", "1", "2"]
FEATURE_LIST = [
    "http_method", "http_status", "body_has_exec_code", "body_length", "body_special_char_count",
    "accepts_gzip", "accepts_deflate", "accepts_br", "accepts_identity", "accepts_zstd",
    "accepts_wildcard_encoding", "accepts_json", "accepts_html", "accepts_xml", "accepts_image",
    "accepts_wildcard_mime", "encoding_rare_flag", "has_accept_encoding", "has_accept_header",
    "ua_is_browser", "ua_is_cli_tool", "ua_is_python_lib", "ua_is_scanner_bot", "ua_is_custom_client",
    "ua_is_missing", "ua_is_other", "conn_keep_alive", "conn_close", "conn_upgrade",
    "has_connection_header", "connection_field_length"
] + [f"uri_emb_{i}" for i in range(50)] + [f"body_emb_{i}" for i in range(50)] + [
    "host_as_org_encoded", "host_asn_encoded", "host_country_encoded",
    "duration", "hour", "weekday", "is_weekend", "log_duration", "short_session"
]

# -------------------- FEATURE ENGINEERING --------------------
def parse_embedding(col):
    if isinstance(col, str):
        return np.fromstring(col.strip("[]"), sep=" ")
    return np.zeros(50)

def feature_engineering(df):
    df = df.copy()
    if "uri_embedding" in df.columns:
        df["uri_embedding"] = df["uri_embedding"].apply(lambda x: x if isinstance(x, str) else "0 " * 50)
        uri_embed = df["uri_embedding"].apply(parse_embedding).tolist()
        uri_embed_df = pd.DataFrame(uri_embed, columns=[f"uri_emb_{i}" for i in range(50)], index=df.index)
        df = pd.concat([df, uri_embed_df], axis=1)
    if "body_embedding" in df.columns:
        df["body_embedding"] = df["body_embedding"].apply(lambda x: x if isinstance(x, str) else "0 " * 50)
        body_embed = df["body_embedding"].apply(parse_embedding).tolist()
        body_embed_df = pd.DataFrame(body_embed, columns=[f"body_emb_{i}" for i in range(50)], index=df.index)
        df = pd.concat([df, body_embed_df], axis=1)
    drop_cols = [
        "category", "id", "http_uri", "http_body", "body_encoding", "uri_embedding", "body_embedding"
    ]
    X_clean = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    X_clean = X_clean.replace(['EMPTY', 'PAD', 'UNKNOWN'], np.nan)
    X_clean = X_clean.infer_objects()
    X_clean = X_clean.apply(pd.to_numeric, errors='coerce')
    X_clean = X_clean.fillna(0)
    for col in FEATURE_LIST:
        if col not in X_clean.columns:
            X_clean[col] = 0
    X_clean = X_clean[FEATURE_LIST]
    return X_clean

# -------------------- DATA LOADER --------------------
def stream_chronological_files(sequence_dir):
    for day in sorted(os.listdir(sequence_dir)):
        day_path = os.path.join(sequence_dir, day)
        if not os.path.isdir(day_path):
            continue
        for fname in sorted(os.listdir(day_path)):
            if fname.endswith("_encoded.csv"):
                yield os.path.join(day_path, fname)

def filter_valid_categories(df, valid_categories=VALID_CATEGORIES):
    return df[df["category"].astype(str).isin(valid_categories)]

# -------------------- TRAINING FUNCTIONS --------------------
def train_rf(X, y):
    clf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    clf.fit(X, y)
    return clf, None

def train_mlp(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=30, random_state=42)
    clf.fit(X_scaled, y)
    return clf, scaler

def train_nb(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = GaussianNB()
    clf.fit(X_scaled, y)
    return clf, scaler
'''
def train_cnn(X, y):
    X = np.array(X)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    lb = LabelBinarizer()
    y_cat = lb.fit_transform(y)
    model = Sequential([
        Conv1D(32, 3, activation='relu', input_shape=(X.shape[1], 1)),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(len(lb.classes_), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X, y_cat, epochs=50, batch_size=128, verbose=0)
    return (model, lb)

def train_lstm(X, y):
    X = np.array(X)
    X = X.reshape((X.shape[0], 1, X.shape[1]))
    lb = LabelBinarizer()
    y_cat = lb.fit_transform(y)
    model = Sequential([
        LSTM(64, input_shape=(1, X.shape[2])),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(len(lb.classes_), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X, y_cat, epochs=50, batch_size=128, verbose=0)
    return (model, lb)
'''

# -------------------- MODEL EVALUATION --------------------
def evaluate_model(model, aux, df, model_type="rf", logger=None):
    df = filter_valid_categories(df)
    X = feature_engineering(df)
    y = df["category"].astype(str)
    if len(X) == 0:
        if logger: logger.info("[EVAL] No data to evaluate!")
        return [np.nan]*4
    try:
        if model_type in ("cnn", "lstm"):
            model_keras = model
            lb = aux
            if model_type == "cnn":
                X_test = np.array(X).reshape((X.shape[0], X.shape[1], 1))
            else:
                X_test = np.array(X).reshape((X.shape[0], 1, X.shape[1]))
            y_true = lb.transform(y)
            y_pred_prob = model_keras.predict(X_test, verbose=0)
            y_pred = lb.inverse_transform(y_pred_prob)
        else:
            scaler = aux
            if scaler is not None:
                X = scaler.transform(X)
            y_pred = model.predict(X)
        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred, average='macro', zero_division=0)
        rec = recall_score(y, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y, y_pred, average='macro', zero_division=0)
        if logger: logger.info(f"Evaluation: acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} f1={f1:.4f}")
        return acc, prec, rec, f1
    except Exception as e:
        if logger: logger.error(f"Model evaluation error: {e}")
        return [np.nan]*4

# -------------------- DRIFT/OUTLIER UTILS --------------------
def max_consecutive_bursts(burst_window):
    max_run, cur_run = 0, 0
    for b in burst_window:
        if b:
            cur_run += 1
            max_run = max(max_run, cur_run)
        else:
            cur_run = 0
    return max_run



# ========== Drift Calculation ==========

def compute_feature_mean(df, feature_list):
    """Return feature mean vector of DataFrame for given features."""
    return df[feature_list].mean().values

def compute_data_drift(current_mean, baseline_mean):
    """Return L2 norm between current and baseline mean vector."""
    return np.linalg.norm(current_mean - baseline_mean)

# ========== Unknown Rate Calculation ==========

def tokenize(text):
    return re.findall(r"\b\w+\b", str(text).lower())

def extract_keywords(path, sensitive_words):
    tokens = re.split(r"[/._\-]", str(path).lower())
    matched_keywords = {token for token in tokens if token in sensitive_words}
    if not matched_keywords:
        return "<no-matching>"
    return ",".join(sorted(matched_keywords))

def block_sequences(df, sensitive_set):
    """Extract semantic sequences for a block."""
    seqs = set()
    for row in df.itertuples(index=False):
        try:
            method = getattr(row, "http_method")
            status = getattr(row, "http_status")
            uri = getattr(row, "http_uri")
        except AttributeError:
            continue
        sens_key = extract_keywords(uri, sensitive_set)
        seq = (method, status, sens_key)
        seqs.add(seq)
    return seqs

def compute_unknown_rate(seqs_this_block, model_sequences):
    if not seqs_this_block:
        return 0.0, 0, 0
    unknown = [seq for seq in seqs_this_block if seq not in model_sequences]
    return len(unknown) / (len(seqs_this_block) + 1e-8), len(seqs_this_block), len(unknown)
