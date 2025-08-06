import os
import numpy as np
import pandas as pd
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import pandas as pd

def evaluate_model(model, aux, df, model_type="rf", logger=None, return_fp=False):
    df = filter_valid_categories(df)
    X = feature_engineering(df)
    y = df["category"].astype(int)   # ALWAYS use int
    if len(X) == 0:
        if logger: logger.info("[EVAL] No data to evaluate!")
        return [np.nan]*4
    try:
        if model_type in ("cnn", "lstm"):
            # ... your keras code here ...
            pass
        else:
            scaler = aux
            if scaler is not None:
                X = scaler.transform(X)
            y_pred = model.predict(X)
            y_pred = pd.Series(y_pred).astype(int)  # Cast predictions to int

        # Logging types
        if logger:
            logger.info(f"GT y dtype: {y.dtype}, unique: {pd.unique(y)}")
            logger.info(f"Pred y_pred dtype: {pd.Series(y_pred).dtype}, unique: {pd.unique(y_pred)}")

        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred, average='macro', zero_division=0)
        rec = recall_score(y, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y, y_pred, average='macro', zero_division=0)
        if logger: logger.info(f"Evaluation: acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} f1={f1:.4f}")

        # For confusion matrix and FP calculation
        if return_fp:
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y, y_pred, labels=[0, 1, 2])
            fp_dict = {}
            for i in range(3):
                fp = cm[:, i].sum() - cm[i, i]
                fp_dict[i] = fp
            return (acc, prec, rec, f1), fp_dict

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

'''
Feature engineering for the initial data
'''

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import re
import base64
import urllib.parse
import ast

# ---- These will be set by the main script ----
glove = None
sensitive_words = None

def set_embedding_resources(glove_vectors, sensitive_word_set):
    global glove, sensitive_words
    glove = glove_vectors
    sensitive_words = sensitive_word_set

# --- Helper Functions (unchanged) ---
def tokenize(text):
    return re.findall(r"\b\w+\b", str(text).lower())

def process_text(text, max_tokens=5):
    tokens = tokenize(text)
    if len(tokens) <= max_tokens:
        return tokens + ["PAD"] * (max_tokens - len(tokens))
    filtered = [t for t in tokens if t in sensitive_words]
    if len(filtered) >= max_tokens:
        return filtered[:max_tokens]
    return filtered + tokens[:max_tokens - len(filtered)]

def is_base64(s):
    try:
        s_bytes = s.encode("utf-8") if isinstance(s, str) else s
        return base64.b64encode(base64.b64decode(s_bytes)).strip() == s_bytes.strip()
    except Exception:
        return False

def is_url_encoded(s):
    return "%" in s and bool(re.search(r"%[0-9A-Fa-f]{2}", s))

def is_hex_encoded(s):
    return bool(re.fullmatch(r"(\\x[0-9A-Fa-f]{2})+", s))

def decode_url(s):
    try:
        return urllib.parse.unquote(s)
    except:
        return s

def count_special_chars(s):
    return len(re.findall(r"[{}();|&=#]", s))

def process_body(text):
    if not isinstance(text, str) or text.strip() == "":
        return {"body_encoding": "EMPTY", "body_has_exec_code": 0, "body_tokens": ["EMPTY_BODY"], "body_length": 0, "body_special_char_count": 0}
    decoded = text.replace("\n", " ")
    enc_type = "UTF-8"
    if is_base64(text): enc_type = "Base64"
    elif is_url_encoded(text):
        enc_type = "URL"
        decoded = decode_url(text)
    elif is_hex_encoded(text): enc_type = "Hex"
    exec_flag = int(bool(re.search(r"(curl|wget|sh|eval|shell_exec|base64_decode|nc|bash|python|perl|php|powershell)", decoded)))
    return {
        "body_encoding": enc_type,
        "body_has_exec_code": exec_flag,
        "body_tokens": process_text(decoded),
        "body_length": len(decoded),
        "body_special_char_count": count_special_chars(decoded)
    }

def embed_tokens(tokens):
    if not isinstance(tokens, list): tokens = []
    vectors = [glove[token] if token in glove else np.zeros(glove.vector_size) for token in tokens]
    return np.mean(vectors, axis=0) if vectors else np.zeros(glove.vector_size)

ua_keywords = {
    "browser": ["mozilla", "chrome", "safari", "edge", "firefox", "opera"],
    "cli_tool": ["curl", "wget", "httpie"],
    "python_lib": ["python", "requests", "aiohttp", "urllib"],
    "scanner_bot": ["expanse", "paloalto", "scanner", "bot", "censys", "nmap", "zgrab", "modatscanner", "wpdetector", "internetmeasurement", "genomecrawler"],
    "custom_client": ["custom", "asynchttpclient", "fasthttp", "l9explore", "keydrop"]
}

def detect_ua_categories(ua):
    ua = str(ua).lower()
    if ua.strip() in ["", "unknown", "nan"]:
        return ["missing"]
    matched = [cat for cat, keys in ua_keywords.items() if any(k in ua for k in keys)]
    return matched if matched else ["other"]

def feature_engineer_df(df):
    """Matches the *original* training feature engineering pipeline EXACTLY."""
    df = df.copy()
    if "protocol" in df.columns:
        df["protocol"] = df["protocol"].map({"https": 1, "http": 0})

    CATEGORY_ORDER = ["attempt", "intrusion-control", "scan"]
    df["category"] = df["category"].astype(str).str.strip().str.lower()
    df = df[df["category"].isin(CATEGORY_ORDER)]
    le_cat = LabelEncoder()
    le_cat.fit(CATEGORY_ORDER)
    df["category"] = le_cat.transform(df["category"])

    HTTP_METHODS = ["GET", "POST", "OPTIONS", "HEAD", "PRI", "PROPFIND", "PUT", "DELETE", "PATCH", "TRACE", "OTHER"]
    df["http_method"] = df["http_method"].astype(str).str.upper()
    df["http_method"] = df["http_method"].apply(lambda m: m if m in HTTP_METHODS else "OTHER")
    le_method = LabelEncoder()
    le_method.fit(HTTP_METHODS)
    df["http_method"] = le_method.transform(df["http_method"])

    # URI tokens/embedding
    df["uri_tokens"] = df["http_uri"].apply(lambda x: process_text(str(x)))
    df["uri_embedding"] = df["uri_tokens"].apply(embed_tokens)

    # Body features + embedding
    df_body = df["http_body"].apply(lambda x: process_body(str(x))).apply(pd.Series)
    df = pd.concat([df.reset_index(drop=True), df_body.reset_index(drop=True)], axis=1)
    df = df[df["body_tokens"].apply(lambda x: isinstance(x, list))]
    df["body_tokens"] = df["body_tokens"].apply(lambda x: x if len(x) == 5 else ["PAD"]*5)
    df["body_embedding"] = df["body_tokens"].apply(embed_tokens)

    # Accept headers
    df["http_accept"] = df["http_accept"].fillna("UNKNOWN")
    df["http_accept_encoding"] = df["http_accept_encoding"].fillna("UNKNOWN")
    df["http_accept"] = df["http_accept"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") else x.split(",") if isinstance(x, str) else [x])
    df["http_accept_encoding"] = df["http_accept_encoding"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") else x.split(",") if isinstance(x, str) else [x])

    encoding_types = ["gzip", "deflate", "br", "identity", "zstd", "*"]
    mime_keywords = ["json", "html", "xml", "image", "*"]
    for enc in encoding_types:
        col = f"accepts_{enc.replace('*', 'wildcard_encoding')}"
        df[col] = df["http_accept_encoding"].apply(lambda x: int(any(enc in e.strip().lower() for e in x)))
    for mime in mime_keywords:
        col = f"accepts_{mime if mime != '*' else 'wildcard_mime'}"
        df[col] = df["http_accept"].apply(lambda x: int(any(mime in e.lower() for e in x)))
    df["encoding_rare_flag"] = df["http_accept_encoding"].apply(lambda x: int(not any(enc.strip().lower() in encoding_types for enc in x)) if x != ["UNKNOWN"] else 0)
    df["has_accept_encoding"] = df["http_accept_encoding"].apply(lambda x: 0 if x == ["UNKNOWN"] else 1)
    df["has_accept_header"] = df["http_accept"].apply(lambda x: 0 if x == ["UNKNOWN"] else 1)

    # UA features
    df["user_agent_categories"] = df["http_user_agent"].apply(detect_ua_categories)
    for cat in list(ua_keywords.keys()) + ["missing", "other"]:
        df[f"ua_is_{cat}"] = df["user_agent_categories"].apply(lambda x: int(cat in x))

    # Connection
    df["http_connection"] = df["http_connection"].fillna("UNKNOWN")
    df["http_connection"] = df["http_connection"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") else x.split(",") if isinstance(x, str) else [x])
    df["http_connection"] = df["http_connection"].apply(lambda x: [t.strip().lower() for t in x if isinstance(t, str)])
    for conn in ["keep-alive", "close", "upgrade"]:
        df[f"conn_{conn.replace('-', '_')}"] = df["http_connection"].apply(lambda x: int(conn in x))
    df["has_connection_header"] = df["http_connection"].apply(lambda x: 0 if x == ["unknown"] else 1)
    df["connection_field_length"] = df["http_connection"].apply(lambda x: len(x))

    # ASN/org encoding
    df["host_as_org_encoded"] = LabelEncoder().fit_transform(df["host_as_org"].astype(str))
    df["host_asn_encoded"] = LabelEncoder().fit_transform(df["host_asn"].astype(str))
    df["host_country_encoded"] = LabelEncoder().fit_transform(df["host_country"].astype(str))

    # Time features
    df["startTime"] = pd.to_datetime(df["startTime"], errors="coerce")
    df["endTime"] = pd.to_datetime(df["endTime"], errors="coerce")
    df["duration"] = (df["endTime"] - df["startTime"]).dt.total_seconds()
    df["hour"] = df["startTime"].dt.hour
    df["weekday"] = df["startTime"].dt.weekday
    df["is_weekend"] = df["weekday"].apply(lambda x: 1 if x >= 5 else 0)
    df["log_duration"] = np.log1p(df["duration"])
    df["short_session"] = df["duration"].apply(lambda x: 1 if x <= 5 else 0)

    # Drop only unneeded columns (keep block_idx if present)
    drop_cols = ["protocol", "http_accept", "http_accept_encoding", "http_user_agent", "http_content_type",
                 "http_referer", "http_connection", "startTime", "endTime", "sessionTimeout", "sessionLength",
                 "clientIP", "hostIP", "client_asn", "client_as_org", "client_country",
                 "host_asn", "host_as_org", "host_country", "user_agent_categories",
                 "uri_tokens", "body_tokens"]
    drop_cols = [col for col in drop_cols if col in df.columns and col != "block_idx"]
    df.drop(columns=drop_cols, inplace=True, errors='ignore')
    return df
