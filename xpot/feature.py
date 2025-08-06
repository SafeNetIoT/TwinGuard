import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import re
import base64
import urllib.parse
import ast
import json
import gensim.downloader as api

# === Config ===
INPUT_FOLDER = "../data/xpot/sequences"
OUTPUT_FOLDER = "../data/xpot/sequences_encoded"
SENSITIVE_WORDS_PATH = "../data/base/sensitive_words_with_counts.txt"
LABEL_MAPPING_PATH = os.path.join(OUTPUT_FOLDER, "label_mapping.json")

START_DATE = "2025-05-01"
END_DATE   = "2025-05-30"

# === Load GloVe ===
glove = api.load("glove-wiki-gigaword-50")

# === Load sensitive words ===
with open(SENSITIVE_WORDS_PATH, "r") as f:
    sensitive_words = {line.split(":")[0] for line in f.readlines()}

# === Helper Functions ===
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
        return {"encoding": "EMPTY", "exec_code": 0, "tokens": ["EMPTY_BODY"], "length": 0, "special_chars": 0}
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

def process_file(input_path, output_path):
    print(f"Processing {input_path}")
    df = pd.read_csv(input_path)
    df["protocol"] = df["protocol"].map({"https": 1, "http": 0})

    CATEGORY_ORDER = ["attempt", "intrusion-control", "scan"]
    df["category"] = df["category"].str.strip().str.lower()
    df = df[df["category"].isin(CATEGORY_ORDER)]

    le_cat = LabelEncoder()
    le_cat.fit(CATEGORY_ORDER)
    df["category"] = le_cat.transform(df["category"])

    HTTP_METHODS = ["GET", "POST", "OPTIONS", "HEAD", "PRI", "PROPFIND", "PUT", "DELETE", "PATCH", "TRACE", "OTHER"]
    df["http_method"] = df["http_method"].str.upper()
    df["http_method"] = df["http_method"].apply(lambda m: m if m in HTTP_METHODS else "OTHER")
    le_method = LabelEncoder()
    le_method.fit(HTTP_METHODS)
    df["http_method"] = le_method.transform(df["http_method"])

    df["uri_tokens"] = df["http_uri"].apply(lambda x: process_text(str(x)))
    df_body = df["http_body"].apply(lambda x: process_body(str(x))).apply(pd.Series)
    df = pd.concat([df, df_body], axis=1)

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

    df["user_agent_categories"] = df["http_user_agent"].apply(detect_ua_categories)
    for cat in list(ua_keywords.keys()) + ["missing", "other"]:
        df[f"ua_is_{cat}"] = df["user_agent_categories"].apply(lambda x: int(cat in x))

    df["http_connection"] = df["http_connection"].fillna("UNKNOWN")
    df["http_connection"] = df["http_connection"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") else x.split(",") if isinstance(x, str) else [x])
    df["http_connection"] = df["http_connection"].apply(lambda x: [t.strip().lower() for t in x if isinstance(t, str)])

    for conn in ["keep-alive", "close", "upgrade"]:
        df[f"conn_{conn.replace('-', '_')}"] = df["http_connection"].apply(lambda x: int(conn in x))

    df["has_connection_header"] = df["http_connection"].apply(lambda x: 0 if x == ["unknown"] else 1)
    df["connection_field_length"] = df["http_connection"].apply(lambda x: len(x))

    df["uri_embedding"] = df["uri_tokens"].apply(embed_tokens)
    df["body_embedding"] = df["body_tokens"].apply(embed_tokens)

    df["host_as_org_encoded"] = LabelEncoder().fit_transform(df["host_as_org"].astype(str))
    df["host_asn_encoded"] = LabelEncoder().fit_transform(df["host_asn"].astype(str))
    df["host_country_encoded"] = LabelEncoder().fit_transform(df["host_country"].astype(str))

    df["startTime"] = pd.to_datetime(df["startTime"], errors="coerce")
    df["endTime"] = pd.to_datetime(df["endTime"], errors="coerce")
    df["duration"] = (df["endTime"] - df["startTime"]).dt.total_seconds()
    df["hour"] = df["startTime"].dt.hour
    df["weekday"] = df["startTime"].dt.weekday
    df["is_weekend"] = df["weekday"].apply(lambda x: 1 if x >= 5 else 0)
    df["log_duration"] = np.log1p(df["duration"])
    df["short_session"] = df["duration"].apply(lambda x: 1 if x <= 5 else 0)

    drop_cols = ["protocol", "http_accept", "http_accept_encoding", "http_user_agent", "http_content_type",
                 "http_referer", "http_connection", "startTime", "endTime", "sessionTimeout", "sessionLength",
                 "clientIP", "hostIP", "client_asn", "client_as_org", "client_country",
                 "host_asn", "host_as_org", "host_country", "user_agent_categories",
                 "uri_tokens", "body_tokens"]
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)
    df.to_csv(output_path, index=False)

# === Apply to all sequence blocks ===
for day_folder in sorted(os.listdir(INPUT_FOLDER)):
    if not (START_DATE <= day_folder <= END_DATE):
        continue
    input_day_path = os.path.join(INPUT_FOLDER, day_folder)
    output_day_path = os.path.join(OUTPUT_FOLDER, day_folder)
    if not os.path.isdir(input_day_path):
        continue
    os.makedirs(output_day_path, exist_ok=True)

    for file in sorted(os.listdir(input_day_path)):
        if file.endswith(".csv"):
            input_path = os.path.join(input_day_path, file)
            output_path = os.path.join(output_day_path, file.replace(".csv", "_encoded.csv"))
            try:
                process_file(input_path, output_path)
            except Exception as e:
                print(f"Error processing {input_path}: {e}")
