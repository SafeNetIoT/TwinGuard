import numpy as np
import pandas as pd

def feature_engineering(df):
    df = df.copy()

    # Fill or correct malformed embedding fields
    df["uri_embedding"] = df["uri_embedding"].apply(lambda x: x if isinstance(x, str) else "0 " * 50)
    df["body_embedding"] = df["body_embedding"].apply(lambda x: x if isinstance(x, str) else "0 " * 50)

    for col in ["uri_embedding", "body_embedding"]:
        df[col] = df[col].apply(lambda x: np.fromstring(x.strip("[]"), sep=" ") if isinstance(x, str) else np.zeros(50))

    uri_embed_df = pd.DataFrame(df["uri_embedding"].tolist(), index=df.index, columns=[f"uri_emb_{i}" for i in range(50)])
    body_embed_df = pd.DataFrame(df["body_embedding"].tolist(), index=df.index, columns=[f"body_emb_{i}" for i in range(50)])

    df = pd.concat([df.drop(columns=["uri_embedding", "body_embedding"]), uri_embed_df, body_embed_df], axis=1)

    basic_attributes = [
        "http_method", "http_status", "body_has_exec_code", "body_length", "body_special_char_count",
        "accepts_gzip", "accepts_deflate", "accepts_br", "accepts_identity", "accepts_zstd",
        "accepts_wildcard_encoding", "accepts_json", "accepts_html", "accepts_xml", "accepts_image",
        "accepts_wildcard_mime", "encoding_rare_flag", "has_accept_encoding", "has_accept_header",
        "ua_is_browser", "ua_is_cli_tool", "ua_is_python_lib", "ua_is_scanner_bot", "ua_is_custom_client",
        "ua_is_missing", "ua_is_other", "conn_keep_alive", "conn_close", "conn_upgrade",
        "has_connection_header", "connection_field_length"
    ] + [f"uri_emb_{i}" for i in range(50)] + [f"body_emb_{i}" for i in range(50)]

    spatial_features = ["host_as_org_encoded", "host_asn_encoded", "host_country_encoded"]
    temporal_features = ["duration", "hour", "weekday", "is_weekend", "log_duration", "short_session"]

    selected_features = basic_attributes + spatial_features + temporal_features

    return df[selected_features]


def feature_engineering_xgb(df):
    df = df.copy()
    # Fill or correct malformed embedding fields
    df["uri_embedding"] = df["uri_embedding"].apply(lambda x: x if isinstance(x, str) else "0 " * 50)
    df["body_embedding"] = df["body_embedding"].apply(lambda x: x if isinstance(x, str) else "0 " * 50)


    # Parse embedding vectors
    for col in ["uri_embedding", "body_embedding"]:
        df[col] = df[col].apply(lambda x: np.fromstring(x.strip("[]"), sep=" ") if isinstance(x, str) else np.zeros(50))

    uri_embed_df = pd.DataFrame(df["uri_embedding"].tolist(), index=df.index, columns=[f"uri_emb_{i}" for i in range(50)])
    body_embed_df = pd.DataFrame(df["body_embedding"].tolist(), index=df.index, columns=[f"body_emb_{i}" for i in range(50)])
    df = pd.concat([df.drop(columns=["uri_embedding", "body_embedding"]), uri_embed_df, body_embed_df], axis=1)

    basic_attributes = [
        "http_method", "http_status", "body_has_exec_code", "body_length", "body_special_char_count",
        "accepts_gzip", "accepts_deflate", "accepts_br", "accepts_identity", "accepts_zstd",
        "accepts_wildcard_encoding", "accepts_json", "accepts_html", "accepts_xml", "accepts_image",
        "accepts_wildcard_mime", "encoding_rare_flag", "has_accept_encoding", "has_accept_header",
        "ua_is_browser", "ua_is_cli_tool", "ua_is_python_lib", "ua_is_scanner_bot", "ua_is_custom_client",
        "ua_is_missing", "ua_is_other", "conn_keep_alive", "conn_close", "conn_upgrade",
        "has_connection_header", "connection_field_length"
    ] + [f"uri_emb_{i}" for i in range(50)] + [f"body_emb_{i}" for i in range(50)]

    spatial_features = ["host_as_org_encoded", "host_asn_encoded", "host_country_encoded"]
    temporal_features = ["duration", "hour", "weekday", "is_weekend", "log_duration", "short_session"]
    selected_features = basic_attributes + spatial_features + temporal_features

    # Clean up types to ensure no object dtype remains
    df = df[selected_features].copy()

    # Convert all boolean-like columns to integers
    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)
        elif df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    return df
