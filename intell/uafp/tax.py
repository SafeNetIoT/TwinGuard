import os
import pandas as pd
import json
import re
from datetime import datetime

# === Config ===
DAILY_FOLDER = "../../data/daily"
TAXONOMY_JSON = "regtax.json"
DATE_RANGES = [
    ("2025-03-15", "2025-04-15"),
    ("2025-04-16", "2025-05-15"),
    ("2025-05-16", "2025-06-15"),
    ("2025-06-16", "2025-07-15")
]

# === Load Taxonomy ===
with open(TAXONOMY_JSON, "r") as f:
    taxonomy = json.load(f)

# === Classification Function ===
def classify_intrusion_row(row, taxonomy):
    hits = []
    for parent_cat, subcats in taxonomy.items():
        for subcat, config in subcats.items():
            for field in config.get("match_fields", []):
                content = str(row.get(field, "")).lower()
                for pattern in config.get("patterns", []):
                    try:
                        if re.search(pattern, content):
                            hits.append(f"{parent_cat} → {subcat}")
                            break
                    except re.error:
                        continue
    return list(set(hits)) if hits else None

# === UA Category Function ===
category_keywords = {
    "browser": ["mozilla", "chrome", "safari", "edge", "firefox", "opera"],
    "cli_tool": ["curl", "wget", "httpie"],
    "python_lib": ["python", "requests", "aiohttp", "urllib"],
    "scanner_bot": ["expanse", "paloalto", "scanner", "bot", "censys", "nmap", "zgrab", "modatscanner", "wpdetector", "internetmeasurement", "genomecrawler"],
    "custom_client": ["custom", "asynchttpclient", "fasthttp", "l9explore", "keydrop"]
}

def detect_ua_group(ua):
    ua = str(ua).lower()
    if ua.strip() in ["", "unknown", "nan"]:
        return ["missing"]
    matched = [cat for cat, keys in category_keywords.items() if any(k in ua for k in keys)]
    return matched if matched else ["other"]

# === Process Each Date Range ===
for start_str, end_str in DATE_RANGES:
    print(f"\n🔄 Processing {start_str} to {end_str}...")
    start_dt = datetime.strptime(start_str, "%Y-%m-%d")
    end_dt = datetime.strptime(end_str, "%Y-%m-%d")
    output_prefix = f"{start_str}_to_{end_str}"

    # Collect matching files
    all_dfs = []
    for fname in sorted(os.listdir(DAILY_FOLDER)):
        if not fname.endswith(".csv"):
            continue
        try:
            date_str = fname.split("_")[-1].split(".")[0]
            file_dt = datetime.strptime(date_str, "%Y-%m-%d")
        except Exception:
            continue
        if not (start_dt <= file_dt <= end_dt):
            continue

        df = pd.read_csv(os.path.join(DAILY_FOLDER, fname))
        df = df[df["category"].str.startswith("intrusion", na=False)]
        df = df[["http_uri", "http_body", "http_user_agent", "http_referer", "host_as_org", "id"]].copy()
        df["intrusion_categories"] = df.apply(lambda row: classify_intrusion_row(row, taxonomy), axis=1)
        all_dfs.append(df)

    if not all_dfs:
        print(f"⚠️ No data found for range {start_str} to {end_str}. Skipping.")
        continue

    df_all = pd.concat(all_dfs, ignore_index=True)

    # Explode & parse
    df_all["parsed_categories"] = df_all["intrusion_categories"].apply(lambda x: x if isinstance(x, list) else [])
    df_exploded = df_all.explode("parsed_categories").dropna(subset=["parsed_categories"])
    df_exploded[["Parent_Category", "Subcategory"]] = df_exploded["parsed_categories"].str.split("→", expand=True)
    df_exploded["Parent_Category"] = df_exploded["Parent_Category"].str.strip()
    df_exploded["Subcategory"] = df_exploded["Subcategory"].str.strip()

    # UA Group Assignment
    df_exploded["user_agent_categories"] = df_exploded["http_user_agent"].apply(detect_ua_group)
    df_exploded["ua_group"] = df_exploded["user_agent_categories"].apply(lambda x: x[0])

    # Fingerprint Summaries
    ua_counts = df_exploded.groupby(["ua_group", "Parent_Category", "Subcategory"]).size().reset_index(name="count")
    ua_counts["percentage"] = ua_counts.groupby("ua_group")["count"].transform(lambda x: x / x.sum())

    org_counts = df_exploded.groupby(["host_as_org", "Parent_Category", "Subcategory"]).size().reset_index(name="count")
    org_counts["percentage"] = org_counts.groupby("host_as_org")["count"].transform(lambda x: x / x.sum())

    # Save Output
    ua_outfile = f"../../data/fp/ua_group_taxonomy_distribution_{output_prefix}.csv"
    org_outfile = f"../../data/fp/cloud_org_taxonomy_distribution_{output_prefix}.csv"

    ua_counts.to_csv(ua_outfile, index=False)
    org_counts.to_csv(org_outfile, index=False)

    print(f"✅ Saved: {ua_outfile}")
    print(f"✅ Saved: {org_outfile}")
