import os
import json
import pandas as pd
from datetime import datetime
import re
import csv

# === Config ===
XPOT_JSON_DIR = "../data/xpot/jsonfile"
OUTPUT_CSV_DIR = "../data/xpot/daily"
os.makedirs(OUTPUT_CSV_DIR, exist_ok=True)

# === Heuristic Labeling Function ===
def classify_category(request_raw):
    body = request_raw.split("\n\n")[-1] if "\n\n" in request_raw else ""
    headers = request_raw

    payload_keywords = r"(sh|exe|shell|download|curl|wget|powershell|cmd|/bin/bash|/bin/sh)"
    payload_found = bool(re.search(payload_keywords, body, re.IGNORECASE)) or bool(re.search(payload_keywords, headers, re.IGNORECASE))

    scouting_keywords = ["wp-login.php", "phpmyadmin", "login", "admin", "backup", "config"]
    if any(k in request_raw for k in scouting_keywords) and not payload_found:
        return "attempt"
    if payload_found:
        return "intrusion-control"
    return "scan"

# === XPOT Parser ===
def parse_xpot_json_file(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)

    records = []
    for entry in data:
        req = entry.get("request", {})
        malware = entry.get("malware", {})

        request_raw = req.get("request_raw", "")
        category = classify_category(request_raw)

        location = req.get("honeypot_location", "")
        host_as_org = location.split("-")[0] if "-" in location else ""
        host_country = location.split("-")[1] if "-" in location else ""

        record = {
            "protocol": "http",
            "category": category,
            "id": "",
            "http_method": req.get("request_method", ""),
            "http_status": "200",
            "http_uri": request_raw.split(" ")[1] if " " in request_raw else "",
            "http_body": request_raw.split("\n\n")[-1] if "\n\n" in request_raw else "",
            "http_accept": request_raw.split("Accept: ")[1].split("\n")[0] if "Accept: " in request_raw else "",
            "http_accept_encoding": request_raw.split("Accept-Encoding: ")[1].split("\n")[0] if "Accept-Encoding: " in request_raw else "",
            "http_user_agent": request_raw.split("User-Agent: ")[1].split("\n")[0] if "User-Agent: " in request_raw else "",
            "http_content_type": request_raw.split("Content-Type: ")[1].split("\n")[0] if "Content-Type: " in request_raw else "",
            "http_referer": "",
            "http_connection": request_raw.split("Connection: ")[1].split("\n")[0] if "Connection: " in request_raw else "",
            "sessionTimeout": "",
            "sessionLength": "",
            "startTime": req.get("request_timestamp", ""),
            "endTime": malware.get("collection_timestamp", req.get("request_timestamp", "")),
            "clientIP": req.get("source_ip", ""),
            "client_asn": "",
            "client_as_org": "",
            "client_country": "",
            "hostIP": malware.get("ipaddress", ""),
            "host_asn": "",
            "host_as_org": host_as_org,
            "host_country": host_country,
        }

        ts = record["startTime"]
        try:
            ts_id = datetime.fromisoformat(ts.replace("Z", "")).strftime("%Y%m%d%H%M%S")
        except:
            ts_id = "unknown"
        record["id"] = f"xpot#{ts_id}"

        records.append(record)

    return records

# === Batch Convert ===
for fname in os.listdir(XPOT_JSON_DIR):
    if fname.endswith(".json") and fname.startswith("xpot_accesslog_"):
        date_str = fname.replace("xpot_accesslog_", "").replace(".json", "")
        print(f"  Converting {fname} → test_day_{date_str}.csv")

        records = parse_xpot_json_file(os.path.join(XPOT_JSON_DIR, fname))
        df = pd.DataFrame(records)
        df.to_csv(
            os.path.join(OUTPUT_CSV_DIR, f"test_day_{date_str}.csv"), 
            index=False,
            quoting=csv.QUOTE_ALL,
            escapechar='\\')
        print(f" Saved {len(df)} records")

print(" All XPOT logs converted.")
