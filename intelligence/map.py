import pandas as pd
import ast

# === Load Dataset ===
file_path = "../data/fingerprinting/classified_intrusion_log.csv"  # Update with your actual path
df = pd.read_csv(file_path)

# === Parse and Clean Intrusion Categories ===
df["parsed_categories"] = df["intrusion_categories"].apply(
    lambda x: ast.literal_eval(x) if pd.notnull(x) and x.strip() else []
)

# === Explode Multi-Label Column into Rows ===
df_exploded = df.explode("parsed_categories").dropna(subset=["parsed_categories"])

# === Split Taxonomy into Levels ===
df_exploded[["Parent_Category", "Subcategory"]] = df_exploded["parsed_categories"].str.split("→", expand=True)
df_exploded["Parent_Category"] = df_exploded["Parent_Category"].str.strip()
df_exploded["Subcategory"] = df_exploded["Subcategory"].str.strip()

# === UA Group Classification Logic ===
category_keywords = {
    "browser": ["mozilla", "chrome", "safari", "edge", "firefox", "opera"],
    "cli_tool": ["curl", "wget", "httpie"],
    "python_lib": ["python", "requests", "aiohttp", "urllib"],
    "scanner_bot": ["expanse", "paloalto", "scanner", "bot", "censys", "nmap", "zgrab", "modatscanner", "wpdetector", "internetmeasurement", "genomecrawler"],
    "custom_client": ["custom", "asynchttpclient", "fasthttp", "l9explore", "keydrop"]
}

def detect_categories(ua):
    ua = str(ua).lower()
    if ua.strip() in ["", "unknown", "nan"]:
        return ["missing"]
    matched = [cat for cat, keywords in category_keywords.items() if any(k in ua for k in keywords)]
    return matched if matched else ["other"]

df_exploded["user_agent_categories"] = df_exploded["http_user_agent"].apply(detect_categories)
df_exploded["ua_group"] = df_exploded["user_agent_categories"].apply(lambda x: x[0])  # Primary category

# === Save Preprocessed File (Optional) ===
df_exploded.to_csv("../data/fingerprinting/taxonomy_with_ua_cloud.csv", index=False)

# === Group & Count by UA Group and Subcategory ===
ua_counts = df_exploded.groupby(["ua_group", "Parent_Category", "Subcategory"]).size().reset_index(name="count")
ua_counts["percentage"] = ua_counts.groupby("ua_group")["count"].transform(lambda x: x / x.sum())

# === Group & Count by Cloud Org and Subcategory ===
org_counts = df_exploded.groupby(["host_as_org", "Parent_Category", "Subcategory"]).size().reset_index(name="count")
org_counts["percentage"] = org_counts.groupby("host_as_org")["count"].transform(lambda x: x / x.sum())

# === Save for Visualization or Analysis ===
ua_counts.to_csv("../data/fingerprinting/ua_group_taxonomy_distribution.csv", index=False)
org_counts.to_csv("../data/fingerprinting/cloud_org_taxonomy_distribution.csv", index=False)

print("Processed taxonomy counts by UA group and cloud org saved.")
