import pandas as pd
import json
import re
from collections import defaultdict, Counter

# === Load CSV Data ===
df = pd.read_csv("../data/raw/http_2025-03-01_to_2025-03-30.csv")

# === Filter for Intrusion Category Only ===
df = df[df["category"].str.startswith("intrusion", na=False)]

# === Keep Only Relevant Columns for Matching ===
columns_to_keep = ["http_uri", "http_body", "http_user_agent", "http_referer", "host_as_org", "id"]
df = df[columns_to_keep].copy()

# === Load Hierarchical Taxonomy JSON ===
with open("regtax.json", "r") as f:
    taxonomy = json.load(f)

# === Classification Function (with hierarchy support) ===
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
                            break  # Stop after first match in subcategory
                    except re.error as e:
                        print(f"⚠️ Invalid regex in '{parent_cat} → {subcat}': {pattern} ({e})")
    return list(set(hits)) if hits else None

# === Apply Classification ===
df["intrusion_categories"] = df.apply(
    lambda row: classify_intrusion_row(row, taxonomy),
    axis=1
)

# === Save Full Classified Log ===
df.to_csv("../data/taxonomy/classified_intrusion_log.csv", index=False)

# === Summary of Intrusion Categories ===
all_matches = df["intrusion_categories"].dropna().explode()
category_counts = Counter(all_matches)
summary_df = pd.DataFrame(category_counts.items(), columns=["Category", "Count"]).sort_values(by="Count", ascending=False)
summary_df.to_csv("../data/taxonomy/intrusion_category_summary.csv", index=False)

print("Intrusion Category Summary:")
print(summary_df)

# === Top 5 Example URIs/Bodies per Category ===
category_examples = defaultdict(list)
for _, row in df.iterrows():
    categories = row.get("intrusion_categories", [])
    if not categories:
        continue
    for cat in categories:
        ref = row.get("http_uri") or row.get("http_body")
        if ref:
            category_examples[cat].append(ref.strip())

top5_examples = []
for cat, samples in category_examples.items():
    top5 = Counter(samples).most_common(5)
    for val, count in top5:
        top5_examples.append({"Category": cat, "Example": val, "Count": count})

top5_df = pd.DataFrame(top5_examples)
top5_df.to_csv("../data/taxonomy/top5_attack_signatures.csv", index=False)

print(" Top 5 attack signatures per category saved.")
print(top5_df.head(10))
