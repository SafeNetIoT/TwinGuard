# Cloud Provider Based Fingerprinting per Attack Category (Histogram + KDE-Based)

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import jensenshannon
import itertools

# Load the processed dataset
file_path = "../data/encoded/http_2025-03-15_to_2025-04-09_encoded.csv"
df = pd.read_csv(file_path)

# Map category integer to label
label_map = {0: 'attempt', 1: 'intrusion-control', 2: 'scan'}
df["category_label"] = df["category"].map(label_map)

'''
# Map host_as_org_encoded to known cloud orgs
org_label_map = {
    0: "AMAZON-02",
    1: "AMAZON-AES",
    2: "GOOGLE",
    3: "DIGITALOCEAN-ASN"
}
df["cloud_org"] = df["host_as_org_encoded"].map(org_label_map)

# Define the target cloud org categories
cloud_org_categories = ["AMAZON-02", "AMAZON-AES", "GOOGLE", "DIGITALOCEAN-ASN"]
'''

# Map host_as_org_encoded to anonymized cloud orgs
org_label_map = {
    0: "Cloud-A",
    1: "Cloud-B",
    2: "Cloud-C",
    3: "Cloud-D"
}
df["cloud_org"] = df["host_as_org_encoded"].map(org_label_map)

# Define the anonymized cloud org categories
cloud_org_categories = ["Cloud-A", "Cloud-B", "Cloud-C", "Cloud-D"]

# Convert embedding strings to vectors
df["uri_embedding"] = df["uri_embedding"].apply(
    lambda x: np.fromstring(x.strip("[]"), sep=" ") if isinstance(x, str) else np.zeros(50)
)

# Expand URI embeddings into columns
uri_df = pd.DataFrame(df["uri_embedding"].tolist(), index=df.index)
uri_df.columns = [f"uri_emb_{i}" for i in range(uri_df.shape[1])]
df = pd.concat([df, uri_df], axis=1)

# Select features to include in the fingerprint
fp_features = [
    "http_method", "http_status", "body_has_exec_code", "body_length",
    "accepts_gzip", "accepts_deflate", "accepts_br", "accepts_json",
    "accepts_html", "accepts_xml", "accepts_image", "accepts_wildcard_mime",
    "has_accept_encoding", "conn_keep_alive", "conn_close", "conn_upgrade",
    "has_connection_header", "connection_field_length"
] + list(uri_df.columns)

# Ensure all fp_features columns are numeric
for col in fp_features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Handle missing values
df[fp_features] = df[fp_features].fillna(0)

# Compute fingerprint per (cloud_org, category_label) pair
fingerprint_df = df.groupby(["cloud_org", "category_label"])[fp_features].mean().reset_index()

# Save to CSV
fingerprint_df.to_csv("../data/fingerprinting/cloud_org_fingerprints_by_category.csv", index=False)
print("Cloud-org-based fingerprints by attack type saved to CSV.")

# Normalize features
scaler = MinMaxScaler()
df[fp_features] = scaler.fit_transform(df[fp_features])

'''
===============================================================================
Histogram Visualization
===============================================================================
'''
# Prepare plot
categories = ['attempt', 'intrusion-control', 'scan']
fig, axes = plt.subplots(len(categories), len(cloud_org_categories), figsize=(40, 16), sharex=False, sharey=False)

for i, category in enumerate(categories):
    for j, org in enumerate(cloud_org_categories):
        ax = axes[i, j]
        subset = df[(df["category_label"] == category) & (df["cloud_org"] == org)]

        if subset.empty:
            ax.axis('off')
            continue

        all_feature_values = []
        for feature in fp_features:
            data = subset[feature].dropna()
            all_feature_values.extend(data.tolist())

        if len(all_feature_values) > 1:
            ax.hist(all_feature_values, bins=61, color="skyblue", alpha=0.7, density=True, edgecolor="black", label="Histogram")

            try:
                kde = gaussian_kde(all_feature_values)
                x_grid = np.linspace(min(all_feature_values), max(all_feature_values), 200)
                ax.plot(x_grid, kde(x_grid), alpha=0.7, color="navy", linewidth=2.5, label="KDE")
            except np.linalg.LinAlgError as e:
                print(f"KDE Error for category '{category}' and Org '{org}': {e}")
                ax.text(0.5, 0.5, 'KDE Error', ha='center', va='center')
                ax.set_xticks([])
                ax.set_yticks([])
        else:
            ax.axis('off')

        if i == 0 and j == 0:
            handles, labels = ax.get_legend_handles_labels()

        ax.set_xticks([])
        ax.set_yticks([])

# Y-axis labels (attack categories)
for i, category in enumerate(categories):
    axes[i, 0].set_ylabel(category, fontsize=40, rotation=90)

# X-axis titles (cloud orgs)
for j, org in enumerate(cloud_org_categories):
    axes[0, j].set_title(org, fontsize=40)

plt.suptitle("Cloud Provider Fingerprint Distributions by Category", fontsize=46, y=0.99)
plt.tight_layout(rect=[0, 0, 0.98, 0.97])

fig.legend(
    handles, labels,
    loc='upper right',
    bbox_to_anchor=(0.985, 1.015),
    fontsize=30,
    frameon=False
)

plt.savefig("fig/cloudfin.png", dpi = 300)
print("User-Agent fingerprint processing complete.")


'''
===============================================================================
JS Divergence Calculation
===============================================================================
'''     
# Normalize feature vectors to probability distributions
def normalize_to_prob_dist(vector):
    vector = np.array(vector, dtype=np.float64)  # Ensure numeric array
    vector -= np.min(vector)  # Ensure non-negative values
    if np.sum(vector) == 0:
        vector += 1e-10  # Avoid division by zero
    return vector / np.sum(vector)

fingerprint_df_normalized = fingerprint_df.copy()
for idx in fingerprint_df_normalized.index:
    original_vector = fingerprint_df_normalized.loc[idx, fp_features].values
    fingerprint_df_normalized.loc[idx, fp_features] = normalize_to_prob_dist(original_vector)

# Calculate JS Divergence for all categories
categories = ['attempt', 'intrusion-control', 'scan']
all_js_results = []

for cat in categories:
    fps = fingerprint_df_normalized.set_index(['cloud_org', 'category_label'])[fp_features]
    cat_fps = fps.xs(cat, level='category_label')

    # Ensure numeric and normalized data
    cat_fps = cat_fps.apply(pd.to_numeric, errors="coerce")
    cat_fps = cat_fps.div(cat_fps.sum(axis=1), axis=0).fillna(1e-10)

    pairs = list(itertools.combinations(cat_fps.index, 2))

    for (org1, org2) in pairs:
        divergence = jensenshannon(cat_fps.loc[org1], cat_fps.loc[org2], base=2)**2
        all_js_results.append({
            "Cloud_Org_Pair": f"{org1} ↔ {org2}",
            "JS_Divergence": divergence,
            "Category": cat
        })

js_df_all = pd.DataFrame(all_js_results)
js_df_all.to_csv("../data/fingerprinting/js_divergence_all_categories_cloud.csv", index=False)
print("JS Divergence results saved to CSV.")

# Combined visualization
custom_palette = {
    "attempt": "#DCF2F1",
    "intrusion-control": "#7FC7D9", 
    "scan": "#365486"  
}

plt.figure(figsize=(16, 12))
sns.barplot(
    data=js_df_all,
    x="JS_Divergence",
    y="Cloud_Org_Pair",
    hue="Category",
    palette=custom_palette
)

# Axis & legend formatting
plt.axvline(x=0.01, color="red", linestyle="--", linewidth=1.5, label="Small-Moderate Threshold (0.01)")
plt.axvline(x=0.015, color="blue", linestyle="--", linewidth=1.5, label="Moderate-Large Threshold (0.015)")

plt.title("JS Divergence Between Cloud Organizations", fontsize=24)
plt.xlabel("JS Divergence", fontsize=22)
plt.ylabel("Cloud Org Pairs", fontsize=22)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(title="Category and Thresholds", loc='upper right', fontsize=20, title_fontsize=20)
plt.tight_layout()
plt.savefig("fig/js_divergence_all_categories_cloud.png", dpi = 300)
