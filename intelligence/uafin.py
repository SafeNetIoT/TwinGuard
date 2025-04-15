# User-Agent Based Fingerprinting per Attack Category

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

# Resolve user-agent group
ua_categories = [
    "browser", "cli_tool", "python_lib", "scanner_bot",
    "custom_client", "missing", "other"
]

def resolve_ua(row):
    for key in ua_categories:
        if row.get(f"ua_is_{key}", 0) == 1:
            return key
    return "unknown"

df["ua_group"] = df.apply(resolve_ua, axis=1)

# Remove rows where ua_group is 'unknown'
df = df[df["ua_group"] != "unknown"]

# Convert embedding strings to vectors
df["uri_embedding"] = df["uri_embedding"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))

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

# Compute fingerprint per (ua_group, category_label) pair
fingerprint_df = df.groupby(["ua_group", "category_label"])[fp_features].mean().reset_index()

# Save to CSV
fingerprint_df.to_csv("../data/fingerprinting/user_agent_fingerprints_by_category.csv", index=False)
print("UA-based fingerprints by attack type saved to CSV.")

# Check for non-numeric values in fp_features
non_numeric_cols = df[fp_features].select_dtypes(exclude=[np.number]).columns
print("Non-numeric columns:", non_numeric_cols)

# Ensure all fp_features columns are numeric
for col in fp_features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Handle missing values
df[fp_features] = df[fp_features].fillna(0)

scaler = MinMaxScaler()
df[fp_features] = scaler.fit_transform(df[fp_features])


'''
===============================================================================
Heatmap Visualization
===============================================================================
'''
'''
# Normalize feature columns for better comparison
feature_cols = [col for col in fingerprint_df.columns if col not in ["ua_group", "category_label"]]
df_norm = fingerprint_df.copy()
df_norm[feature_cols] = fingerprint_df[feature_cols].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

# Melt for heatmap plotting
df_melted = df_norm.melt(id_vars=["ua_group", "category_label"], var_name="Feature", value_name="Normalized Value")

# Check for non-numeric or missing values in the Normalized Value column
print(df_melted["Normalized Value"].dtype)
print(df_melted["Normalized Value"].isnull().sum())
print(df_melted[~df_melted["Normalized Value"].apply(np.isreal)])

# Convert Normalized Value to numeric and handle missing values
df_melted["Normalized Value"] = pd.to_numeric(df_melted["Normalized Value"], errors="coerce")
df_melted["Normalized Value"] = df_melted["Normalized Value"].fillna(0)

# Plot heatmap by category
plt.figure(figsize=(18, 10))
heatmap_data = df_melted.pivot_table(index=["ua_group", "category_label"], columns="Feature", values="Normalized Value")
sns.heatmap(heatmap_data, cmap="viridis", cbar_kws={'label': 'Normalized Value'})
plt.title("User-Agent Fingerprint Patterns by Attack Category")
plt.xlabel("HTTP Feature")
plt.ylabel("User-Agent Group + Category")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("fig/ua_fingerprint_heatmap_by_category.png")
'''

'''
===============================================================================
Histogram Visualization
===============================================================================
'''
'''
# Prepare plot
categories = ['attempt', 'intrusion-control', 'scan']
fig, axes = plt.subplots(len(categories), len(ua_categories), figsize=(38, 18), sharex=False, sharey=False)

for i, category in enumerate(categories):
    for j, ua in enumerate(ua_categories):
        ax = axes[i, j]
        subset = df[(df["category_label"] == category) & (df["ua_group"] == ua)]

        if subset.empty:
            ax.axis('off')
            continue

        all_feature_values = []
        for feature in fp_features:
            data = subset[feature].dropna()
            all_feature_values.extend(data.tolist())

        if len(all_feature_values) > 1:
            # Plot histogram
            ax.hist(all_feature_values, bins=61, color="skyblue", alpha=0.7, density=True, edgecolor="black", label="Histogram")

            # Plot KDE line
            try:
                kde = gaussian_kde(all_feature_values)
                x_grid = np.linspace(min(all_feature_values), max(all_feature_values), 200)
                ax.plot(x_grid, kde(x_grid), alpha=0.7, color="navy", linewidth=2.5, label="KDE")
            except np.linalg.LinAlgError as e:
                print(f"KDE Error for category '{category}' and UA '{ua}': {e}")
                ax.text(0.5, 0.5, 'KDE Error', ha='center', va='center')
                ax.set_xticks([])
                ax.set_yticks([])
        else:
            ax.axis('off')

        # Add legend
        ax.legend(fontsize=18)

        # ax.set_title(f"{category} - {ua}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

# Label y-axis with category names
for i, category in enumerate(categories):
    axes[i, 0].set_ylabel(category, fontsize=28, rotation=90)

# Label top x-axis with UA group names
for j, ua in enumerate(ua_categories):
    axes[0, j].set_title(ua, fontsize=28)


plt.suptitle("User-Agent Fingerprint Distributions by Category", fontsize=34, y=0.99)
plt.tight_layout()
plt.savefig("fig/ua_fingerprint_histogram_kde_combined_features.png", dpi=300)
'''

'''
===============================================================================
JS Divergence Calculation
===============================================================================
'''

# Normalize to probability distribution
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
    fps = fingerprint_df_normalized.set_index(['ua_group', 'category_label'])[fp_features]
    cat_fps = fps.xs(cat, level='category_label')

    # Ensure numeric and normalized data
    cat_fps = cat_fps.apply(pd.to_numeric, errors="coerce")
    cat_fps = cat_fps.div(cat_fps.sum(axis=1), axis=0).fillna(1e-10)

    pairs = list(itertools.combinations(cat_fps.index, 2))

    for (ua1, ua2) in pairs:
        divergence = jensenshannon(cat_fps.loc[ua1], cat_fps.loc[ua2], base=2)**2
        all_js_results.append({
            "UA_Group_Pair": f"{ua1} ↔ {ua2}",
            "JS_Divergence": divergence,
            "Category": cat
        })

js_df_all = pd.DataFrame(all_js_results)
js_df_all.to_csv("../data/fingerprinting/js_divergence_all_categories.csv", index=False)
print("JS Divergence results saved to CSV.")

# Combined visualization
custom_palette = {
    "attempt": "#DCF2F1",
    "intrusion-control": "#7FC7D9", 
    "scan": "#365486"  
}

plt.figure(figsize=(16, 10))
sns.barplot(
    data=js_df_all,
    x="JS_Divergence",
    y="UA_Group_Pair",
    hue="Category",
    palette=custom_palette
)

# Add dashed vertical lines for thresholds
plt.axvline(x=0.01, color="red", linestyle="--", linewidth=1.5, label="Small-Moderate (0.01)")
plt.axvline(x=0.03, color="blue", linestyle="--", linewidth=1.5, label="Moderate-Large (0.03)")


plt.title("JS Divergence Between User-Agent Groups", fontsize=24)
plt.xlabel("JS Divergence", fontsize=26)
plt.ylabel("User-Agent Group Pairs", fontsize=26)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.legend(title="Category and Thresholds", fontsize=20, title_fontsize=22, loc = 'upper right')
plt.tight_layout()
plt.savefig("fig/js_divergence_all_categories.png", dpi=300)