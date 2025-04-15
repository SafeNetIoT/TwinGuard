import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load pre-aggregated CSVs
ua_df = pd.read_csv("../data/fingerprinting/ua_group_taxonomy_distribution.csv")
org_df = pd.read_csv("../data/fingerprinting/cloud_org_taxonomy_distribution.csv")

'''
===============================================================================
1. Heatmap: UA Group vs. Subcategory
===============================================================================
'''
ua_pivot = ua_df.pivot_table(index="ua_group", columns="Subcategory", values="percentage", fill_value=0)

plt.figure(figsize=(12, 6))
sns.heatmap(ua_pivot, cmap="Blues", linewidths=0.3, linecolor='white')

plt.title("Attack Subcategory Distribution per User-Agent Group", fontsize=26, pad=20)
plt.xlabel("Subcategory", fontsize=22)
plt.ylabel("User-Agent Group", fontsize=22)
plt.xticks(rotation=45, ha="right", fontsize=18)
plt.yticks(rotation=0, fontsize=18)
plt.tight_layout()
plt.savefig("../figs/uataxheat.png", dpi=300, bbox_inches='tight')
plt.close()

'''
===============================================================================
2. Heatmap: Cloud Provider vs. Subcategory
===============================================================================
'''
org_label_map = {
    "AMAZON-02": "Cloud-A",
    "AMAZON-AES": "Cloud-B",
    "GOOGLE": "Cloud-C",
    "DIGITALOCEAN-ASN": "Cloud-D"
}
ordered_orgs = ["AMAZON-02", "AMAZON-AES", "GOOGLE", "DIGITALOCEAN-ASN"]

org_pivot = org_df.pivot_table(index="host_as_org", columns="Subcategory", values="percentage", fill_value=0)
org_pivot = org_pivot.loc[ordered_orgs]
org_pivot.index = [org_label_map.get(org, org) for org in org_pivot.index]

plt.figure(figsize=(12, 6))
sns.heatmap(org_pivot, cmap="Blues", linewidths=0.3, linecolor='white')

plt.title("Attack Subcategory Distribution per Cloud Provider", fontsize=26, pad=20)
plt.xlabel("Subcategory", fontsize=22)
plt.ylabel("Cloud Provider", fontsize=22)
plt.xticks(rotation=45, ha="right", fontsize=18)
plt.yticks(rotation=0, fontsize=18)
plt.tight_layout()
plt.savefig("../figs/cloudtaxheat.png", dpi=300, bbox_inches='tight')
plt.close()


'''
===============================================================================
3. Stacked Bar: UA Group vs Parent Category
===============================================================================
'''

# Group and pivot
parent_pivot = ua_df.groupby(["ua_group", "Parent_Category"])["percentage"].sum().reset_index()
parent_plot = parent_pivot.pivot(index="ua_group", columns="Parent_Category", values="percentage").fillna(0)

# Sort by total
parent_plot = parent_plot.loc[parent_plot.sum(axis=1).sort_values(ascending=False).index]

# Plot
fig, ax = plt.subplots(figsize=(16, 8))  # slightly larger
parent_plot.plot(
    kind="bar",
    stacked=True,
    ax=ax,
    colormap="RdYlBu"
)

# Titles and labels with larger fonts
ax.set_title("Attack Distribution by Parent Category per User-Agent Group", fontsize=26, pad=24)
ax.set_ylabel("Percentage", fontsize=20)
ax.set_xlabel("User-Agent Group", fontsize=20)
ax.set_xticks(range(len(parent_plot.index)))
ax.set_xticklabels(parent_plot.index, rotation=0, ha='center', fontsize=16)
ax.tick_params(axis='y', labelsize=16)

# Adjust legend to fit fully within the plot
leg = ax.legend(
    title="Parent Category",
    bbox_to_anchor=(1.006, 1),
    loc='upper left',
    fontsize=16,
    title_fontsize=18,
    borderaxespad=0
)

# Adjust layout more generously
plt.subplots_adjust(left=0.08, right=0.78, top=0.98, bottom=0.2)

# Save with tight bounding box to include legend fully
plt.savefig("../figs/uatax_parent.png", dpi=300, bbox_inches='tight')
plt.close()


'''
===============================================================================
4. Stacked Bar: UA Group vs Top-N Subcategories
===============================================================================
'''
# Select top N subcategories
top_subcats = ua_df.groupby("Subcategory")["count"].sum().index.tolist()
ua_top = ua_df[ua_df["Subcategory"].isin(top_subcats)]

subcat_plot = ua_top.pivot_table(index="ua_group", columns="Subcategory", values="percentage", fill_value=0)
subcat_plot = subcat_plot.loc[subcat_plot.sum(axis=1).sort_values(ascending=False).index]

fig, ax = plt.subplots(figsize=(12, 6))
subcat_plot.plot(kind="bar", stacked=True, ax=ax, colormap="RdYlBu")

# Custom two-line x-tick labels
label_mapping = {
    "python_lib": "python\nlib",
    "scanner_bot": "scanner\nbot",
    "cli_tool": "CLI\ntool",
    "missing": "missing",
    "custom_client": "custom\nclient",
    "other": "other",
    "browser": "browser"
}

xtick_labels = [label_mapping.get(label, label) for label in subcat_plot.index]



ax.set_title("Top Attack Subcategories per User-Agent Group", fontsize=24)
ax.set_ylabel("Percentage", fontsize=22)
ax.set_xlabel("User-Agent Group", fontsize=22)
ax.set_xticks(range(len(subcat_plot.index)))
ax.set_xticklabels(xtick_labels, rotation=0, ha='center', fontsize=20)
ax.tick_params(axis='y', labelsize=20)

ax.legend(title="Subcategory", bbox_to_anchor=(1.006, 1), loc='upper left', fontsize=20, title_fontsize=20, borderaxespad=0)
plt.subplots_adjust(left=0.08, right=0.78, top=0.98, bottom=0.2)
plt.savefig("../figs/uatax.png", dpi=300, bbox_inches='tight')
plt.close()

'''
===============================================================================
5. Stacked Bar: Cloud Org vs Parent Category
===============================================================================
'''

parent_pivot_org = org_df.groupby(["host_as_org", "Parent_Category"])["percentage"].sum().reset_index()
parent_plot_org = parent_pivot_org.pivot(index="host_as_org", columns="Parent_Category", values="percentage").fillna(0)
parent_plot_org = parent_plot_org.loc[parent_plot_org.sum(axis=1).sort_values(ascending=False).index]

org_label_map = {
    "AMAZON-02": "Cloud-A",
    "DIGITALOCEAN-ASN": "Cloud-D",
    "AMAZON-AES": "Cloud-B",
    "GOOGLE": "Cloud-C"
}

ordered_orgs = ["AMAZON-02", "AMAZON-AES", "GOOGLE", "DIGITALOCEAN-ASN"]
parent_plot_org = parent_plot_org.loc[ordered_orgs]

fig, ax = plt.subplots(figsize=(16, 8))
parent_plot_org.plot(kind="bar", stacked=True, ax=ax, colormap="RdYlBu")

ax.set_title("Attack Distribution by Parent Category per Cloud Provider", fontsize=24)
ax.set_ylabel("Percentage", fontsize=22)
ax.set_xlabel("Cloud Provider Org", fontsize=22)

display_labels = [org_label_map.get(org, org) for org in parent_plot_org.index]
ax.set_xticks(range(len(display_labels)))
ax.set_xticklabels(display_labels, rotation=0, ha='center', fontsize=18)
ax.tick_params(axis='y', labelsize=18)

ax.legend(title="Parent Category", bbox_to_anchor=(1.006, 1), loc='upper left', fontsize=16, title_fontsize=18, borderaxespad=0)
plt.subplots_adjust(left=0.08, right=0.78, top=0.98, bottom=0.2)
plt.savefig("../figs/cloudtax_parent.png", dpi=300, bbox_inches='tight')
plt.close()


'''
===============================================================================
6. Stacked Bar: Cloud Org vs Top-N Subcategories
===============================================================================
'''
top_subcats_org = org_df.groupby("Subcategory")["count"].sum().index.tolist()
org_top = org_df[org_df["Subcategory"].isin(top_subcats_org)]

subcat_plot_org = org_top.pivot_table(index="host_as_org", columns="Subcategory", values="percentage", fill_value=0)
subcat_plot_org = subcat_plot_org.loc[subcat_plot_org.sum(axis=1).sort_values(ascending=False).index]
subcat_plot_org = subcat_plot_org.loc[ordered_orgs]

fig, ax = plt.subplots(figsize=(12, 6))
subcat_plot_org.plot(kind="bar", stacked=True, ax=ax, colormap="RdYlBu")

ax.set_title("Top Attack Subcategories per Cloud Provider", fontsize=24)
ax.set_ylabel("Percentage", fontsize=22)
ax.set_xlabel("Cloud Provider Organization", fontsize=22)
display_labels = [org_label_map.get(org, org) for org in subcat_plot_org.index]
ax.set_xticks(range(len(display_labels)))
ax.set_xticklabels(display_labels, rotation=0, ha='center', fontsize=20)
ax.tick_params(axis='y', labelsize=20)

ax.legend(title="Subcategory", bbox_to_anchor=(1.006, 1), loc='upper left', fontsize=20, title_fontsize=20, borderaxespad=0)
plt.subplots_adjust(left=0.08, right=0.78, top=0.98, bottom=0.2)
plt.savefig("../figs/cloudtax.png", dpi=300, bbox_inches='tight')
plt.close()