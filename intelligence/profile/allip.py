import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

# Load your file (update path if needed)
df = pd.read_csv("escalation_segments_with_context.csv")
df['startTime'] = pd.to_datetime(df['startTime'])
df['endTime'] = pd.to_datetime(df['endTime'])

# Sort and prepare
df = df.sort_values(by=["clientIP", "startTime"])

# Color map for escalation types
color_map = {
    'scan': '#299d8f',
    'attempt': '#f3a361',
    'intrusion-control': '#e66d50'
}

# Assign y-position to each IP for vertical offset
unique_ips = df['clientIP'].unique()
ip_ypos = {ip: i for i, ip in enumerate(unique_ips)}

def mask_ip(ip):
    parts = ip.split(".")
    if len(parts) == 4:
        return f"xx.{parts[1]}.{parts[2]}.xx"
    elif ":" in ip:  # IPv6 fallback
        parts = ip.split(":")
        return f"xx:{parts[1]}:{parts[2]}:xx"
    else:
        return "xx.xx.xx.xx"


# Plot
fig, ax = plt.subplots(figsize=(20, len(unique_ips) * 0.5))

for _, row in df.iterrows():
    y = ip_ypos[row['clientIP']]
    color = color_map.get(row['from'], 'black')
    ax.plot([row['startTime'], row['endTime']], [y, y], color=color, linewidth=5, solid_capstyle='round')

# Y-axis settings
masked_labels = [mask_ip(ip) for ip in ip_ypos.keys()]
ax.set_yticks(list(ip_ypos.values()))
ax.set_yticklabels(masked_labels, fontsize=20)
ax.set_ylabel("Client IP", fontsize=22)

# X-axis: use daily ticks
ax.xaxis.set_major_locator(mdates.DayLocator(interval=14))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
plt.xticks(rotation=0, fontsize=20)
ax.set_xlabel("Time", fontsize=22)

# Legend
legend_elements = [
    Line2D([0], [0], color='#299d8f', lw=6, label='scan'),
    Line2D([0], [0], color='#f3a361', lw=6, label='attempt'),
    Line2D([0], [0], color='#e66d50', lw=6, label='intrusion-control')
]
ax.legend(
    handles=legend_elements, 
    loc='upper center', 
    bbox_to_anchor=(0.5, 1.02), 
    ncol=3, 
    frameon=False,
    fontsize=20)

#plt.title("Escalation Timelines per IP", fontsize=12)
plt.tight_layout()
plt.subplots_adjust(bottom=0.25)
plt.savefig("escalation_timelines_per_ip.pdf", bbox_inches='tight')
