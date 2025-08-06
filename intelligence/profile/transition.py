import pandas as pd
from collections import Counter

# === Load Dataset ===
df_top = pd.read_csv("../../data/top100/top100_ips_all_requests.csv")
df_top['startTime'] = pd.to_datetime(df_top['startTime'], errors='coerce')
df_top = df_top.sort_values(['clientIP', 'startTime'])

# === Multi-Escalation Sequence Tracker with Context ===
records = []

for ip, ip_df in df_top.groupby('clientIP'):
    ip_df = ip_df.sort_values('startTime')
    categories = ip_df['category'].tolist()
    times = ip_df['startTime'].tolist()

    for i in range(len(categories) - 1):
        from_cat = categories[i]
        to_cat = categories[i + 1]
        t1 = times[i]
        t2 = times[i + 1]

        if from_cat in ['scan', 'attempt', 'intrusion-control'] and to_cat in ['scan', 'attempt', 'intrusion-control'] and from_cat != to_cat:
            # Extract only rows within this escalation segment
            df_slice = ip_df[(ip_df['startTime'] >= t1) & (ip_df['startTime'] <= t2)]

            uris = df_slice['http_uri'].dropna().tolist()
            bodies = df_slice['http_body'].dropna().tolist()
            referers = df_slice['http_referer'].dropna().tolist()
            uas = df_slice['http_user_agent'].dropna().tolist()

            record = {
                "clientIP": ip,
                "from": from_cat,
                "to": to_cat,
                "startTime": t1,
                "endTime": t2,
                "duration_min": round((t2 - t1).total_seconds() / 60, 2),
                "top_uris": "; ".join([uri for uri, _ in Counter(uris).most_common(5)]),
                "top_bodies": "; ".join([b for b, _ in Counter(bodies).most_common(5)]),
                "top_referers": "; ".join([r for r, _ in Counter(referers).most_common(5)]),
                "user_agent": Counter(uas).most_common(1)[0][0] if uas else "unknown"
            }
            records.append(record)

# === Save Final Output ===
df_out = pd.DataFrame(records)
df_out.to_csv("escalation_segments_with_context.csv", index=False)
print("✓ Saved to 'escalation_segments_with_context.csv'")
