import os
import pandas as pd
from datetime import datetime, timedelta

# === XPOT Version Paths ===
INPUT_FOLDER = "../data/xpot/daily"          # where your daily XPOT csvs are
OUTPUT_FOLDER = "../data/xpot/sequences"     # where you want 1-hour window csvs
WINDOW_SIZE_MINUTES = 60

# === Optionally set date range (update as needed) ===
START_DATE = "2025-05-01"  # inclusive
END_DATE   = "2025-05-30"  # inclusive

for file in sorted(os.listdir(INPUT_FOLDER)):
    if not file.endswith(".csv"):
        continue

    # Extract date from filename (edit if XPOT files have different naming!)
    day = file.split("_")[-1].replace(".csv", "")  # e.g., "2025-06-10"
    if not (START_DATE <= day <= END_DATE):
        continue

    print(f"Processing {day}")

    df = pd.read_csv(os.path.join(INPUT_FOLDER, file))
    df["startTime"] = pd.to_datetime(df["startTime"], errors="coerce")
    df = df.dropna(subset=["startTime"])
    df = df.sort_values("startTime")

    if df.empty:
        continue

    start_ts = df["startTime"].min().replace(minute=0, second=0)
    end_ts = df["startTime"].max()

    output_day_dir = os.path.join(OUTPUT_FOLDER, day)
    os.makedirs(output_day_dir, exist_ok=True)

    window_idx = 0
    current_start = start_ts

    while current_start <= end_ts:
        current_end = current_start + timedelta(minutes=WINDOW_SIZE_MINUTES)
        window_df = df[
            (df["startTime"] >= current_start) &
            (df["startTime"] < current_end)
        ]
        if not window_df.empty:
            out_file = os.path.join(
                output_day_dir,
                f"window_{current_start.strftime('%H_%M')}_{current_end.strftime('%H_%M')}.csv"
            )
            window_df.to_csv(out_file, index=False)
        current_start = current_end
        window_idx += 1

print("XPOT daily → 1-hour window sequence splitting done.")
