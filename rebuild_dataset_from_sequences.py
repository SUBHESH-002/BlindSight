"""
rebuild_dataset_from_sequences.py
==================================
Uses the OFFICIAL KTH 00sequences.txt frame labels to extract every
individual action clip from the local AVI files, then caps to exactly
200 samples per class.

Each KTH video file contains 4 distinct action repetitions with known
frame boundaries. This script extracts all of them (~400 per class)
then randomly samples down to the target of 200.

Usage (from the project root):
    python rebuild_dataset_from_sequences.py
"""

import os
import re
import urllib.request

import cv2
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR  = os.path.join(SCRIPT_DIR, "kth_videos")
OUTPUT_CSV = os.path.join(SCRIPT_DIR, "kth_temporal_features.csv")

TARGET_ACTIONS = {
    "boxing":       0,
    "walking":      1,
    "handclapping": 2,
}

FRAME_H  = 16
FRAME_W  = 16
N_FRAMES = 5
TARGET_PER_CLASS = 200

SEQUENCES_URL = "https://www.csc.kth.se/cvap/actions/00sequences.txt"


# ---------------------------------------------------------------------------
# 1. Download sequence labels
# ---------------------------------------------------------------------------
def download_sequences_txt(url=SEQUENCES_URL):
    print(f"Downloading KTH sequence labels from:\n  {url}")
    with urllib.request.urlopen(url, timeout=30) as resp:
        content = resp.read().decode("utf-8", errors="replace")
    print("  Downloaded OK.\n")
    return content


# ---------------------------------------------------------------------------
# 2. Parse 00sequences.txt
# ---------------------------------------------------------------------------
def parse_sequences(text):
    """
    Returns dict: { "person01_boxing_d1": [(1,95), (96,185), ...], ... }
    Skips *missing* entries.
    """
    result = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or "frames" not in line:
            continue
        parts = line.split("frames", 1)
        if len(parts) != 2:
            continue
        key = parts[0].strip()
        rng = parts[1].strip()
        if "*missing*" in rng:
            continue
        clips = [(int(m.group(1)), int(m.group(2)))
                 for m in re.finditer(r"(\d+)-(\d+)", rng)]
        if clips:
            result[key] = clips
    return result


# ---------------------------------------------------------------------------
# 3. Extract one clip's feature sequence
# ---------------------------------------------------------------------------
def extract_clip_features(video_path, start_frame, end_frame):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    s = max(0, start_frame - 1)          # KTH uses 1-based indices
    e = min(end_frame - 1, total - 1)

    if (e - s + 1) < N_FRAMES:
        cap.release()
        return None

    indices = np.linspace(s, e, N_FRAMES, dtype=int)
    seq = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return None
        # Simulated CCTV occlusion mask
        cv2.rectangle(frame, (60, 40), (100, 80), (0, 0, 0), -1)
        small = cv2.resize(frame, (FRAME_W, FRAME_H),
                           interpolation=cv2.INTER_AREA)
        seq.append((small / 255.0).tolist())

    cap.release()
    return seq if len(seq) == N_FRAMES else None


# ---------------------------------------------------------------------------
# 4. Build full dataset from all clips
# ---------------------------------------------------------------------------
def build_dataset(sequences):
    records = []
    action_counts = {a: 0 for a in TARGET_ACTIONS}
    missing = []

    for seq_key in sorted(sequences):
        clips = sequences[seq_key]
        parts = seq_key.split("_")
        if len(parts) < 3:
            continue
        action = parts[1]
        if action not in TARGET_ACTIONS:
            continue

        class_id = TARGET_ACTIONS[action]
        filename = f"{seq_key}_uncomp.avi"
        filepath = os.path.join(VIDEO_DIR, filename)

        if not os.path.isfile(filepath):
            missing.append(filename)
            continue

        for i, (start, end) in enumerate(clips):
            features = extract_clip_features(filepath, start, end)
            if features is None:
                continue
            records.append({
                "video_file":   f"{seq_key}_clip{i+1}",
                "target_class": class_id,
                "features":     features,
            })
            action_counts[action] += 1

    if missing:
        print(f"Warning: {len(missing)} AVI file(s) not found in kth_videos/")

    df = pd.DataFrame(records)
    print("Raw clips extracted per action:")
    for action, count in action_counts.items():
        print(f"  {action:<16}: {count}")
    return df


# ---------------------------------------------------------------------------
# 5. Cap to exactly TARGET_PER_CLASS per class
# ---------------------------------------------------------------------------
def cap_to_target(df, target=TARGET_PER_CLASS):
    parts = []
    for cls in sorted(df["target_class"].unique()):
        sub = df[df["target_class"] == cls]
        if len(sub) > target:
            sub = sub.sample(n=target, random_state=42)
        parts.append(sub)
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.chdir(SCRIPT_DIR)

    seq_text  = download_sequences_txt()
    sequences = parse_sequences(seq_text)
    print(f"Parsed {len(sequences)} video entries from 00sequences.txt\n")

    print(f"Extracting clips from: {VIDEO_DIR}")
    df = build_dataset(sequences)

    if df.empty:
        print("\nERROR: No samples were extracted.")
        print("Make sure kth_videos/ exists and contains the AVI files.")
        return

    print(f"\nTotal raw clips: {len(df)}")

    df_final = cap_to_target(df, target=TARGET_PER_CLASS)

    print(f"\nFinal dataset (capped to {TARGET_PER_CLASS} per class):")
    id_to_action = {v: k for k, v in TARGET_ACTIONS.items()}
    counts = df_final["target_class"].value_counts().sort_index()
    for cls_id, cnt in counts.items():
        print(f"  Class {cls_id} ({id_to_action[cls_id]:<14}): {cnt}")
    print(f"  Total: {len(df_final)}")

    df_final.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved -> {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
