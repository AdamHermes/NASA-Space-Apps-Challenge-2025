import os
import re
import shutil
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import difflib

def prepare_lightcurve_dataset(lightcurve_dir, cumulative_path, out_dir="dataset", test_size=0.2):
    """
    Reformat Kepler light curve CSVs into structured train/test folders
    based on labels in cumulative.csv.

    Parameters
    ----------
    lightcurve_dir : str or Path
        Directory containing individual star_XXX_tce_YYY.csv files.
    cumulative_path : str or Path
        Path to cumulative.csv with kepid, kepoi_name, and koi_disposition columns.
    out_dir : str or Path
        Output directory to store train/test splits.
    test_size : float
        Fraction of samples to reserve for testing.
    """

    lightcurve_dir = Path(lightcurve_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ======================
    # 1. Load cumulative data
    # ======================
    df = pd.read_csv(cumulative_path, comment="#")
    df = df[df["koi_disposition"] != "FALSE POSITIVE"]
    print(f"[INFO] Loaded cumulative file with {len(df)} entries.")

    # Normalize naming pattern (e.g. K00757450.01 → star_757450_tce_1)
    def parse_star_tce(kepoi_name):
        match = re.match(r"K0*([0-9]+)\.0*([0-9]+)", kepoi_name)
        if match:
            star_id, tce_num = match.groups()
            return int(tce_num)
        return None

    df["star_id"], df["tce_num"] = df["kepid"], df["kepoi_name"].map(parse_star_tce)

    # ======================
    # 2. Create label column
    # ======================
    def label_from_disposition(x):
        # Adjust according to your needs
        if x.strip().upper() in ["CONFIRMED"]:
            return "0"
        elif x.strip().upper() in ["CANDIDATE"]:
            return "1"
        else:
            print("Cannot find label for", x)
            return -1

    df["label"] = df["koi_disposition"].apply(label_from_disposition)

    # ======================
    # 3. Collect available files
    # ======================
    all_files = list(lightcurve_dir.glob("star_*_tce_*.csv"))
    print(f"[INFO] Found {len(all_files)} light curve CSVs.")

    matched = []
    unmatched = []

    # Prepare a set of all valid (star_id, tce) pairs from cumulative.csv
    valid_pairs = set(zip(df["star_id"], df["tce_num"]))

    for f in all_files:
        m = re.search(r"star_(\d+)_tce_(\d+)", f.name)
        if not m:
            continue

        sid, tce = int(m.group(1)), int(m.group(2))

        if (sid, tce) in valid_pairs:
            label = df.loc[(df["star_id"] == sid) & (df["tce_num"] == tce), "label"].iloc[0]
            matched.append((f, label))
        else:
            unmatched.append(f.name)

    print(f"[INFO] Matched {len(matched)} CSV files with cumulative.csv")
    print(f"[DEBUG] Unmatched files: {len(unmatched)}")

    # Print a few examples
    if unmatched:
        print("[DEBUG] First 10 unmatched filenames:")
        for name in unmatched[:10]:
            print("   ", name)

        # Optional: show possible close matches based on star id only
        print("\n[DEBUG] Checking a few for close matches in cumulative.csv IDs:")
        unmatched_ids = [int(re.search(r"star_(\d+)", f).group(1)) for f in unmatched[:5]]
        for uid in unmatched_ids:
            closest = difflib.get_close_matches(str(uid), df["star_id"].astype(str).tolist(), n=3)
            print(f"   {uid} → close matches in cumulative: {closest}")

    # ======================
    # 4. Split into train/test
    # ======================
    files, labels = zip(*matched)
    train_files, test_files, train_labels, test_labels = train_test_split(
        files, labels, test_size=test_size, stratify=labels, random_state=42
    )

    def copy_to_folder(subset, labels, split):
        for src, lbl in zip(subset, labels):
            dest = out_dir / split / lbl
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest / src.name)

    print("[INFO] Copying training files...")
    copy_to_folder(train_files, train_labels, "train")

    print("[INFO] Copying testing files...")
    copy_to_folder(test_files, test_labels, "test")

    print(f"[DONE] Dataset prepared under '{out_dir}'")
    print(f"Train samples: {len(train_files)}, Test samples: {len(test_files)}")

    return out_dir


lightcurve_dir = "data/lightcurve"
cumulative_path = "data/cumulative.csv"
outdir = "data/lc_dataset"

prepare_lightcurve_dataset(
    lightcurve_dir=lightcurve_dir,
    cumulative_path=cumulative_path,
    out_dir=outdir
)