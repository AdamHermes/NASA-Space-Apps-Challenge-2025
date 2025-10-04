import pandas as pd
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import threading


def download(star_id="11446443", mission="kepler", tce=1):
    # Construct the URL for the GET request
    url = f"https://exo.mast.stsci.edu/api/v0.1/dvdata/{mission}/{star_id}/table?tce={tce}"
    response = requests.get(url)
    print("Requesting:", url)

    if response.status_code != 200:
        print(f"Error {response.status_code} for star {star_id}, TCE {tce}")
        return None

    data = response.json()

    # Convert JSON table to DataFrame if data exists
    if 'data' in data:
        df = pd.DataFrame(data['data'])
        # print(df.head())
        return df
    else:
        print(f"No data returned for star {star_id}, TCE {tce}")
        return None

def download_from_koi(csv_path="data/cumulative.csv", out_dir="data/lightcurve"):
    df = pd.read_csv(csv_path)
    df_filtered = df[df['koi_disposition'] != "FALSE POSITIVE"]
    print(f"Number to download: {len(df_filtered)}")

    # Loop over each row
    for idx, row in df[:1].iterrows():
        start = time.time()
        star_id = str(row['kepid'])
        
        # Extract TCE from koi_name (after the dot)
        koi_name = str(row['kepoi_name'])
        try:
            tce = int(koi_name.split('.')[-1])
        except:
            print(f"Could not extract TCE from {koi_name}, skipping.")
            continue
        
        # Call download function
        star_df = download(star_id=star_id, mission="kepler", tce=tce)
        end = time.time()
        # Optional: save each table to CSV
        if star_df is not None:
            star_df.to_csv(f"{out_dir}/star_{star_id}_tce_{tce}.csv", index=False)
            print(f"Saved {out_dir}/star_{star_id}_tce_{tce}.csv | time: {(end - start):.4f}")

def download_from_koi_multi(csv_path="data/cumulative.csv", out_dir="data/lightcurve", max_workers=5):
    df = pd.read_csv(csv_path)
    df_filtered = df[df['koi_disposition'] != "FALSE POSITIVE"]
    print(f"Number of stars to download: {len(df_filtered)}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total = len(df_filtered)
    counter = 0
    counter_lock = threading.Lock()
    stop_event = threading.Event()

    def process_row(row):
        nonlocal counter
        if stop_event.is_set():
            return None
        
        start = time.time()
        star_id = str(row['kepid'])
        koi_name = str(row['kepoi_name'])
        try:
            tce = int(koi_name.split('.')[-1])
        except:
            print(f"Could not extract TCE from {koi_name}, skipping.")
            with counter_lock:
                counter += 1
                print(f"Progress: {counter}/{total} ({counter/total*100:.1f}%)")
            return None
        
        file_path = out_dir / f"star_{star_id}_tce_{tce}.csv"
        if file_path.exists():
            print(f"File already exists, skipping: {file_path}")
            with counter_lock:
                counter += 1
                print(f"Progress: {counter}/{total} ({counter/total*100:.1f}%)")
            return star_id

        star_df = download(star_id=star_id, mission="kepler", tce=tce)

        if stop_event.is_set():  # check again after download
            return None
        
        if star_df is not None:
            star_df.to_csv(file_path, index=False)
            end = time.time()
            print(f"Saved {file_path} | time: {(end - start):.2f}s")
        
        # Update progress counter
        with counter_lock:
            counter += 1
            if counter % 100 == 0:
                print(f"Progress: {counter}/{total} ({counter/total*100:.1f}%)")
        return star_id

    # Use ThreadPoolExecutor to parallelize downloads
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_row, row) for idx, row in df_filtered[:].iterrows()]
        try:
            for future in as_completed(futures):
                _ = future.result()
        except KeyboardInterrupt:
            # Set the stop_event to signal threads to exit
            stop_event.set()
            print("Download cancelled by user. Waiting for threads to exit...")
            # Optionally wait for all threads to finish
            for f in futures:
                f.cancel()

download_from_koi_multi(csv_path="data/cumulative.csv", out_dir="data/lightcurve", max_workers=100)

# download_from_koi("data/cumulative.csv")
