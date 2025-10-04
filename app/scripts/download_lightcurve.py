import pandas as pd
import requests

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
        print(df.head())
        return df
    else:
        print(f"No data returned for star {star_id}, TCE {tce}")
        return None

def download_from_koi(csv_path="data/cumulative.csv"):
    df = pd.read_csv(csv_path)
    
    # Loop over each row
    for idx, row in df[:1].iterrows():
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
        
        # Optional: save each table to CSV
        if star_df is not None:
            star_df.to_csv(f"output/star_{star_id}_tce_{tce}.csv", index=False)
            print(f"Saved output/star_{star_id}_tce_{tce}.csv")

# Example call
download_from_koi("data/cumulative.csv")
